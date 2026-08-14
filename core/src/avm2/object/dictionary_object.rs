//! Object representation for `flash.utils.Dictionary`

use crate::avm2::Error;
use crate::avm2::activation::Activation;
use crate::avm2::dynamic_map::DynamicKey;
use crate::avm2::object::WeakObject;
use crate::avm2::object::script_object::ScriptObjectData;
use crate::avm2::object::{ClassObject, Object, TObject};
use crate::avm2::value::Value;
use crate::string::AvmString;
use core::fmt;
use gc_arena::barrier::unlock;
use gc_arena::{Collect, Gc, GcWeak, Mutation, lock::RefLock};
use ruffle_common::utils::HasPrefixField;
use std::cell::Cell;

/// Indices returned by `get_next_enumerant` that are >= `WEAK_OFFSET` encode positions
/// in the weak-object-entries list so they never collide with base-map indices (which are
/// small sequential integers starting at 1).
///
/// The value must be representable as a positive `i32` because the AVM2
/// `hasnext2` opcode treats negative indices as "end of iteration".
/// 0x4000_0000 leaves room for ~1 billion weak entries before wrapping into
/// the negative i32 range, while staying well above any realistic base-map size.
const WEAK_OFFSET: u32 = 0x4000_0000;

/// A class instance allocator that allocates Dictionary objects.
pub fn dictionary_allocator<'gc>(
    class: ClassObject<'gc>,
    activation: &mut Activation<'_, 'gc>,
) -> Result<Object<'gc>, Error<'gc>> {
    let base = ScriptObjectData::new(class);

    Ok(DictionaryObject(Gc::new(
        activation.gc(),
        DictionaryObjectData {
            base,
            weak_keys: Cell::new(false),
            weak_object_entries: RefLock::new(Vec::new()),
        },
    ))
    .into())
}

/// An object that allows associations between objects and values.
///
/// This is implemented by way of "object space", parallel to the property
/// space that ordinary properties live in. This space has no namespaces, and
/// keys are objects instead of strings.
#[derive(Clone, Collect, Copy)]
#[collect(no_drop)]
pub struct DictionaryObject<'gc>(pub Gc<'gc, DictionaryObjectData<'gc>>);

#[derive(Clone, Collect, Copy, Debug)]
#[collect(no_drop)]
pub struct DictionaryObjectWeak<'gc>(pub GcWeak<'gc, DictionaryObjectData<'gc>>);

impl fmt::Debug for DictionaryObject<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DictionaryObject")
            .field("ptr", &Gc::as_ptr(self.0))
            .finish()
    }
}

#[derive(Clone, Collect, HasPrefixField)]
#[collect(no_drop)]
#[repr(C, align(8))]
pub struct DictionaryObjectData<'gc> {
    /// Base script object
    base: ScriptObjectData<'gc>,
    /// Whether object keys are held as weak GC references.
    weak_keys: Cell<bool>,
    /// Weak-reference storage for object keys and their associated values.
    /// Only consulted when `weak_keys` is true; string/uint keys continue to
    /// live in the base `ScriptObjectData` map.
    ///
    /// Each slot is `(weak_key, value)` where `value` is `None` for an
    /// explicitly deleted entry.  Slots whose key has been collected by the
    /// GC (`weak_key.upgrade()` returns `None`) are also treated as absent.
    weak_object_entries: RefLock<Vec<(WeakObject<'gc>, Option<Value<'gc>>)>>,
}

impl<'gc> DictionaryObject<'gc> {
    /// Switch this dictionary to weak-key mode.
    /// Should be called at most once, immediately after construction.
    pub fn enable_weak_keys(self, mc: &Mutation<'gc>) {
        // `weak_keys` is a plain `Cell<bool>` that holds no GC pointers, so
        // no write-barrier is required.  We still accept `mc` to signal to the
        // caller that a mutable context is needed.
        let _ = mc;
        self.0.weak_keys.set(true);
    }

    fn is_weak_keys(self) -> bool {
        self.0.weak_keys.get()
    }

    /// Find the encoded enumerant index of the next live weak entry at or
    /// after array position `start_pos` (0-based).
    /// Returns `WEAK_OFFSET + pos + 1`, or `0` if there are no more live entries.
    fn next_live_weak_entry(self, start_pos: usize, mc: &Mutation<'gc>) -> u32 {
        let entries = self.0.weak_object_entries.borrow();
        for (pos, (weak_key, value)) in entries.iter().enumerate().skip(start_pos) {
            if value.is_some() && weak_key.upgrade(mc).is_some() {
                return WEAK_OFFSET + pos as u32 + 1;
            }
        }
        0
    }

    /// Retrieve a value in the dictionary's object space.
    pub fn get_property_by_object(self, name: Object<'gc>) -> Value<'gc> {
        if self.is_weak_keys() {
            // `name` is a live strong reference, so pointer comparison is safe
            // within the same GC epoch (the GC cannot run while `name` is held).
            let name_ptr = name.as_ptr();
            let entries = self.0.weak_object_entries.borrow();
            for (weak_key, value) in entries.iter() {
                if weak_key.as_ptr() == name_ptr {
                    return value.unwrap_or(Value::Undefined);
                }
            }
            Value::Undefined
        } else {
            self.base()
                .values()
                .get(&DynamicKey::Object(name))
                .map(|v| v.value)
                .unwrap_or(Value::Undefined)
        }
    }

    /// Set a value in the dictionary's object space.
    pub fn set_property_by_object(self, name: Object<'gc>, value: Value<'gc>, mc: &Mutation<'gc>) {
        if self.is_weak_keys() {
            let name_ptr = name.as_ptr();
            let mut entries = unlock!(
                Gc::write(mc, self.0),
                DictionaryObjectData,
                weak_object_entries
            )
            .borrow_mut();
            // Update an existing live entry if present.
            for (weak_key, existing_value) in entries.iter_mut() {
                if weak_key.as_ptr() == name_ptr && existing_value.is_some() {
                    *existing_value = Some(value);
                    return;
                }
            }
            // Append a new weak entry (positions are never shifted so that
            // ongoing `for..in` iterations remain valid).
            entries.push((name.downgrade(), Some(value)));
        } else {
            self.base()
                .values_mut(mc)
                .insert(DynamicKey::Object(name), value);
        }
    }

    /// Delete a value from the dictionary's object space.
    pub fn delete_property_by_object(self, name: Object<'gc>, mc: &Mutation<'gc>) {
        if self.is_weak_keys() {
            let name_ptr = name.as_ptr();
            let mut entries = unlock!(
                Gc::write(mc, self.0),
                DictionaryObjectData,
                weak_object_entries
            )
            .borrow_mut();
            // Mark the entry as deleted without shifting positions.
            for (weak_key, value) in entries.iter_mut() {
                if weak_key.as_ptr() == name_ptr && value.is_some() {
                    *value = None;
                    return;
                }
            }
        } else {
            self.base().values_mut(mc).remove(&DynamicKey::Object(name));
        }
    }

    pub fn has_property_by_object(self, name: Object<'gc>) -> bool {
        if self.is_weak_keys() {
            let name_ptr = name.as_ptr();
            let entries = self.0.weak_object_entries.borrow();
            entries
                .iter()
                .any(|(w, v)| v.is_some() && w.as_ptr() == name_ptr)
        } else {
            self.base()
                .values()
                .contains_key(&DynamicKey::Object(name))
        }
    }
}

impl<'gc> TObject<'gc> for DictionaryObject<'gc> {
    fn gc_base(&self) -> Gc<'gc, ScriptObjectData<'gc>> {
        HasPrefixField::as_prefix_gc(self.0)
    }

    // Calling `setPropertyIsEnumerable` on a `Dictionary` has no effect -
    // stringified properties are always enumerable.
    fn set_local_property_is_enumerable(
        &self,
        _mc: &Mutation<'gc>,
        _name: AvmString<'gc>,
        _is_enumerable: bool,
    ) {
    }

    fn get_next_enumerant(
        self,
        last_index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<u32, Error<'gc>> {
        if !self.is_weak_keys() {
            // Non-weak: all keys (including object keys) live in the base map.
            let base = self.base();
            return Ok(base.get_next_enumerant(last_index));
        }

        let mc = activation.gc();
        if last_index < WEAK_OFFSET {
            // Phase 1: iterate string/uint keys from the base map.
            let next_base = self.base().get_next_enumerant(last_index);
            if next_base != 0 {
                return Ok(next_base);
            }
            // Base map exhausted – transition to weak object entries.
            Ok(self.next_live_weak_entry(0, mc))
        } else {
            // Phase 2: iterate weak object entries.
            // `last_index - WEAK_OFFSET` is the 1-based position; the next
            // position to search starts right after it.
            let next_pos = (last_index - WEAK_OFFSET) as usize;
            Ok(self.next_live_weak_entry(next_pos, mc))
        }
    }

    fn get_enumerant_name(
        self,
        index: u32,
        activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if !self.is_weak_keys() || index < WEAK_OFFSET {
            // Non-weak dict or base-map index: delegate to default behaviour.
            let base = self.base();
            return Ok(base.get_enumerant_name(index).unwrap_or(Value::Null));
        }
        // Weak entry index: decode array position and return the key object.
        let pos = (index - WEAK_OFFSET - 1) as usize;
        let entries = self.0.weak_object_entries.borrow();
        if let Some((weak_key, value)) = entries.get(pos) {
            if value.is_some() {
                if let Some(key_obj) = weak_key.upgrade(activation.gc()) {
                    return Ok(Value::Object(key_obj));
                }
            }
        }
        Ok(Value::Null)
    }

    fn get_enumerant_value(
        self,
        index: u32,
        _activation: &mut Activation<'_, 'gc>,
    ) -> Result<Value<'gc>, Error<'gc>> {
        if !self.is_weak_keys() || index < WEAK_OFFSET {
            // Non-weak dict or base-map index.
            return Ok(*self
                .base()
                .values()
                .value_at(index as usize)
                .unwrap_or(&Value::Undefined));
        }
        // Weak entry index.
        let pos = (index - WEAK_OFFSET - 1) as usize;
        let entries = self.0.weak_object_entries.borrow();
        Ok(entries
            .get(pos)
            .and_then(|(_, v)| *v)
            .unwrap_or(Value::Undefined))
    }
}
