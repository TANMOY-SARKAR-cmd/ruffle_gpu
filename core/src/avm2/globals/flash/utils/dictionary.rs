use crate::avm2::activation::Activation;
use crate::avm2::value::Value;
use crate::avm2::Error;

pub use crate::avm2::object::dictionary_allocator;

/// Implements `Dictionary.initWeakKeys()` – a private native method called by
/// the AS3 constructor when `weakKeys=true`.  It switches the dictionary to
/// weak-reference key storage so that object keys do not prevent GC collection.
pub fn init_weak_keys<'gc>(
    activation: &mut Activation<'_, 'gc>,
    this: Value<'gc>,
    _args: &[Value<'gc>],
) -> Result<Value<'gc>, Error<'gc>> {
    if let Some(dict) = this
        .as_object()
        .and_then(|o| o.as_dictionary_object())
    {
        dict.enable_weak_keys(activation.gc());
    }
    Ok(Value::Undefined)
}
