use crate::descriptors::Descriptors;
use crate::globals::Globals;
use fnv::FnvHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, Mutex, Weak};

type PoolInner<T> = Mutex<Vec<T>>;
type Constructor<Type, Description> = Box<dyn Fn(&Descriptors, &Description) -> Type>;

#[derive(Debug, Default)]
pub struct TexturePool {
    pools: FnvHashMap<TextureKey, BufferPool<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible>>,
    globals_cache: FnvHashMap<GlobalsKey, Arc<Globals>>,
}

impl TexturePool {
    pub fn new() -> Self {
        Default::default()
    }

    /// Trim each sub-pool so that no more than `max_per_key` idle textures are
    /// held per distinct `(size, format, usage, sample_count)` key, and halve
    /// the globals cache once it exceeds `MAX_GLOBALS` entries.
    ///
    /// Call this at the end of every frame instead of replacing the pool with a
    /// fresh `TexturePool::new()`.  This keeps GPU textures alive across frames
    /// so that the next frame can reuse them, while bounding peak memory use.
    ///
    /// When the globals cache exceeds the limit, half of its entries are dropped
    /// rather than clearing everything at once, avoiding a cliff-edge frame spike
    /// for content that rotates through more than `MAX_GLOBALS` configurations.
    pub fn purge_if_oversized(&mut self) {
        const MAX_PER_KEY: usize = 4;
        const MAX_GLOBALS: usize = 16;

        for pool in self.pools.values_mut() {
            pool.purge_excess(MAX_PER_KEY);
        }
        let globals_len = self.globals_cache.len();
        if globals_len > MAX_GLOBALS {
            let keep = MAX_GLOBALS / 2;
            let to_remove = globals_len.saturating_sub(keep);
            let keys_to_remove: Vec<_> = self.globals_cache.keys().take(to_remove).cloned().collect();
            for key in keys_to_remove {
                self.globals_cache.remove(&key);
            }
        }
    }

    pub fn get_texture(
        &mut self,
        descriptors: &Descriptors,
        size: wgpu::Extent3d,
        usage: wgpu::TextureUsages,
        format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> PoolEntry<(wgpu::Texture, wgpu::TextureView), AlwaysCompatible> {
        let key = TextureKey {
            size,
            usage,
            format,
            sample_count,
        };
        let pool = self.pools.entry(key).or_insert_with(|| {
            let label = if cfg!(feature = "render_debug_labels") {
                use std::sync::atomic::{AtomicU32, Ordering};
                static ID_COUNT: AtomicU32 = AtomicU32::new(0);
                let id = ID_COUNT.fetch_add(1, Ordering::Relaxed);
                create_debug_label!("Pooled texture {}", id)
            } else {
                None
            };
            BufferPool::new(Box::new(move |descriptors, _description| {
                let texture = descriptors.device.create_texture(&wgpu::TextureDescriptor {
                    label: label.as_deref(),
                    size,
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    view_formats: &[format],
                    usage,
                });
                let view = texture.create_view(&Default::default());
                (texture, view)
            }))
        });
        pool.take(descriptors, AlwaysCompatible)
    }

    pub fn get_globals(
        &mut self,
        descriptors: &Descriptors,
        viewport_width: u32,
        viewport_height: u32,
    ) -> Arc<Globals> {
        self.globals_cache
            .entry(GlobalsKey {
                viewport_width,
                viewport_height,
            })
            .or_insert_with(|| {
                Arc::new(Globals::new(
                    &descriptors.device,
                    &descriptors.bind_layouts.globals,
                    viewport_width,
                    viewport_height,
                ))
            })
            .clone()
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct TextureKey {
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    format: wgpu::TextureFormat,
    sample_count: u32,
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
struct GlobalsKey {
    viewport_width: u32,
    viewport_height: u32,
}

pub trait BufferDescription: Clone + Debug {
    type Cost: Ord;

    /// If the potential buffer represented by this description (`self`)
    /// fits another existing buffer and its description (`other`),
    /// return the cost to use that buffer instead of making a new one.
    ///
    /// Cost is an arbitrary unit, but lower is better.
    /// None means that the other buffer cannot be used in place of this one.
    fn cost_to_use(&self, other: &Self) -> Option<Self::Cost>;
}

#[derive(Clone, Debug)]
pub struct AlwaysCompatible;

impl BufferDescription for AlwaysCompatible {
    type Cost = ();

    fn cost_to_use(&self, _other: &Self) -> Option<()> {
        Some(())
    }
}

pub struct BufferPool<Type, Description: BufferDescription> {
    available: Arc<PoolInner<(Type, Description)>>,
    constructor: Constructor<Type, Description>,
}

impl<Type, Description: BufferDescription> Debug for BufferPool<Type, Description> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish()
    }
}

impl<Type, Description: BufferDescription> BufferPool<Type, Description> {
    pub fn new(constructor: Constructor<Type, Description>) -> Self {
        Self {
            available: Arc::new(Mutex::new(vec![])),
            constructor,
        }
    }

    /// Discard any idle pool entries beyond `max_count`, releasing their GPU
    /// memory.  Entries are removed from the front of the Vec (oldest first),
    /// so the most-recently returned ones (at the back) are retained as they
    /// are most likely to be reused.
    pub fn purge_excess(&self, max_count: usize) {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let len = guard.len();
        if len > max_count {
            guard.drain(..len - max_count);
        }
    }

    pub fn take(
        &self,
        descriptors: &Descriptors,
        description: Description,
    ) -> PoolEntry<Type, Description> {
        let mut guard = self
            .available
            .lock()
            .expect("Should not be able to lock recursively");
        let mut best: Option<(Description::Cost, usize)> = None;
        for i in 0..guard.len() {
            if let Some(cost) = description.cost_to_use(&guard[i].1) {
                if let Some(best) = &mut best {
                    if best.0 > cost {
                        *best = (cost, i);
                    }
                } else if best.is_none() {
                    best = Some((cost, i));
                }
            }
        }

        let (item, used_description) = if let Some((_, best)) = best {
            guard.swap_remove(best)
        } else {
            let item = (self.constructor)(descriptors, &description);
            (item, description)
        };
        PoolEntry {
            item: Some(item),
            description: used_description,
            pool: Arc::downgrade(&self.available),
        }
    }
}

pub struct PoolEntry<Type, Description: BufferDescription> {
    item: Option<Type>,
    description: Description,
    pool: Weak<PoolInner<(Type, Description)>>,
}

impl<Type, Description: BufferDescription> Debug for PoolEntry<Type, Description>
where
    Type: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PoolEntry").field(&self.item).finish()
    }
}

impl<Type, Description: BufferDescription> Drop for PoolEntry<Type, Description> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Some(pool) = self.pool.upgrade()
        {
            pool.lock()
                .expect("Should not be able to lock recursively")
                .push((item, self.description.clone()))
        }
    }
}

impl<Type, Description: BufferDescription> Deref for PoolEntry<Type, Description> {
    type Target = Type;

    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("Item should exist until dropped")
    }
}

// ── Vertex-instance buffer pool ───────────────────────────────────────────────
//
// Keeps a pool of `VERTEX | COPY_DST` GPU buffers that are reused across frames.
// On every frame each consumed buffer is returned to the pool when its owning
// `PooledVertexBuffer` smart-pointer is dropped; the next frame can then
// reacquire that same GPU allocation instead of calling `device.create_buffer`.
//
// Cross-frame safety:
//   New data is written via `queue.write_buffer`, which schedules an
//   internal copy command.  Because all submissions go to the same queue the
//   copy in frame N+1 is guaranteed to execute after frame N's vertex reads,
//   so there is no GPU-side race condition.

type VertexPoolInner = Mutex<Vec<wgpu::Buffer>>;

/// A pool of reusable `VERTEX | COPY_DST` GPU buffers for instance data.
///
/// Use [`VertexInstancePool::acquire`] to obtain a [`PooledVertexBuffer`];
/// the buffer is automatically returned to the pool when it is dropped.
#[derive(Debug, Clone, Default)]
pub struct VertexInstancePool {
    inner: Arc<VertexPoolInner>,
    /// Running count of new GPU buffer allocations this frame (i.e. cases
    /// where no pooled buffer was available and a fresh one was created).
    /// Read and reset via [`VertexInstancePool::take_alloc_count`].
    alloc_count: Arc<std::sync::atomic::AtomicU32>,
}

impl VertexInstancePool {
    pub fn new() -> Self {
        Default::default()
    }

    /// Return a buffer whose `size()` is ≥ `min_bytes`.
    ///
    /// If the pool holds a suitable buffer it is recycled, preferring the
    /// smallest one that fits (best-fit) to minimise wasted GPU memory.
    /// If no suitable buffer exists, a new one is created whose capacity is
    /// `min_bytes` rounded up to the next power of two (at least 256 bytes).
    pub fn acquire(&self, device: &wgpu::Device, min_bytes: u64) -> PooledVertexBuffer {
        let buffer = {
            let mut guard = self.inner.lock().expect("vertex pool lock");
            // Find the smallest pooled buffer that is large enough (best-fit).
            let best = guard
                .iter()
                .enumerate()
                .filter(|(_, b)| b.size() >= min_bytes)
                .min_by_key(|(_, b)| b.size())
                .map(|(i, _)| i);
            if let Some(pos) = best {
                guard.swap_remove(pos)
            } else {
                let size = min_bytes.next_power_of_two().max(256);
                // Count this as a new GPU allocation.
                self.alloc_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }
        };
        PooledVertexBuffer {
            buffer: Some(buffer),
            pool: Arc::clone(&self.inner),
        }
    }

    /// Discard pooled buffers in excess of `MAX_POOLED` to bound memory use.
    pub fn purge_if_oversized(&self) {
        const MAX_POOLED: usize = 32;
        let mut guard = self.inner.lock().expect("vertex pool lock");
        if guard.len() > MAX_POOLED {
            guard.truncate(MAX_POOLED);
        }
    }

    /// Return the number of new GPU buffer allocations since the last call to
    /// this method, and reset the counter to zero.
    ///
    /// Call once per frame (after submission) to track allocation pressure.
    pub fn take_alloc_count(&self) -> u32 {
        self.alloc_count
            .swap(0, std::sync::atomic::Ordering::Relaxed)
    }
}

/// A smart-pointer wrapping a `VERTEX | COPY_DST` GPU buffer.
///
/// When dropped, the underlying `wgpu::Buffer` is returned to the
/// [`VertexInstancePool`] it was acquired from so that it can be reused
/// in the next frame.
#[derive(Debug)]
pub struct PooledVertexBuffer {
    buffer: Option<wgpu::Buffer>,
    pool: Arc<VertexPoolInner>,
}

impl std::ops::Deref for PooledVertexBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("buffer not yet taken")
    }
}

impl Drop for PooledVertexBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            if let Ok(mut guard) = self.pool.lock() {
                guard.push(buf);
            }
            // If the lock is poisoned (shouldn't happen) the buffer is simply dropped.
        }
    }
}
