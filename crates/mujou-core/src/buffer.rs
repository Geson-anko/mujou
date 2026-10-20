//! Buffers and messages: the immutable values that slots hold.
//!
//! A message is immutable and shared as `Arc<Message>`, so a reader that holds one
//! sees it unchanged however often the writer writes. The writer never writes into
//! a buffer a reader may see: it takes a buffer from a pool, fills it, and builds a
//! new message. A buffer goes back to its pool when the last message (or lent
//! tensor) that holds it is dropped, so in a steady state writing does not
//! allocate.
//!
//! A message may carry data along with its values: the carry fields of its
//! [`Schema`], which a node passes on from the message it received to the one
//! it sends without reading them (`specs/messages.md`). Each buffer of a message
//! is an `Arc<Buffer>`, so passing the carry on clones the pointers and copies
//! no bytes. A carried buffer still belongs to the pool of the writer that
//! filled it, and goes back there when the last message holding it is dropped.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::schema::{FieldSpec, Schema};

/// One storage unit of a buffer. Allocating in units of 64 bytes aligns every
/// buffer to a cache line, as torch's CPU allocator does, so that a tensor lent
/// over DLPack is as well aligned as one torch made.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Line([u8; 64]);

/// The buffers of one field, recycled across messages.
///
/// The free list is a mutex held only to push or pop one box. If contention ever
/// shows up in measurements, it can become a lock-free queue.
pub(crate) struct BufferPool {
    field: FieldSpec,
    free: Mutex<Vec<Box<[Line]>>>,
    /// How many buffers this pool has allocated in total.
    allocated: AtomicUsize,
}

impl BufferPool {
    fn new(field: FieldSpec) -> Arc<Self> {
        Arc::new(Self {
            field,
            free: Mutex::new(Vec::new()),
            allocated: AtomicUsize::new(0),
        })
    }

    fn take(self: &Arc<Self>) -> Buffer {
        let reused = lock(&self.free).pop();
        let storage = reused.unwrap_or_else(|| {
            self.allocated.fetch_add(1, Ordering::Relaxed);
            let lines = self.field.nbytes().div_ceil(size_of::<Line>());
            vec![Line([0; 64]); lines].into_boxed_slice()
        });
        Buffer {
            storage: Some(storage),
            pool: Arc::clone(self),
        }
    }
}

#[cfg(test)]
impl BufferPool {
    pub(crate) fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// How many buffers wait in the free list.
    pub(crate) fn free(&self) -> usize {
        lock(&self.free).len()
    }
}

/// The bytes of one field of a message, borrowed from a pool and returned to it
/// on drop. Immutable once its message is built, after which it is shared as
/// `Arc<Buffer>`: by the messages that carry it on, and by the tensors lent over
/// DLPack.
pub struct Buffer {
    /// `None` only while dropping.
    storage: Option<Box<[Line]>>,
    pool: Arc<BufferPool>,
}

impl Buffer {
    pub fn field(&self) -> &FieldSpec {
        &self.pool.field
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        let storage = self
            .storage
            .as_deref()
            .expect("storage is present until drop");
        // SAFETY: `Line` is plain bytes without padding, and the storage holds at
        // least `nbytes` of them.
        unsafe { std::slice::from_raw_parts(storage.as_ptr().cast(), self.field().nbytes()) }
    }

    /// Only reachable before the buffer is shared as `Arc<Buffer>`.
    fn bytes_mut(&mut self) -> &mut [u8] {
        let nbytes = self.field().nbytes();
        let storage = self
            .storage
            .as_deref_mut()
            .expect("storage is present until drop");
        // SAFETY: as in `bytes`, and the `&mut` borrow makes the access exclusive.
        unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr().cast(), nbytes) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Some(storage) = self.storage.take() {
            lock(&self.pool.free).push(storage);
        }
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("field", self.field())
            .finish_non_exhaustive()
    }
}

/// An immutable value of a slot: one buffer per field, in schema order, the
/// time it was written at on the writer's timer, and whether its writer sent it.
///
/// The carry buffers may be shared with the message they were passed on from
/// (and so belong to another writer's pools).
#[derive(Debug)]
pub struct Message {
    /// The value buffers, then the carry buffers.
    buffers: Vec<Arc<Buffer>>,
    /// Where the carry buffers start in `buffers`.
    carry_from: usize,
    time_ns: u64,
    sent: bool,
}

impl Message {
    /// The buffers of the value fields, in schema order.
    pub fn values(&self) -> &[Arc<Buffer>] {
        &self.buffers[..self.carry_from]
    }

    /// The buffers of the carry fields, in schema order (none if the schema
    /// carries nothing).
    pub fn carry(&self) -> &[Arc<Buffer>] {
        &self.buffers[self.carry_from..]
    }

    pub fn time_ns(&self) -> u64 {
        self.time_ns
    }

    /// Whether the slot's writer sent it (or a restore said so), rather than it
    /// being the value the slot was created or initialized with.
    pub fn sent(&self) -> bool {
        self.sent
    }
}

/// How a writer fills the buffers of a new message: one compact row-major byte
/// slice per field it writes, in schema order. That is every field (the value
/// fields, then the carry fields), or only the value fields when the carry is
/// passed on from another message. The buffers come from the pool with old
/// contents, so `fill` must overwrite every byte.
pub trait Fill<E>: FnOnce(&mut [&mut [u8]]) -> Result<(), E> {}
impl<E, F: FnOnce(&mut [&mut [u8]]) -> Result<(), E>> Fill<E> for F {}

/// The schema of a slot and the pools of its fields, shared by every slot that
/// one writer writes.
pub(crate) struct Messages {
    schema: Arc<Schema>,
    pools: Vec<Arc<BufferPool>>,
}

impl Messages {
    pub(crate) fn new(schema: Arc<Schema>) -> Arc<Self> {
        let pools = schema
            .fields()
            .iter()
            .cloned()
            .map(BufferPool::new)
            .collect();
        Arc::new(Self { schema, pools })
    }

    pub(crate) fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    /// Fills fresh buffers of every field, or of the value fields followed by
    /// the `relay`ed carry buffers. Nothing is published if `fill` fails, and the
    /// buffers go back to the pool.
    ///
    /// The caller has checked that `relay` has the carry fields of the schema.
    pub(crate) fn fill<E>(
        &self,
        relay: Option<&[Arc<Buffer>]>,
        fill: impl Fill<E>,
    ) -> Result<Vec<Arc<Buffer>>, E> {
        let Some(relay) = relay else {
            return filled(&self.pools, fill);
        };
        let mut buffers = filled(&self.pools[..self.schema.values().len()], fill)?;
        buffers.extend(relay.iter().map(Arc::clone));
        Ok(buffers)
    }

    /// Fills fresh buffers of the carry fields only, for a carry that a restore
    /// puts back.
    pub(crate) fn carry<E>(&self, fill: impl Fill<E>) -> Result<Vec<Arc<Buffer>>, E> {
        filled(&self.pools[self.schema.values().len()..], fill)
    }

    pub(crate) fn message(
        &self,
        buffers: Vec<Arc<Buffer>>,
        time_ns: u64,
        sent: bool,
    ) -> Arc<Message> {
        debug_assert_eq!(buffers.len(), self.schema.fields().len());
        Arc::new(Message {
            buffers,
            carry_from: self.schema.values().len(),
            time_ns,
            sent,
        })
    }

    /// A message of every field, filled by `fill`.
    pub(crate) fn build<E>(
        &self,
        time_ns: u64,
        sent: bool,
        fill: impl Fill<E>,
    ) -> Result<Arc<Message>, E> {
        let buffers = self.fill(None, fill)?;
        Ok(self.message(buffers, time_ns, sent))
    }

    pub(crate) fn zeros(&self, time_ns: u64) -> Arc<Message> {
        let zero = |bytes: &mut [&mut [u8]]| {
            bytes.iter_mut().for_each(|b| b.fill(0));
            Ok::<_, std::convert::Infallible>(())
        };
        let Ok(msg) = self.build(time_ns, false, zero);
        msg
    }
}

#[cfg(test)]
impl Messages {
    pub(crate) fn pools(&self) -> &[Arc<BufferPool>] {
        &self.pools
    }
}

/// Takes one buffer from each of `pools`, fills them and shares them.
fn filled<E>(pools: &[Arc<BufferPool>], fill: impl Fill<E>) -> Result<Vec<Arc<Buffer>>, E> {
    let mut buffers: Vec<Buffer> = pools.iter().map(BufferPool::take).collect();
    let mut bytes: Vec<&mut [u8]> = buffers.iter_mut().map(Buffer::bytes_mut).collect();
    fill(&mut bytes)?;
    Ok(buffers.into_iter().map(Arc::new).collect())
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing panics while holding these locks, so a poisoned value is still valid.
    m.lock().unwrap_or_else(|e| e.into_inner())
}
