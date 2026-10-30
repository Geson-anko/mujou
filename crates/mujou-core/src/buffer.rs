//! Buffers and messages: the immutable values that slots hold.
//!
//! A message is immutable and shared as `Arc<Message>`, so a reader that holds one
//! sees it unchanged however often the writer writes. The writer never writes into
//! a buffer a reader may see: it takes a buffer from a pool, fills it, and builds a
//! new message. A buffer goes back to its pool when the last message (or lent
//! tensor) that holds it is dropped, so in a steady state writing does not
//! allocate.
//!
//! A field may instead hold a value of the writer's own ([`Held`]), which the
//! core keeps without reading it: the bindings hold a copy of a CUDA tensor this
//! way, so that it stays on its device (`specs/messages.md`).

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::lock;
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
/// `Arc<Buffer>`: by the message, and by the tensors lent over DLPack.
pub struct Buffer {
    /// `None` only while dropping.
    storage: Option<Box<[Line]>>,
    pool: Arc<BufferPool>,
}

impl Buffer {
    pub fn field(&self) -> &FieldSpec {
        &self.pool.field
    }

    /// The field's value as compact row-major bytes.
    pub fn bytes(&self) -> &[u8] {
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

/// A value that a writer keeps outside the pools, such as a copy of a tensor on
/// a device. The core never reads it: the writer must not change it once it is
/// handed over, and its readers know what it is.
pub type Held = Arc<dyn Any + Send + Sync>;

/// The value of one field of a message.
#[derive(Clone)]
pub enum Value {
    /// Compact row-major bytes in a pooled buffer.
    Bytes(Arc<Buffer>),
    /// A value of the writer's own.
    Held(Held),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bytes(buffer) => buffer.fmt(f),
            Self::Held(_) => f.write_str("Held(..)"),
        }
    }
}

/// An immutable value of a slot: one value per field, in schema order, and
/// the time its writer wrote it at on the run's clock.
#[derive(Debug)]
pub struct Message {
    values: Vec<Value>,
    time_ns: Option<u64>,
}

impl Message {
    /// The values of the fields, in schema order.
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// The time the slot's writer wrote it at, or `None` for the value the slot
    /// holds until its writer first writes (zeros, or an initial value).
    pub fn time_ns(&self) -> Option<u64> {
        self.time_ns
    }
}

/// The value of one field of a new message, as its writer has set it so far.
enum Out {
    Unset,
    Bytes(Buffer),
    Held(Held),
}

/// Where a writer puts the value of one field of a new message: the bytes of a
/// buffer from the field's pool, or a value of its own.
pub struct FieldOut<'a> {
    pool: &'a Arc<BufferPool>,
    value: Out,
}

impl FieldOut<'_> {
    pub fn spec(&self) -> &FieldSpec {
        &self.pool.field
    }

    /// The bytes of a buffer for the field, compact and row-major, taken from
    /// the pool at the first call. The buffer comes with old contents, so
    /// overwrite every byte.
    pub fn bytes(&mut self) -> &mut [u8] {
        if !matches!(self.value, Out::Bytes(_)) {
            self.value = Out::Bytes(self.pool.take());
        }
        let Out::Bytes(buffer) = &mut self.value else {
            unreachable!("set above")
        };
        buffer.bytes_mut()
    }

    /// Makes `value` the field's value instead of a buffer (a buffer taken
    /// before goes back to its pool).
    pub fn hold(&mut self, value: Held) {
        self.value = Out::Held(value);
    }

    fn finish(self) -> Value {
        match self.value {
            Out::Bytes(buffer) => Value::Bytes(Arc::new(buffer)),
            Out::Held(held) => Value::Held(held),
            Out::Unset => panic!("a fill left the field '{}' unset", self.pool.field.name),
        }
    }
}

/// How a writer fills a new message: one [`FieldOut`] per field, in schema
/// order, each of which it must set.
pub trait Fill<E>: FnOnce(&mut [FieldOut<'_>]) -> Result<(), E> {}
impl<E, F: FnOnce(&mut [FieldOut<'_>]) -> Result<(), E>> Fill<E> for F {}

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

    /// The values of every field, as `fill` sets them. Nothing is published if
    /// `fill` fails, and the buffers it took go back to the pool. Panics if
    /// `fill` returns without setting a field.
    pub(crate) fn fill<E>(&self, fill: impl Fill<E>) -> Result<Vec<Value>, E> {
        let mut out: Vec<FieldOut<'_>> = self
            .pools
            .iter()
            .map(|pool| FieldOut {
                pool,
                value: Out::Unset,
            })
            .collect();
        fill(&mut out)?;
        Ok(out.into_iter().map(FieldOut::finish).collect())
    }

    pub(crate) fn message(&self, values: Vec<Value>, time_ns: Option<u64>) -> Arc<Message> {
        debug_assert_eq!(values.len(), self.schema.fields().len());
        Arc::new(Message { values, time_ns })
    }

    /// A message of every field, filled by `fill`.
    pub(crate) fn build<E>(
        &self,
        time_ns: Option<u64>,
        fill: impl Fill<E>,
    ) -> Result<Arc<Message>, E> {
        let values = self.fill(fill)?;
        Ok(self.message(values, time_ns))
    }

    /// Zeros that no writer wrote.
    pub(crate) fn zeros(&self) -> Arc<Message> {
        let zero = |out: &mut [FieldOut<'_>]| {
            out.iter_mut().for_each(|field| field.bytes().fill(0));
            Ok::<_, std::convert::Infallible>(())
        };
        let Ok(msg) = self.build(None, zero);
        msg
    }
}

#[cfg(test)]
impl Messages {
    pub(crate) fn pools(&self) -> &[Arc<BufferPool>] {
        &self.pools
    }
}
