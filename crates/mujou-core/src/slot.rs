//! Slots: the latest values that nodes share without waiting for each other.
//!
//! A slot holds the latest [`Message`] of one writer for any number of readers
//! (`docs/messages.md`). A message is immutable and shared as `Arc<Message>`, so a
//! reader that holds one sees it unchanged however often the writer writes. The
//! writer never writes into a buffer a reader may see: it takes a buffer from a
//! pool, fills it, and swaps the new message in. A buffer goes back to its pool
//! when the last message that holds it is dropped, so in a steady state writing
//! does not allocate.
//!
//! - [`Outlet`]: the sending side of one port, writing its outward slot and the
//!   slot of each channel
//! - [`InputGroup`] / [`InputSlot`]: the slots the host writes from outside, where
//!   one write to several slots becomes visible to every reader at once
//!   (`docs/network.md`)

use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;

pub use crate::dlpack::DLDataType as DType;
use crate::dlpack::{format_dtype, format_shape};
use crate::time::GlobalTimer;

/// One named tensor of a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DType,
}

impl FieldSpec {
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }

    pub fn elem_bytes(&self) -> usize {
        usize::from(self.dtype.bits / 8) * usize::from(self.dtype.lanes)
    }

    /// The size of the field as compact row-major bytes. Cannot overflow for the
    /// fields of a [`Schema`], which checks it.
    pub fn nbytes(&self) -> usize {
        self.numel() * self.elem_bytes()
    }

    /// `nbytes`, or `None` if it exceeds `isize::MAX`, the most one allocation
    /// (and a pointer offset within it) can span.
    fn checked_nbytes(&self) -> Option<usize> {
        let nbytes = self.shape.iter().try_fold(self.elem_bytes(), |acc, &n| {
            acc.checked_mul(usize::try_from(n).ok()?)
        })?;
        (nbytes <= isize::MAX as usize).then_some(nbytes)
    }
}

/// The fields of a message, fixed when the network is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    fields: Vec<FieldSpec>,
}

impl Schema {
    /// Checks that there is at least one field, the names are distinct, every
    /// extent is at least 1 (a 0-dimensional shape is fine), every dtype is a
    /// scalar of whole bytes, and every field fits in one allocation.
    pub fn new(fields: Vec<FieldSpec>) -> Result<Self, SlotError> {
        if fields.is_empty() {
            return Err(SlotError::InvalidSchema(
                "a schema needs at least one field".to_owned(),
            ));
        }
        let mut names = HashSet::new();
        for f in &fields {
            if !names.insert(f.name.as_str()) {
                return Err(SlotError::InvalidSchema(format!(
                    "duplicate field '{}'",
                    f.name
                )));
            }
            if f.shape.iter().any(|&n| n < 1) {
                return Err(SlotError::InvalidSchema(format!(
                    "field '{}': every extent must be >= 1, got shape {}",
                    f.name,
                    format_shape(&f.shape)
                )));
            }
            let DType { bits, lanes, .. } = f.dtype;
            if lanes != 1 || bits == 0 || bits % 8 != 0 {
                return Err(SlotError::InvalidSchema(format!(
                    "field '{}': unsupported dtype {}",
                    f.name,
                    format_dtype(f.dtype)
                )));
            }
            if f.checked_nbytes().is_none() {
                return Err(SlotError::InvalidSchema(format!(
                    "field '{}': shape {} of {} takes more than isize::MAX bytes",
                    f.name,
                    format_shape(&f.shape),
                    format_dtype(f.dtype)
                )));
            }
        }
        Ok(Self { fields })
    }

    pub fn fields(&self) -> &[FieldSpec] {
        &self.fields
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotError {
    InvalidSchema(String),
    /// A value does not fit the schema. `what` is `"shape"`, `"dtype"` or
    /// `"field count"`.
    Mismatch {
        slot: String,
        field: String,
        what: &'static str,
        expected: String,
        actual: String,
    },
    /// The value is not in host memory.
    UnsupportedDevice {
        slot: String,
        field: String,
        device_type: i32,
    },
    /// The producer handed over something that is not a valid DLPack tensor.
    InvalidDLPack {
        slot: String,
        field: String,
        reason: String,
    },
}

impl SlotError {
    /// Names the slot, for errors raised where only the field is known.
    pub fn at_slot(mut self, name: &str) -> Self {
        match &mut self {
            Self::InvalidSchema(_) => {}
            Self::Mismatch { slot, .. }
            | Self::UnsupportedDevice { slot, .. }
            | Self::InvalidDLPack { slot, .. } => *slot = name.to_owned(),
        }
        self
    }
}

/// `slot 'a/out', field 'x'`, leaving out what is empty: a tensor message has
/// one field named `""`, and an error from [`crate::dlpack::copy_from`] has no
/// slot until [`SlotError::at_slot`] names it.
pub fn location(slot: &str, field: &str) -> String {
    match (slot.is_empty(), field.is_empty()) {
        (false, false) => format!("slot '{slot}', field '{field}'"),
        (false, true) => format!("slot '{slot}'"),
        (true, false) => format!("field '{field}'"),
        (true, true) => String::new(),
    }
}

/// [`location`] followed by `": "`, or nothing if it is empty.
fn write_location(f: &mut fmt::Formatter<'_>, slot: &str, field: &str) -> fmt::Result {
    match location(slot, field) {
        at if at.is_empty() => Ok(()),
        at => write!(f, "{at}: "),
    }
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchema(reason) => write!(f, "invalid schema: {reason}"),
            Self::Mismatch {
                slot,
                field,
                what,
                expected,
                actual,
            } => {
                write_location(f, slot, field)?;
                write!(f, "expected {what} {expected}, got {actual}")
            }
            Self::UnsupportedDevice {
                slot,
                field,
                device_type,
            } => {
                write_location(f, slot, field)?;
                write!(
                    f,
                    "expected a tensor in host memory (DLPack device type 1 or 3), \
                     got device type {device_type}; move it to the CPU first"
                )
            }
            Self::InvalidDLPack {
                slot,
                field,
                reason,
            } => {
                write_location(f, slot, field)?;
                write!(f, "invalid DLPack tensor: {reason}")
            }
        }
    }
}

impl std::error::Error for SlotError {}

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
struct BufferPool {
    field: FieldSpec,
    free: Mutex<Vec<Box<[Line]>>>,
    /// How many buffers this pool has allocated in total.
    allocated: AtomicUsize,
}

impl BufferPool {
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

/// The bytes of one field of a message, borrowed from a pool and returned to it
/// on drop. Immutable once its message is built.
pub struct Buffer {
    /// `None` only while dropping.
    storage: Option<Box<[Line]>>,
    pool: Arc<BufferPool>,
}

impl Buffer {
    pub fn field(&self) -> &FieldSpec {
        &self.pool.field
    }

    pub fn bytes(&self) -> &[u8] {
        let storage = self
            .storage
            .as_deref()
            .expect("storage is present until drop");
        // SAFETY: `Line` is plain bytes without padding, and the storage holds at
        // least `nbytes` of them.
        unsafe { std::slice::from_raw_parts(storage.as_ptr().cast(), self.field().nbytes()) }
    }

    /// Only reachable before the buffer is in a message, which hands out `&Buffer`.
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

/// An immutable value of a slot: one buffer per field, in schema order, and the
/// global time it was written at.
#[derive(Debug)]
pub struct Message {
    buffers: Vec<Buffer>,
    time_ns: u64,
}

impl Message {
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    pub fn time_ns(&self) -> u64 {
        self.time_ns
    }
}

/// How a writer fills the buffers of a new message: one compact row-major byte
/// slice per field, in schema order. The buffers come from the pool with old
/// contents, so `fill` must overwrite every byte.
pub trait Fill<E>: FnOnce(&mut [&mut [u8]]) -> Result<(), E> {}
impl<E, F: FnOnce(&mut [&mut [u8]]) -> Result<(), E>> Fill<E> for F {}

/// The schema of a slot and the pools of its fields, shared by every slot that
/// one writer writes.
struct Messages {
    schema: Arc<Schema>,
    pools: Vec<Arc<BufferPool>>,
}

impl Messages {
    fn new(schema: Arc<Schema>) -> Arc<Self> {
        let pools = schema
            .fields()
            .iter()
            .map(|field| {
                Arc::new(BufferPool {
                    field: field.clone(),
                    free: Mutex::new(Vec::new()),
                    allocated: AtomicUsize::new(0),
                })
            })
            .collect();
        Arc::new(Self { schema, pools })
    }

    /// Fills fresh buffers. Nothing is published if `fill` fails, and the buffers
    /// go back to the pool.
    fn fill<E>(&self, fill: impl Fill<E>) -> Result<Vec<Buffer>, E> {
        let mut buffers: Vec<Buffer> = self.pools.iter().map(BufferPool::take).collect();
        let mut bytes: Vec<&mut [u8]> = buffers.iter_mut().map(Buffer::bytes_mut).collect();
        fill(&mut bytes)?;
        Ok(buffers)
    }

    fn build<E>(&self, time_ns: u64, fill: impl Fill<E>) -> Result<Arc<Message>, E> {
        let buffers = self.fill(fill)?;
        Ok(Arc::new(Message { buffers, time_ns }))
    }

    fn zeros(&self, time_ns: u64) -> Arc<Message> {
        let zero = |bytes: &mut [&mut [u8]]| {
            bytes.iter_mut().for_each(|b| b.fill(0));
            Ok::<_, std::convert::Infallible>(())
        };
        let Ok(msg) = self.build(time_ns, zero);
        msg
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing panics while holding these locks, so a poisoned value is still valid.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Wakes the readers blocked in [`Slot::wait_newer`].
///
/// A writer takes the lock only when someone is waiting, so writing to a slot
/// nobody waits on stays lock-free.
#[derive(Default)]
struct Waiters {
    count: AtomicUsize,
    lock: Mutex<()>,
    changed: Condvar,
}

impl Waiters {
    /// Called by the writer after it swapped the new value in.
    fn notify(&self) {
        // Pairs with the fence in `wait_while`: either the writer sees the waiter
        // counted, or the waiter sees the new value before it sleeps.
        fence(Ordering::SeqCst);
        if self.count.load(Ordering::Relaxed) > 0 {
            // Taking the lock orders the notification after a waiter that checked
            // the old value has gone to sleep, so the wakeup is not lost.
            drop(lock(&self.lock));
            self.changed.notify_all();
        }
    }

    /// Blocks while `unchanged()` holds, up to `timeout`. Returns whether it stopped
    /// holding.
    fn wait_while(&self, timeout: Duration, unchanged: impl Fn() -> bool) -> bool {
        let deadline = Instant::now().checked_add(timeout);
        self.count.fetch_add(1, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let mut guard = lock(&self.lock);
        let changed = loop {
            if !unchanged() {
                break true;
            }
            let wait = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::MAX,
            };
            if wait.is_zero() {
                break false;
            }
            guard = self
                .changed
                .wait_timeout(guard, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        };
        drop(guard);
        self.count.fetch_sub(1, Ordering::Relaxed);
        changed
    }
}

/// The latest message of one writer, read by any number of readers.
pub struct Slot {
    name: String,
    messages: Arc<Messages>,
    value: ArcSwap<Message>,
    waiters: Waiters,
}

impl Slot {
    fn new(name: String, messages: Arc<Messages>, initial: Arc<Message>) -> Self {
        Self {
            name,
            messages,
            value: ArcSwap::new(initial),
            waiters: Waiters::default(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.messages.schema
    }

    /// The latest message. Never blocks.
    pub fn load(&self) -> Arc<Message> {
        self.value.load_full()
    }

    /// Waits until the slot holds a message other than `last`, up to `timeout` of
    /// real time. With `last` absent, returns the current message at once.
    ///
    /// Only for the host (`subscribe`): a node must never wait for another, so the
    /// ports that nodes use do not expose it.
    pub fn wait_newer(
        &self,
        last: Option<&Arc<Message>>,
        timeout: Duration,
    ) -> Option<Arc<Message>> {
        let Some(last) = last else {
            return Some(self.load());
        };
        let unchanged = || Arc::ptr_eq(&self.value.load(), last);
        self.waiters
            .wait_while(timeout, unchanged)
            .then(|| self.load())
    }

    /// Sets the value and its time as they were when the state was taken.
    ///
    /// Call it only while no node runs (before the start, or while every node is
    /// paused with the timer frozen): it is a second way to write the slot, beside
    /// its writer, and the readers' handles reach it too.
    pub fn restore<E>(&self, time_ns: u64, fill: impl Fill<E>) -> Result<(), E> {
        let msg = self.messages.build(time_ns, fill)?;
        self.publish(msg);
        Ok(())
    }

    fn publish(&self, msg: Arc<Message>) {
        self.value.store(msg);
        self.waiters.notify();
    }
}

/// Which slots of an [`Outlet`] a message goes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The outward slot and every channel.
    All,
    /// Only the slot of channel `k`.
    Channel(usize),
}

/// The sending side of one port and the only writer of its slots: the outward
/// slot, which the host reads, and one slot per channel, which the peer reads.
///
/// Being the only writer is up to the caller: `send_with` takes `&self`, and the
/// port that owns the outlet accepts sends only from its node's thread. Sends
/// from two threads at once stay memory safe, but the slots may then keep their
/// messages out of time order, and `Target::All` may leave the outward slot and
/// the channels holding different messages.
pub struct Outlet {
    messages: Arc<Messages>,
    outbox: Arc<Slot>,
    channels: Vec<Arc<Slot>>,
    timer: &'static GlobalTimer,
}

impl Outlet {
    /// Creates the outward slot `"{name}/out"` and the channel slots
    /// `"{name}/ch{k}"`, all holding zeros stamped with the current time.
    pub fn new(
        schema: Arc<Schema>,
        channels: usize,
        name: &str,
        timer: &'static GlobalTimer,
    ) -> Self {
        let messages = Messages::new(schema);
        let zeros = messages.zeros(timer.time_ns());
        let slot =
            |name: String| Arc::new(Slot::new(name, Arc::clone(&messages), Arc::clone(&zeros)));
        Self {
            outbox: slot(format!("{name}/out")),
            channels: (0..channels)
                .map(|k| slot(format!("{name}/ch{k}")))
                .collect(),
            messages,
            timer,
        }
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.messages.schema
    }

    pub fn outbox(&self) -> &Arc<Slot> {
        &self.outbox
    }

    pub fn channels(&self) -> &[Arc<Slot>] {
        &self.channels
    }

    /// The slot of channel `k`. Panics if there is no such channel.
    pub fn channel(&self, k: usize) -> &Arc<Slot> {
        &self.channels[k]
    }

    /// Fills a new message and publishes it to `target`, stamped with the time
    /// after `fill` returns. `Target::All` publishes the very same message to every
    /// slot. Panics if a `Target::Channel` does not exist.
    pub fn send_with<E>(&self, target: Target, fill: impl Fill<E>) -> Result<(), E> {
        let buffers = self.messages.fill(fill)?;
        let msg = Arc::new(Message {
            buffers,
            time_ns: self.timer.time_ns(),
        });
        match target {
            Target::All => {
                for slot in &self.channels {
                    slot.publish(Arc::clone(&msg));
                }
                self.outbox.publish(msg);
            }
            Target::Channel(k) => self.channels[k].publish(msg),
        }
        Ok(())
    }
}

/// The version counter of the slots the host writes, which makes one write to
/// several slots visible to every reader at once (`docs/network.md`).
///
/// A write places the new value of each slot as its next value tagged with
/// version `V + 1`, then increments `V`. A reader takes the next value only if its
/// version is at most `V`, so the single increment switches all slots together,
/// without the writer or any reader waiting.
pub struct InputGroup {
    version: AtomicU64,
    /// Serializes writers (the host may write from several threads). Readers never
    /// take it.
    write_lock: Mutex<()>,
    timer: &'static GlobalTimer,
}

/// The value of an [`InputSlot`]: the current value, and the next value once its
/// version is published.
struct Pair {
    current: Arc<Message>,
    next: Option<(u64, Arc<Message>)>,
}

impl Pair {
    fn resolve(&self, version: u64) -> &Arc<Message> {
        match &self.next {
            Some((v, next)) if *v <= version => next,
            _ => &self.current,
        }
    }
}

/// A slot the host writes through its [`InputGroup`].
pub struct InputSlot {
    name: String,
    group: Arc<InputGroup>,
    messages: Arc<Messages>,
    pair: ArcSwap<Pair>,
}

impl InputGroup {
    pub fn new(timer: &'static GlobalTimer) -> Arc<Self> {
        Arc::new(Self {
            version: AtomicU64::new(0),
            write_lock: Mutex::new(()),
            timer,
        })
    }

    /// Creates the slot `"{name}/in"`, holding zeros stamped with the current time.
    pub fn input(self: &Arc<Self>, schema: Arc<Schema>, name: &str) -> Arc<InputSlot> {
        let messages = Messages::new(schema);
        let current = messages.zeros(self.timer.time_ns());
        Arc::new(InputSlot {
            name: format!("{name}/in"),
            group: Arc::clone(self),
            messages,
            pair: ArcSwap::from_pointee(Pair {
                current,
                next: None,
            }),
        })
    }

    /// Writes one message to each of `slots`, all stamped with the same time.
    /// `fill(i, bytes)` fills the message of `slots[i]`. Every message is filled
    /// before any is placed, so if one `fill` fails no slot changes.
    ///
    /// Panics if a slot belongs to another group or appears more than once.
    pub fn write<E>(
        &self,
        slots: &[&InputSlot],
        mut fill: impl FnMut(usize, &mut [&mut [u8]]) -> Result<(), E>,
    ) -> Result<(), E> {
        assert!(
            slots.iter().all(|s| std::ptr::eq(&*s.group, self)),
            "every slot must belong to this input group"
        );
        assert!(
            slots
                .iter()
                .enumerate()
                .all(|(i, a)| slots[..i].iter().all(|b| !std::ptr::eq(*a, *b))),
            "each slot may appear only once in a write"
        );
        let _guard = lock(&self.write_lock);
        let filled = slots
            .iter()
            .enumerate()
            .map(|(i, slot)| slot.messages.fill(|bytes: &mut [&mut [u8]]| fill(i, bytes)))
            .collect::<Result<Vec<_>, E>>()?;
        let time_ns = self.timer.time_ns();
        // Under the lock no write is in flight, so every next value is already
        // published (its version is at most `version`) and becomes current.
        let version = self.version.load(Ordering::Relaxed);
        for (slot, buffers) in slots.iter().zip(filled) {
            let next = Arc::new(Message { buffers, time_ns });
            let current = Arc::clone(slot.pair.load().resolve(version));
            slot.pair.store(Arc::new(Pair {
                current,
                next: Some((version + 1, next)),
            }));
        }
        // Release: a reader that sees the new version also sees every pair above.
        self.version.store(version + 1, Ordering::Release);
        Ok(())
    }
}

impl InputSlot {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.messages.schema
    }

    /// Whether `group` is the group that writes this slot.
    pub fn belongs_to(&self, group: &InputGroup) -> bool {
        std::ptr::eq(&*self.group, group)
    }

    /// The latest value whose write has completed. Never blocks.
    pub fn load(&self) -> Arc<Message> {
        // The pair first, then the version: a version at least as new as the pair
        // can only promote its next value, never skip past it.
        let pair = self.pair.load();
        let version = self.group.version.load(Ordering::Acquire);
        Arc::clone(pair.resolve(version))
    }

    /// Sets the value and its time as they were when the state was taken.
    ///
    /// Call it only while no node runs (before the start, or while every node is
    /// paused with the timer frozen): it is a second way to write the slot, beside
    /// its writer, and the readers' handles reach it too.
    pub fn restore<E>(&self, time_ns: u64, fill: impl Fill<E>) -> Result<(), E> {
        let current = self.messages.build(time_ns, fill)?;
        let _guard = lock(&self.group.write_lock);
        self.pair.store(Arc::new(Pair {
            current,
            next: None,
        }));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::thread;

    use super::*;
    use crate::dlpack::{self, DLTensor};
    use crate::time::TimerState;

    const I64: DType = DType {
        code: 0,
        bits: 64,
        lanes: 1,
    };

    fn timer() -> &'static GlobalTimer {
        let timer = Box::leak(Box::new(GlobalTimer::new()));
        timer.use_virtual_clock(true);
        timer
            .load_state(TimerState {
                time_ns: 0,
                scale: 1.0,
            })
            .unwrap();
        timer
    }

    fn field(name: &str, shape: &[i64]) -> FieldSpec {
        FieldSpec {
            name: name.to_owned(),
            shape: shape.to_vec(),
            dtype: I64,
        }
    }

    /// Two scalar int64 fields, so that a torn message would show two values.
    fn pair_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![field("a", &[]), field("b", &[])]).unwrap())
    }

    fn scalar_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![field("", &[])]).unwrap())
    }

    /// Fills every field with `k`.
    fn fill_k(k: i64) -> impl FnOnce(&mut [&mut [u8]]) -> Result<(), Infallible> {
        move |bytes| {
            for b in bytes {
                b.copy_from_slice(&k.to_ne_bytes());
            }
            Ok(())
        }
    }

    fn values(msg: &Message) -> Vec<i64> {
        msg.buffers()
            .iter()
            .map(|b| i64::from_ne_bytes(b.bytes().try_into().unwrap()))
            .collect()
    }

    fn allocations(slot: &Slot) -> usize {
        slot.messages
            .pools
            .iter()
            .map(|p| p.allocated.load(Ordering::Relaxed))
            .sum()
    }

    #[test]
    fn schema_rejects_invalid_fields() {
        let bool8 = DType {
            code: 6,
            bits: 8,
            lanes: 1,
        };
        assert!(Schema::new(vec![]).is_err());
        assert!(Schema::new(vec![field("a", &[]), field("a", &[2])]).is_err());
        assert!(Schema::new(vec![field("a", &[2, 0])]).is_err());
        // 2^61 int64 elements are 2^64 bytes, and 2^32 x 2^32 overflows usize.
        assert!(Schema::new(vec![field("a", &[1 << 61])]).is_err());
        assert!(Schema::new(vec![field("a", &[1 << 32, 1 << 32])]).is_err());
        let mut f = field("a", &[2]);
        f.dtype.lanes = 4;
        assert!(Schema::new(vec![f]).is_err());
        let ok = Schema::new(vec![FieldSpec {
            name: "m".to_owned(),
            shape: vec![3],
            dtype: bool8,
        }]);
        assert_eq!(ok.unwrap().fields()[0].nbytes(), 3);
    }

    #[test]
    fn starts_with_zeros_at_creation_time() {
        let timer = timer();
        timer.advance_ns(5).unwrap();
        let outlet = Outlet::new(pair_schema(), 1, "n.p", timer);
        let msg = outlet.channel(0).load();
        assert_eq!(values(&msg), [0, 0]);
        assert_eq!(msg.time_ns(), 5);
        assert_eq!(outlet.outbox().name(), "n.p/out");
        assert_eq!(outlet.channel(0).name(), "n.p/ch0");
    }

    #[test]
    fn send_to_all_shares_one_message() {
        let timer = timer();
        let outlet = Outlet::new(scalar_schema(), 3, "p", timer);
        timer.advance_ns(42).unwrap();
        outlet.send_with(Target::All, fill_k(7)).unwrap();
        let out = outlet.outbox().load();
        assert_eq!(values(&out), [7]);
        assert_eq!(out.time_ns(), 42);
        for slot in outlet.channels() {
            assert!(Arc::ptr_eq(&slot.load(), &out));
        }
    }

    #[test]
    fn send_to_one_channel_changes_only_it() {
        let outlet = Outlet::new(scalar_schema(), 3, "p", timer());
        outlet.send_with(Target::Channel(1), fill_k(9)).unwrap();
        assert_eq!(values(&outlet.channel(1).load()), [9]);
        for slot in [outlet.outbox(), outlet.channel(0), outlet.channel(2)] {
            assert_eq!(values(&slot.load()), [0]);
        }
    }

    #[test]
    fn failed_fill_publishes_nothing() {
        let outlet = Outlet::new(scalar_schema(), 1, "p", timer());
        let before = outlet.outbox().load();
        assert_eq!(outlet.send_with(Target::All, |_| Err("bad")), Err("bad"));
        assert!(Arc::ptr_eq(&outlet.outbox().load(), &before));
    }

    #[test]
    fn held_message_survives_later_writes() {
        let outlet = Outlet::new(scalar_schema(), 0, "p", timer());
        outlet.send_with(Target::All, fill_k(1)).unwrap();
        let held = outlet.outbox().load();
        for k in 2..=100 {
            outlet.send_with(Target::All, fill_k(k)).unwrap();
        }
        assert_eq!(values(&held), [1]);
        assert_eq!(values(&outlet.outbox().load()), [100]);
    }

    #[test]
    fn pool_reuses_buffers_in_steady_state() {
        let outlet = Outlet::new(scalar_schema(), 1, "p", timer());
        let slot = outlet.outbox();
        for k in 0..1000 {
            outlet.send_with(Target::All, fill_k(k)).unwrap();
            // A reader that holds each message only while it reads it.
            let _ = values(&slot.load());
        }
        // The zeros, the latest message and the one being written.
        assert!(allocations(slot) <= 3, "allocated {}", allocations(slot));
    }

    #[test]
    fn readers_see_whole_messages_in_order() {
        const WRITES: i64 = 20_000;
        let outlet = Outlet::new(pair_schema(), 1, "p", timer());
        let slot = Arc::clone(outlet.channel(0));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let slot = Arc::clone(&slot);
                thread::spawn(move || {
                    let mut last = 0;
                    while last < WRITES {
                        let v = values(&slot.load());
                        assert_eq!(v[0], v[1], "torn message");
                        assert!(v[0] >= last, "went back from {last} to {}", v[0]);
                        last = v[0];
                    }
                })
            })
            .collect();
        for k in 1..=WRITES {
            outlet.send_with(Target::All, fill_k(k)).unwrap();
        }
        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn wait_newer_returns_on_write_or_timeout() {
        let outlet = Outlet::new(scalar_schema(), 0, "p", timer());
        let slot = Arc::clone(outlet.outbox());
        let first = slot.load();
        assert!(Arc::ptr_eq(
            &slot.wait_newer(None, Duration::ZERO).unwrap(),
            &first
        ));
        assert!(
            slot.wait_newer(Some(&first), Duration::from_millis(20))
                .is_none()
        );

        let waiter = {
            let (slot, first) = (Arc::clone(&slot), Arc::clone(&first));
            thread::spawn(move || slot.wait_newer(Some(&first), Duration::from_secs(10)))
        };
        thread::sleep(Duration::from_millis(20));
        outlet.send_with(Target::All, fill_k(3)).unwrap();
        let newer = waiter.join().unwrap().expect("woken by the write");
        assert_eq!(values(&newer), [3]);
    }

    #[test]
    fn write_without_waiters_takes_no_lock() {
        let outlet = Arc::new(Outlet::new(scalar_schema(), 0, "p", timer()));
        let _held = lock(&outlet.outbox().waiters.lock);
        let (done, finished) = mpsc::channel();
        let writer = {
            let outlet = Arc::clone(&outlet);
            thread::spawn(move || {
                outlet.send_with(Target::All, fill_k(1)).unwrap();
                done.send(()).unwrap();
            })
        };
        // Would time out if the write waited for the lock held above.
        finished.recv_timeout(Duration::from_secs(5)).unwrap();
        writer.join().unwrap();
    }

    #[test]
    fn restore_sets_value_and_time() {
        let outlet = Outlet::new(scalar_schema(), 1, "p", timer());
        outlet.channel(0).restore(123, fill_k(4)).unwrap();
        let msg = outlet.channel(0).load();
        assert_eq!((values(&msg), msg.time_ns()), (vec![4], 123));
        assert_eq!(values(&outlet.outbox().load()), [0]);
    }

    #[test]
    fn input_group_write_stamps_one_time() {
        let timer = timer();
        let group = InputGroup::new(timer);
        let (a, b) = (
            group.input(scalar_schema(), "a"),
            group.input(scalar_schema(), "b"),
        );
        assert_eq!(a.name(), "a/in");
        timer.advance_ns(77).unwrap();
        group
            .write(&[&a, &b], |i, bytes| fill_k(i as i64 + 1)(bytes))
            .unwrap();
        let (ma, mb) = (a.load(), b.load());
        assert_eq!((values(&ma), values(&mb)), (vec![1], vec![2]));
        assert_eq!((ma.time_ns(), mb.time_ns()), (77, 77));
    }

    #[test]
    fn input_group_write_is_all_or_nothing() {
        let group = InputGroup::new(timer());
        let (a, b) = (
            group.input(scalar_schema(), "a"),
            group.input(scalar_schema(), "b"),
        );
        let result = group.write(&[&a, &b], |i, bytes| {
            if i == 1 {
                Err("bad")
            } else {
                fill_k(5)(bytes).map_err(|_| "")
            }
        });
        assert_eq!(result, Err("bad"));
        assert_eq!((values(&a.load()), values(&b.load())), (vec![0], vec![0]));
    }

    #[test]
    fn input_restore_drops_pending_next_value() {
        let group = InputGroup::new(timer());
        let a = group.input(scalar_schema(), "a");
        group.write(&[&a], |_, bytes| fill_k(1)(bytes)).unwrap();
        a.restore(9, fill_k(2)).unwrap();
        let msg = a.load();
        assert_eq!((values(&msg), msg.time_ns()), (vec![2], 9));
    }

    /// The observable form of "no reader ever sees a mix of old and new values":
    /// whichever order a reader reads the two slots in, the second read is never
    /// older than the first.
    #[test]
    fn input_group_switches_all_slots_at_once() {
        const WRITES: i64 = 100_000;
        let group = InputGroup::new(timer());
        let (a, b) = (
            group.input(scalar_schema(), "a"),
            group.input(scalar_schema(), "b"),
        );
        let done = Arc::new(AtomicBool::new(false));
        let reader = |first: Arc<InputSlot>, second: Arc<InputSlot>| {
            let done = Arc::clone(&done);
            thread::spawn(move || {
                let mut reads = 0u64;
                while !done.load(Ordering::Relaxed) {
                    let x = values(&first.load())[0];
                    let y = values(&second.load())[0];
                    assert!(
                        y >= x,
                        "read {x} from {} and then {y} from {}",
                        first.name(),
                        second.name()
                    );
                    reads += 1;
                }
                reads
            })
        };
        let ab = reader(Arc::clone(&a), Arc::clone(&b));
        let ba = reader(Arc::clone(&b), Arc::clone(&a));
        for k in 1..=WRITES {
            group.write(&[&a, &b], |_, bytes| fill_k(k)(bytes)).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        assert!(ab.join().unwrap() > 0 && ba.join().unwrap() > 0);
        assert_eq!(values(&a.load()), [WRITES]);
    }

    #[test]
    fn lent_tensor_keeps_message_alive_until_deleted() {
        let outlet = Outlet::new(
            Arc::new(Schema::new(vec![field("x", &[2, 3])]).unwrap()),
            0,
            "p",
            timer(),
        );
        outlet
            .send_with(Target::All, |bytes: &mut [&mut [u8]]| {
                for (i, chunk) in bytes[0].chunks_exact_mut(8).enumerate() {
                    chunk.copy_from_slice(&(i as i64).to_ne_bytes());
                }
                Ok::<_, Infallible>(())
            })
            .unwrap();
        let msg = outlet.outbox().load();
        let field = msg.buffers()[0].field().clone();
        let before = Arc::strong_count(&msg);

        for versioned in [true, false] {
            let (tensor, delete): (&DLTensor, Box<dyn FnOnce()>) = if versioned {
                let t = dlpack::export_versioned(Arc::clone(&msg), 0);
                // SAFETY: `t` stays valid until its deleter is called below.
                unsafe {
                    assert_eq!((*t).version, dlpack::VERSION);
                    (&(*t).dl_tensor, Box::new(move || (*t).deleter.unwrap()(t)))
                }
            } else {
                let t = dlpack::export_legacy(Arc::clone(&msg), 0);
                // SAFETY: as above.
                unsafe { (&(*t).dl_tensor, Box::new(move || (*t).deleter.unwrap()(t))) }
            };
            assert_eq!(Arc::strong_count(&msg), before + 1);
            let mut copy = vec![0u8; field.nbytes()];
            // SAFETY: the exported tensor describes the live message buffer.
            unsafe { dlpack::copy_from(tensor, &field, &mut copy) }.unwrap();
            assert_eq!(copy, msg.buffers()[0].bytes());
            // SAFETY: the strides point to `ndim` values.
            let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 2) };
            assert_eq!(strides, [3, 1]);
            delete();
            assert_eq!(Arc::strong_count(&msg), before);
        }
    }
}
