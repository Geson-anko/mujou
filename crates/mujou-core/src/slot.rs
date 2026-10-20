//! Slots: the latest values that nodes share without waiting for each other.
//!
//! A slot holds the latest [`Message`] of one writer for any number of readers
//! (`specs/messages.md`). The writer builds each message in fresh buffers and
//! swaps it in, so a reader never sees a message change.
//!
//! Each slot keeps its value behind its own mutex, held only to clone or swap
//! an `Arc`, never while a buffer is filled or a message dropped. A lock-free
//! swap through `arc-swap` was used before, and broke memory safety: its readers
//! record the pointers they load in a list shared by every `ArcSwap` in the
//! process, so once a message is freed and its address reused for a value of
//! another slot (even of another type), a writer of that slot could hand the
//! value to a reader of this one (`specs/native-core.md`).
//!
//! A slot has one of two writers, its `Source`:
//!
//! - [`Outlet`]: the sending side of one port, writing its outward slot and the
//!   slot of each channel
//! - [`InputGroup`]: the host, writing the slots of the inputs from outside, where
//!   one write to several slots becomes visible to every reader at once
//!   (`specs/network.md`)

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::buffer::{Buffer, Fill, Message, Messages, lock};
use crate::schema::{FieldError, FieldSpec, Schema, format_dtype, format_shape};
use crate::time::{Timer, Timestamp, remaining};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotError {
    /// A value of `field` cannot be written to `slot`.
    Field {
        slot: String,
        field: String,
        error: FieldError,
    },
    /// The carried buffers passed on do not have the carry fields of the
    /// schema, or the schema carries nothing.
    CarryMismatch {
        slot: String,
        expected: Vec<FieldSpec>,
        actual: Vec<FieldSpec>,
    },
}

/// `slot 'a/out', field 'x'`, or `slot 'a/out'` for the one field of a tensor
/// message, which is named `""`.
pub fn location(slot: &str, field: &str) -> String {
    if field.is_empty() {
        format!("slot '{slot}'")
    } else {
        format!("slot '{slot}', field '{field}'")
    }
}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field { slot, field, error } => write!(f, "{}: {error}", location(slot, field)),
            Self::CarryMismatch {
                slot,
                expected,
                actual,
            } => {
                write!(f, "{}: ", location(slot, ""))?;
                if expected.is_empty() {
                    write!(f, "expected to carry nothing, so it passes nothing on")
                } else {
                    let (expected, actual) = (format_fields(expected), format_fields(actual));
                    write!(f, "expected to carry {expected}, got {actual}")
                }
            }
        }
    }
}

impl std::error::Error for SlotError {}

/// `('y': (2,) int64, 'mask': (2,) bool)`, or `nothing` for no fields: how a
/// [`SlotError::CarryMismatch`] lists them.
fn format_fields(fields: &[FieldSpec]) -> String {
    if fields.is_empty() {
        return "nothing".to_owned();
    }
    let fields: Vec<String> = fields
        .iter()
        .map(|f| {
            format!(
                "'{}': {} {}",
                f.name,
                format_shape(&f.shape),
                format_dtype(f.dtype)
            )
        })
        .collect();
    format!("({})", fields.join(", "))
}

/// Checks that `relay` has the carry fields of `schema`, which must carry
/// something. `slot` names the slot in the error.
fn check_relay(schema: &Schema, relay: &[Arc<Buffer>], slot: &str) -> Result<(), SlotError> {
    let expected = schema.carry();
    let matches = !expected.is_empty()
        && relay.len() == expected.len()
        && relay.iter().zip(expected).all(|(b, f)| b.field() == f);
    if matches {
        return Ok(());
    }
    Err(SlotError::CarryMismatch {
        slot: slot.to_owned(),
        expected: expected.to_vec(),
        actual: relay.iter().map(|b| b.field().clone()).collect(),
    })
}

/// Who writes a slot.
enum Source {
    /// Its [`Outlet`], which replaces the current value.
    Own,
    /// The host, through this group, which places a next value that the
    /// group's version switches in.
    Group(Arc<InputGroup>),
}

/// The value of a slot: the current value, and the next value once its
/// version is published. Only an [`InputGroup`] sets a next value.
struct Pair {
    current: Arc<Message>,
    next: Option<(u64, Arc<Message>)>,
}

impl Pair {
    fn new(current: Arc<Message>) -> Self {
        Self {
            current,
            next: None,
        }
    }

    fn resolve(&self, version: u64) -> &Arc<Message> {
        match &self.next {
            Some((v, next)) if *v <= version => next,
            _ => &self.current,
        }
    }
}

/// The latest message of one writer, read by any number of readers.
pub struct Slot {
    name: String,
    messages: Arc<Messages>,
    source: Source,
    /// Held only to read or replace the pair, and by a reader in
    /// [`Slot::wait_newer`] while it checks the value (it sleeps without it).
    value: Mutex<Pair>,
    /// How many readers are in [`Slot::wait_newer`]. A writer notifies only when
    /// it is not zero.
    waiters: AtomicUsize,
    changed: Condvar,
}

impl Slot {
    fn new(name: String, messages: Arc<Messages>, source: Source, initial: Arc<Message>) -> Self {
        Self {
            name,
            messages,
            source,
            value: Mutex::new(Pair::new(initial)),
            waiters: AtomicUsize::new(0),
            changed: Condvar::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn schema(&self) -> &Arc<Schema> {
        self.messages.schema()
    }

    /// Whether `group` is the group that writes this slot.
    pub fn belongs_to(&self, group: &InputGroup) -> bool {
        matches!(&self.source, Source::Group(g) if std::ptr::eq(&**g, group))
    }

    /// The version of the writes that are complete. An outlet never places a
    /// next value, so any version reads its current one.
    fn version(&self) -> u64 {
        match &self.source {
            Source::Own => 0,
            Source::Group(group) => group.version.load(Ordering::Acquire),
        }
    }

    /// The latest message whose write has completed. Waits only for another
    /// reader or the writer to read or replace the pair.
    pub fn load(&self) -> Arc<Message> {
        // The pair first, then the version: a version at least as new as the pair
        // can only promote its next value, never skip past it.
        let pair = lock(&self.value);
        Arc::clone(pair.resolve(self.version()))
    }

    /// Waits until the slot holds a message other than `last`, up to `timeout` of
    /// real time.
    ///
    /// Only for the host (`subscribe`): a node must never wait for another, so the
    /// ports that nodes use do not expose it.
    pub fn wait_newer(&self, last: &Arc<Message>, timeout: Duration) -> Option<Arc<Message>> {
        let deadline = Instant::now().checked_add(timeout);
        // Counted before taking the lock: a writer that takes the lock after this
        // reader has checked the value therefore sees the count and notifies.
        self.waiters.fetch_add(1, Ordering::Relaxed);
        let mut pair = lock(&self.value);
        let newer = loop {
            let value = pair.resolve(self.version());
            if !Arc::ptr_eq(value, last) {
                break Some(Arc::clone(value));
            }
            let wait = remaining(deadline);
            if wait.is_zero() {
                break None;
            }
            pair = self
                .changed
                .wait_timeout(pair, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        };
        drop(pair);
        self.waiters.fetch_sub(1, Ordering::Relaxed);
        newer
    }

    /// Sets the value, its time and whether it was sent as they were when the
    /// state was taken (`sent` is false for an initial value), dropping any
    /// pending write.
    ///
    /// Call it only while no node runs and the host does not write (before the
    /// start, or while every node is paused with the timer frozen): it is a
    /// second way to write the slot, beside its writer, and the readers' handles
    /// reach it too.
    ///
    /// `fill` fills every field, the carry fields included.
    pub fn restore<E>(&self, time_ns: u64, sent: bool, fill: impl Fill<E>) -> Result<(), E> {
        let msg = self.messages.build(time_ns, sent, fill)?;
        self.publish(msg);
        Ok(())
    }

    /// New buffers of the carry fields, filled by `fill` from the pools of this
    /// slot's writer: the carry a restore gives back to the port that reads this
    /// slot, as if it had last received it from here.
    pub fn new_carry<E>(&self, fill: impl Fill<E>) -> Result<Vec<Arc<Buffer>>, E> {
        self.messages.carry(fill)
    }

    /// Makes `msg` the current value.
    fn publish(&self, msg: Arc<Message>) {
        let old = std::mem::replace(&mut *lock(&self.value), Pair::new(msg));
        if self.waiters.load(Ordering::Relaxed) > 0 {
            self.changed.notify_all();
        }
        // Dropping the last holder of a message returns its buffers to the pools,
        // which is done without the slot's lock.
        drop(old);
    }

    /// Wakes the readers in [`Slot::wait_newer`] after a group published the
    /// version of a next value. Taking the lock first lets every reader that
    /// checked the older version go to sleep, or see the new one.
    fn wake(&self) {
        drop(lock(&self.value));
        if self.waiters.load(Ordering::Relaxed) > 0 {
            self.changed.notify_all();
        }
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
/// port that owns the outlet accepts sends only from its node's hooks, which
/// never run on two threads at once. Sends from two threads at once stay memory
/// safe, but the slots may then keep their messages out of time order, and
/// `Target::All` may leave the outward slot and the channels holding different
/// messages.
pub struct Outlet {
    messages: Arc<Messages>,
    outbox: Arc<Slot>,
    channels: Vec<Arc<Slot>>,
    timer: Arc<Timer>,
}

impl Outlet {
    /// Creates the outward slot `"{name}/out"` and the channel slots
    /// `"{name}/ch{k}"`, all holding zeros stamped with the current time.
    /// Every message is stamped with the time on `timer`.
    ///
    /// The slots and the host's inputs of one run must use the timer of its
    /// runtime: on another timer, the times of the messages would not be on
    /// the axis the nodes' Δt is measured on, and a pause of the run would not
    /// stop them, so a state taken at T could hold later messages.
    pub fn new(schema: Arc<Schema>, channels: usize, name: &str, timer: Arc<Timer>) -> Self {
        let messages = Messages::new(schema);
        let zeros = messages.zeros(timer.time_ns());
        let slot = |name: String| {
            let (messages, zeros) = (Arc::clone(&messages), Arc::clone(&zeros));
            Arc::new(Slot::new(name, messages, Source::Own, zeros))
        };
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
        self.messages.schema()
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
    ///
    /// Without `relay`, `fill` fills every field. With it, `fill` fills the value
    /// fields, and the message carries the buffers of `relay` (the carry of a
    /// message this outlet's node received), shared and not copied. They must
    /// have the carry fields of the schema, or nothing is published and the error
    /// is a [`SlotError::CarryMismatch`].
    pub fn send_with<E: From<SlotError>>(
        &self,
        target: Target,
        relay: Option<&[Arc<Buffer>]>,
        fill: impl Fill<E>,
    ) -> Result<(), E> {
        let name = match target {
            Target::All => self.outbox.name(),
            Target::Channel(k) => self.channels[k].name(),
        };
        if let Some(relay) = relay {
            check_relay(self.schema(), relay, name)?;
        }
        let buffers = self.messages.fill(relay, fill)?;
        let msg = self.messages.message(buffers, self.timer.time_ns(), true);
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
/// several slots visible to every reader at once (`specs/network.md`).
///
/// A write places the new value of each slot as its next value tagged with
/// version `V + 1`, then increments `V`. A reader takes the next value only if its
/// version is at most `V`, so the single increment switches all slots together.
/// Neither the writer nor a reader waits for the other to make progress: each waits
/// only while the lock of one slot is held to read or replace its pair.
pub struct InputGroup {
    version: AtomicU64,
    /// Serializes writers (the host may write from several threads). Readers never
    /// take it.
    write_lock: Mutex<()>,
    timer: Arc<Timer>,
}

impl InputGroup {
    /// Every write is stamped with the time on `timer`.
    ///
    /// The slots and the host's inputs of one run must use the timer of its
    /// runtime: on another timer, the times of the messages would not be on
    /// the axis the nodes' Δt is measured on, and a pause of the run would not
    /// stop them, so a state taken at T could hold later messages.
    pub fn new(timer: Arc<Timer>) -> Arc<Self> {
        Arc::new(Self {
            version: AtomicU64::new(0),
            write_lock: Mutex::new(()),
            timer,
        })
    }

    /// Creates the slot `"{name}/in"`, holding zeros stamped with the current time.
    pub fn input(self: &Arc<Self>, schema: Arc<Schema>, name: &str) -> Arc<Slot> {
        let messages = Messages::new(schema);
        let zeros = messages.zeros(self.timer.time_ns());
        let source = Source::Group(Arc::clone(self));
        Arc::new(Slot::new(format!("{name}/in"), messages, source, zeros))
    }

    /// Writes one message to each of `slots`, all stamped with the same time,
    /// and returns that time. `fill(i, bytes)` fills every field (the carry
    /// fields included) of the message of `slots[i]`.
    /// Every message is filled before any is placed, so if one `fill` fails no
    /// slot changes.
    ///
    /// Panics if a slot belongs to another group or appears more than once.
    pub fn write<E>(
        &self,
        slots: &[&Slot],
        mut fill: impl FnMut(usize, &mut [&mut [u8]]) -> Result<(), E>,
    ) -> Result<Timestamp, E> {
        assert!(
            slots.iter().all(|s| s.belongs_to(self)),
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
            .map(|(i, slot)| {
                slot.messages
                    .fill(None, |bytes: &mut [&mut [u8]]| fill(i, bytes))
            })
            .collect::<Result<Vec<_>, E>>()?;
        let time_ns = self.timer.time_ns();
        // Under the lock no write is in flight, so every next value is already
        // published (its version is at most `version`) and becomes current.
        let version = self.version.load(Ordering::Relaxed);
        for (slot, buffers) in slots.iter().zip(filled) {
            let next = Some((version + 1, slot.messages.message(buffers, time_ns, true)));
            let mut pair = lock(&slot.value);
            let current = Arc::clone(pair.resolve(version));
            let old = std::mem::replace(&mut *pair, Pair { current, next });
            drop(pair);
            drop(old);
        }
        // Release: a reader that sees the new version also sees every pair above.
        self.version.store(version + 1, Ordering::Release);
        for slot in slots {
            slot.wake();
        }
        Ok(Timestamp::from_ns(time_ns))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::thread;

    use super::*;
    use crate::dlpack::ffi::DLDataType;
    use crate::time::TimerState;

    const I64: DLDataType = DLDataType {
        code: 0,
        bits: 64,
        lanes: 1,
    };

    fn timer() -> Arc<Timer> {
        let timer = Arc::new(Timer::new());
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
        Arc::new(Schema::new(vec![field("a", &[]), field("b", &[])], vec![]).unwrap())
    }

    fn scalar_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![field("", &[])], vec![]).unwrap())
    }

    /// Fills every field with `k`.
    fn fill_k(k: i64) -> impl FnOnce(&mut [&mut [u8]]) -> Result<(), SlotError> {
        move |bytes| {
            for b in bytes {
                b.copy_from_slice(&k.to_ne_bytes());
            }
            Ok(())
        }
    }

    /// An error for a `fill` to fail with.
    fn bad() -> SlotError {
        SlotError::Field {
            slot: "p/out".to_owned(),
            field: String::new(),
            error: FieldError::InvalidDLPack {
                reason: "bad".to_owned(),
            },
        }
    }

    fn values(msg: &Message) -> Vec<i64> {
        msg.values()
            .iter()
            .map(|b| i64::from_ne_bytes(b.bytes().try_into().unwrap()))
            .collect()
    }

    fn allocations(slot: &Slot) -> usize {
        slot.messages.pools().iter().map(|p| p.allocated()).sum()
    }

    #[test]
    fn starts_with_zeros_at_creation_time() {
        let timer = timer();
        timer.advance_ns(5).unwrap();
        let outlet = Outlet::new(pair_schema(), 1, "n.p", Arc::clone(&timer));
        let msg = outlet.channel(0).load();
        assert_eq!(values(&msg), [0, 0]);
        assert_eq!(msg.time_ns(), 5);
        assert_eq!(outlet.outbox().name(), "n.p/out");
        assert_eq!(outlet.channel(0).name(), "n.p/ch0");
    }

    #[test]
    fn send_to_all_shares_one_message() {
        let timer = timer();
        let outlet = Outlet::new(scalar_schema(), 3, "p", Arc::clone(&timer));
        timer.advance_ns(42).unwrap();
        outlet.send_with(Target::All, None, fill_k(7)).unwrap();
        let out = outlet.outbox().load();
        assert_eq!(values(&out), [7]);
        assert_eq!(out.time_ns(), 42);
        for slot in outlet.channels() {
            assert!(Arc::ptr_eq(&slot.load(), &out));
        }
    }

    #[test]
    fn only_written_messages_are_sent() {
        let outlet = Outlet::new(scalar_schema(), 2, "p", timer());
        assert!(!outlet.outbox().load().sent());
        outlet
            .send_with(Target::Channel(1), None, fill_k(1))
            .unwrap();
        assert!(outlet.channel(1).load().sent());
        assert!(!outlet.channel(0).load().sent());
        outlet.send_with(Target::All, None, fill_k(2)).unwrap();
        assert!(outlet.outbox().load().sent() && outlet.channel(0).load().sent());

        let group = InputGroup::new(timer());
        let input = group.input(scalar_schema(), "i");
        assert!(!input.load().sent());
        group.write(&[&input], |_, bytes| fill_k(3)(bytes)).unwrap();
        assert!(input.load().sent());
    }

    #[test]
    fn send_to_one_channel_changes_only_it() {
        let outlet = Outlet::new(scalar_schema(), 3, "p", timer());
        outlet
            .send_with(Target::Channel(1), None, fill_k(9))
            .unwrap();
        assert_eq!(values(&outlet.channel(1).load()), [9]);
        for slot in [outlet.outbox(), outlet.channel(0), outlet.channel(2)] {
            assert_eq!(values(&slot.load()), [0]);
        }
    }

    #[test]
    fn failed_fill_publishes_nothing() {
        let outlet = Outlet::new(scalar_schema(), 1, "p", timer());
        let before = outlet.outbox().load();
        let bad = bad();
        let failed = outlet.send_with(Target::All, None, |_| Err(bad.clone()));
        assert_eq!(failed, Err(bad));
        assert!(Arc::ptr_eq(&outlet.outbox().load(), &before));
    }

    #[test]
    fn held_message_survives_later_writes() {
        let outlet = Outlet::new(scalar_schema(), 0, "p", timer());
        outlet.send_with(Target::All, None, fill_k(1)).unwrap();
        let held = outlet.outbox().load();
        for k in 2..=100 {
            outlet.send_with(Target::All, None, fill_k(k)).unwrap();
        }
        assert_eq!(values(&held), [1]);
        assert_eq!(values(&outlet.outbox().load()), [100]);
    }

    #[test]
    fn pool_reuses_buffers_in_steady_state() {
        let outlet = Outlet::new(scalar_schema(), 1, "p", timer());
        let slot = outlet.outbox();
        for k in 0..1000 {
            outlet.send_with(Target::All, None, fill_k(k)).unwrap();
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
            outlet.send_with(Target::All, None, fill_k(k)).unwrap();
        }
        for r in readers {
            r.join().unwrap();
        }
    }

    /// A probabilistic smoke test against slots handing a reader the value of
    /// another slot, as the lock-free `arc-swap` once did. A message or input pair
    /// freed by one slot is often allocated again, at the same address, for the
    /// next write to another slot, and writing every slot from one thread makes
    /// that likely. Readers of `pairs` and of `input` (whose pairs have the same
    /// allocation size as messages) must still see only their own messages, and
    /// a reader of `relayed` only messages with its own value and the carry of
    /// `carrying`.
    ///
    /// Passing proves little: against `arc-swap` it failed in about half of the runs
    /// (`specs/native-core.md`).
    #[test]
    fn readers_never_see_a_message_of_another_slot() {
        const WRITES: i64 = 100_000;
        let pairs = Outlet::new(pair_schema(), 0, "pairs", timer());
        let scalars = Outlet::new(scalar_schema(), 0, "scalars", timer());
        let group = InputGroup::new(timer());
        let input = group.input(pair_schema(), "input");
        let carrying = Outlet::new(carry_schema(), 1, "carrying", timer());
        let relayed = Outlet::new(carry_schema(), 0, "relayed", timer());
        let done = Arc::new(AtomicBool::new(false));
        let reader = |load: Box<dyn Fn() -> Arc<Message> + Send>, expected: [&'static str; 2]| {
            let done = Arc::clone(&done);
            thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    let msg = load();
                    let fields: Vec<_> = msg
                        .values()
                        .iter()
                        .chain(msg.carry())
                        .map(|b| b.field().name.as_str())
                        .collect();
                    assert_eq!(fields, expected, "read a message of another slot");
                }
            })
        };
        let readers: Vec<_> = (0..4)
            .flat_map(|_| {
                let (slot, input) = (Arc::clone(pairs.outbox()), Arc::clone(&input));
                let relayed = Arc::clone(relayed.outbox());
                [
                    reader(Box::new(move || slot.load()), ["a", "b"]),
                    reader(Box::new(move || input.load()), ["a", "b"]),
                    reader(Box::new(move || relayed.load()), ["a", "c"]),
                ]
            })
            .collect();
        for k in 1..=WRITES {
            pairs.send_with(Target::All, None, fill_k(k)).unwrap();
            scalars.send_with(Target::All, None, fill_k(k)).unwrap();
            group.write(&[&input], |_, bytes| fill_k(k)(bytes)).unwrap();
            carrying.send_with(Target::All, None, fill_k(k)).unwrap();
            relay(&relayed, carrying.channel(0), k).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        for r in readers {
            r.join().unwrap();
        }
    }

    #[test]
    fn wait_newer_returns_on_write_or_timeout() {
        let outlet = Outlet::new(scalar_schema(), 0, "p", timer());
        let slot = Arc::clone(outlet.outbox());
        let first = slot.load();
        assert!(slot.wait_newer(&first, Duration::from_millis(20)).is_none());

        let waiter = {
            let (slot, first) = (Arc::clone(&slot), Arc::clone(&first));
            thread::spawn(move || slot.wait_newer(&first, Duration::from_secs(10)))
        };
        thread::sleep(Duration::from_millis(20));
        outlet.send_with(Target::All, None, fill_k(3)).unwrap();
        let newer = waiter.join().unwrap().expect("woken by the write");
        assert_eq!(values(&newer), [3]);
    }

    #[test]
    fn wait_newer_returns_on_a_write_of_the_host() {
        let group = InputGroup::new(timer());
        let input = group.input(scalar_schema(), "i");
        let first = input.load();
        let waiter = {
            let (input, first) = (Arc::clone(&input), Arc::clone(&first));
            thread::spawn(move || input.wait_newer(&first, Duration::from_secs(10)))
        };
        thread::sleep(Duration::from_millis(20));
        group.write(&[&input], |_, bytes| fill_k(3)(bytes)).unwrap();
        let newer = waiter.join().unwrap().expect("woken by the write");
        assert_eq!(values(&newer), [3]);
    }

    #[test]
    fn restore_sets_value_time_and_whether_sent() {
        let outlet = Outlet::new(scalar_schema(), 1, "p", timer());
        let slot = outlet.channel(0);
        slot.restore(123, true, fill_k(4)).unwrap();
        let msg = slot.load();
        assert_eq!(
            (values(&msg), msg.time_ns(), msg.sent()),
            (vec![4], 123, true)
        );
        assert_eq!(values(&outlet.outbox().load()), [0]);
        slot.restore(1, false, fill_k(4)).unwrap();
        assert!(!slot.load().sent());
    }

    #[test]
    fn input_group_write_stamps_one_time() {
        let timer = timer();
        let group = InputGroup::new(Arc::clone(&timer));
        let (a, b) = (
            group.input(scalar_schema(), "a"),
            group.input(scalar_schema(), "b"),
        );
        assert_eq!(a.name(), "a/in");
        timer.advance_ns(77).unwrap();
        let written_at = group
            .write(&[&a, &b], |i, bytes| fill_k(i as i64 + 1)(bytes))
            .unwrap();
        let (ma, mb) = (a.load(), b.load());
        assert_eq!((values(&ma), values(&mb)), (vec![1], vec![2]));
        assert_eq!((ma.time_ns(), mb.time_ns()), (77, 77));
        assert_eq!(written_at, Timestamp::from_ns(77));
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
        a.restore(9, true, fill_k(2)).unwrap();
        let msg = a.load();
        assert_eq!(
            (values(&msg), msg.time_ns(), msg.sent()),
            (vec![2], 9, true)
        );
        a.restore(9, false, fill_k(2)).unwrap();
        assert!(!a.load().sent());
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
        let reader = |first: Arc<Slot>, second: Arc<Slot>| {
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

    /// A scalar int64 value `a` carrying a scalar int64 `c`.
    fn carry_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![field("a", &[])], vec![field("c", &[])]).unwrap())
    }

    fn carried(msg: &Message) -> Vec<i64> {
        msg.carry()
            .iter()
            .map(|b| i64::from_ne_bytes(b.bytes().try_into().unwrap()))
            .collect()
    }

    /// How many buffers the pools of the value fields and of the carry fields of
    /// `slot`'s writer have allocated.
    fn split_allocations(slot: &Slot) -> (usize, usize) {
        let messages = &slot.messages;
        let count = |pools: &[Arc<crate::buffer::BufferPool>]| -> usize {
            pools.iter().map(|p| p.allocated()).sum()
        };
        let carry_from = messages.schema().values().len();
        let pools = messages.pools();
        (count(&pools[..carry_from]), count(&pools[carry_from..]))
    }

    /// How many buffers wait in the free list of the carry pool of `slot`'s writer.
    fn free_carry(slot: &Slot) -> usize {
        slot.messages.pools()[1].free()
    }

    /// Sends `k` as the value and passes on the carry of what `from` holds now.
    fn relay(outlet: &Outlet, from: &Slot, k: i64) -> Result<(), SlotError> {
        let msg = from.load();
        outlet.send_with(Target::All, Some(msg.carry()), fill_k(k))
    }

    #[test]
    fn relay_shares_the_carried_buffers() {
        let source = Outlet::new(carry_schema(), 1, "a", timer());
        let relayer = Outlet::new(carry_schema(), 1, "b", timer());
        source.send_with(Target::All, None, fill_k(7)).unwrap();
        relay(&relayer, source.channel(0), 8).unwrap();
        let (got, from) = (relayer.channel(0).load(), source.channel(0).load());
        assert_eq!((values(&got), carried(&got)), (vec![8], vec![7]));
        assert!(Arc::ptr_eq(&got.carry()[0], &from.carry()[0]));
        assert_eq!(
            split_allocations(relayer.outbox()),
            (2, 1),
            "only the zeros"
        );
    }

    #[test]
    fn relays_along_a_path_allocate_no_carry() {
        let a = Outlet::new(carry_schema(), 1, "a", timer());
        let b = Outlet::new(carry_schema(), 1, "b", timer());
        let c = Outlet::new(carry_schema(), 1, "c", timer());
        for k in 0..1000 {
            a.send_with(Target::All, None, fill_k(k)).unwrap();
            relay(&b, a.channel(0), k + 1).unwrap();
            relay(&c, b.channel(0), k + 2).unwrap();
            let end = c.channel(0).load();
            assert_eq!((values(&end), carried(&end)), (vec![k + 2], vec![k]));
        }
        // The relays only ever allocated the carry of their zeros.
        assert_eq!(split_allocations(b.outbox()).1, 1);
        assert_eq!(split_allocations(c.outbox()).1, 1);
        // The latest message of each of the three outlets holds one carry of a's,
        // and a fills one more while they do.
        let (_, made) = split_allocations(a.outbox());
        assert!(made <= 4, "allocated {made}");
    }

    #[test]
    fn relayed_buffers_return_to_the_pool_that_filled_them() {
        let source = Outlet::new(carry_schema(), 1, "a", timer());
        let relayer = Outlet::new(carry_schema(), 0, "b", timer());
        source.send_with(Target::All, None, fill_k(1)).unwrap();
        relay(&relayer, source.channel(0), 0).unwrap();
        // The source moves on, but the relayer still holds the carry of 1.
        source.send_with(Target::All, None, fill_k(2)).unwrap();
        let (free, own) = (free_carry(source.outbox()), free_carry(relayer.outbox()));
        relay(&relayer, source.channel(0), 0).unwrap();
        assert_eq!(free_carry(source.outbox()), free + 1);
        assert_eq!(
            free_carry(relayer.outbox()),
            own,
            "not to the relayer's pool"
        );
    }

    #[test]
    fn writers_fill_and_restore_the_carry() {
        let outlet = Outlet::new(carry_schema(), 1, "p", timer());
        let zeros = outlet.channel(0).load();
        assert_eq!((values(&zeros), carried(&zeros)), (vec![0], vec![0]));
        outlet.send_with(Target::All, None, fill_k(3)).unwrap();
        assert_eq!(carried(&outlet.outbox().load()), [3]);
        outlet.channel(0).restore(5, true, fill_k(4)).unwrap();
        assert_eq!(carried(&outlet.channel(0).load()), [4]);

        let group = InputGroup::new(timer());
        let input = group.input(carry_schema(), "i");
        assert_eq!(carried(&input.load()), [0]);
        group.write(&[&input], |_, bytes| fill_k(6)(bytes)).unwrap();
        assert_eq!(carried(&input.load()), [6]);
        input.restore(1, true, fill_k(9)).unwrap();
        assert_eq!(carried(&input.load()), [9]);
    }

    #[test]
    fn new_carry_comes_from_the_slots_pool() {
        let outlet = Outlet::new(carry_schema(), 1, "p", timer());
        let slot = outlet.channel(0);
        let carry = slot.new_carry(fill_k(5)).unwrap();
        assert_eq!(carry.len(), 1);
        assert_eq!(carry[0].field(), &field("c", &[]));
        assert_eq!(carry[0].bytes(), 5i64.to_ne_bytes());
        assert_eq!(split_allocations(slot), (1, 2));
        drop(carry);
        assert_eq!(free_carry(slot), 1);

        let group = InputGroup::new(timer());
        let input = group.input(carry_schema(), "i");
        let carry = input.new_carry(fill_k(6)).unwrap();
        assert_eq!(carry[0].bytes(), 6i64.to_ne_bytes());
    }

    #[test]
    fn mismatched_relay_publishes_nothing() {
        let source = Outlet::new(carry_schema(), 1, "a", timer());
        let other = Outlet::new(
            Arc::new(Schema::new(vec![field("a", &[])], vec![field("d", &[])]).unwrap()),
            1,
            "x",
            timer(),
        );
        let shaped = Outlet::new(
            Arc::new(Schema::new(vec![field("a", &[])], vec![field("c", &[2])]).unwrap()),
            1,
            "y",
            timer(),
        );
        let mut int32 = field("c", &[]);
        int32.dtype.bits = 32;
        let typed = Outlet::new(
            Arc::new(Schema::new(vec![field("a", &[])], vec![int32]).unwrap()),
            1,
            "z",
            timer(),
        );
        let plain = Outlet::new(pair_schema(), 1, "w", timer());
        for target in [&other, &shaped, &typed, &plain] {
            let before = target.outbox().load();
            let (values_before, _) = split_allocations(target.outbox());
            let err = relay(target, source.channel(0), 1).unwrap_err();
            assert!(matches!(err, SlotError::CarryMismatch { .. }), "{err}");
            assert!(Arc::ptr_eq(&target.outbox().load(), &before));
            assert_eq!(split_allocations(target.outbox()).0, values_before);
        }
        let err = relay(&other, source.channel(0), 1).unwrap_err().to_string();
        assert_eq!(
            err,
            "slot 'x/out': expected to carry ('d': () int64), got ('c': () int64)"
        );
        let err = plain
            .send_with(Target::Channel(0), Some(&[]), fill_k(1))
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "slot 'w/ch0': expected to carry nothing, so it passes nothing on"
        );
    }

    #[test]
    fn failed_fill_with_carry_publishes_nothing() {
        let outlet = Outlet::new(carry_schema(), 1, "p", timer());
        let source = Outlet::new(carry_schema(), 1, "a", timer());
        let before = outlet.outbox().load();
        let bad = bad();
        let failed = outlet.send_with(Target::All, None, |_| Err(bad.clone()));
        assert_eq!(failed, Err(bad.clone()));
        let msg = source.channel(0).load();
        let failed = outlet.send_with(Target::All, Some(msg.carry()), |_| Err(bad.clone()));
        assert_eq!(failed, Err(bad));
        assert!(Arc::ptr_eq(&outlet.outbox().load(), &before));
    }

    /// The point of carrying data in the message: a reader of a relayed path
    /// sees a value together with the carry of the very message it was made
    /// from, however the threads interleave.
    #[test]
    fn relayed_carry_stays_with_its_value() {
        const WRITES: i64 = 20_000;
        let source = Outlet::new(carry_schema(), 1, "a", timer());
        let relayer = Arc::new(Outlet::new(carry_schema(), 1, "b", timer()));
        let from = Arc::clone(source.channel(0));
        let done = Arc::new(AtomicBool::new(false));
        let relaying = {
            let (relayer, done) = (Arc::clone(&relayer), Arc::clone(&done));
            thread::spawn(move || {
                while !done.load(Ordering::Relaxed) {
                    let msg = from.load();
                    let k = values(&msg)[0];
                    relayer
                        .send_with(Target::All, Some(msg.carry()), fill_k(k))
                        .unwrap();
                }
            })
        };
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let (slot, done) = (Arc::clone(relayer.channel(0)), Arc::clone(&done));
                thread::spawn(move || {
                    let mut last = 0;
                    while !done.load(Ordering::Relaxed) {
                        let msg = slot.load();
                        let (v, c) = (values(&msg)[0], carried(&msg)[0]);
                        assert_eq!(v, c, "a value with the carry of another message");
                        assert!(v >= last, "went back from {last} to {v}");
                        last = v;
                    }
                })
            })
            .collect();
        for k in 1..=WRITES {
            source.send_with(Target::All, None, fill_k(k)).unwrap();
        }
        done.store(true, Ordering::Relaxed);
        relaying.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }
    }
}
