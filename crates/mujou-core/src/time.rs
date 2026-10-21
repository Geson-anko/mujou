//! Timers: the time axes the nodes run on.
//!
//! Every node discretizes its dynamics with the elapsed time measured on the
//! timer of its run, so the time can be scaled and paused without changing the
//! dynamics (`specs/execution-model.md`). A run uses the global timer
//! ([`global`]) unless it is given a [`Timer`] of its own; systems that should
//! not stop each other's time run on separate timers. A run reads its timer
//! only through its [`RunClock`], which stops at the run's end.
//!
//! The time is held as nanoseconds (`u64`); seconds (`f64`) are derived from it.
//! It saturates at `u64::MAX` (about 584 years).

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The process-wide default timer, used by every run not given its own. It
/// starts at 0 when first used.
pub fn global() -> &'static Arc<Timer> {
    static TIMER: LazyLock<Arc<Timer>> =
        LazyLock::new(|| Arc::new(Timer::new(Source::Real, Timestamp::from_ns(0))));
    &TIMER
}

/// A time on a timer, in nanoseconds since the timer's 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp(u64);

impl Timestamp {
    pub const fn from_ns(ns: u64) -> Self {
        Self(ns)
    }

    pub const fn as_ns(self) -> u64 {
        self.0
    }

    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1e9
    }

    /// The whole seconds, rounded down.
    pub const fn whole_secs(self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// The time from `other` to `self` in seconds, negative when `other` is
    /// later: the difference is exact in `i128`, then rounded once to `f64` and
    /// once by the division, as Python rounds `(a.ns - b.ns) / 1e9`.
    pub fn secs_since(self, other: Self) -> f64 {
        (i128::from(self.0) - i128::from(other.0)) as f64 / 1e9
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerError {
    /// The scale must be finite and positive. Use pause to stop the time.
    InvalidScale(f64),
    /// A duration in seconds must be finite and non-negative.
    InvalidDuration(f64),
    /// The time is paused and must not move.
    Paused,
}

impl fmt::Display for TimerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScale(scale) => {
                write!(f, "time scale must be finite and > 0, got {scale}")
            }
            Self::InvalidDuration(seconds) => {
                write!(f, "duration must be finite and >= 0 seconds, got {seconds}")
            }
            Self::Paused => write!(f, "cannot advance the time while paused"),
        }
    }
}

impl std::error::Error for TimerError {}

/// How [`Timer::wait_until_ns_or`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Wait {
    Reached,
    Interrupted,
    TimedOut,
}

/// Converts seconds to the nearest nanosecond.
pub fn seconds_to_ns(seconds: f64) -> Result<u64, TimerError> {
    if !(seconds.is_finite() && seconds >= 0.0) {
        return Err(TimerError::InvalidDuration(seconds));
    }
    Ok((seconds * 1e9).round() as u64)
}

/// What is left of a wait until `deadline` in real time (no end if `None`).
pub(crate) fn remaining(deadline: Option<Instant>) -> Duration {
    deadline.map_or(Duration::MAX, |d| {
        d.saturating_duration_since(Instant::now())
    })
}

/// What moves a timer's time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The real clock, at the timer's scale.
    Real,
    /// Only [`Timer::advance_ns`], for deterministic tests.
    Virtual,
}

/// A time axis that can be scaled and paused. Shared as `Arc<Timer>` by
/// everything that runs on it.
pub struct Timer {
    state: Mutex<State>,
    /// Notified whenever the time may have jumped or its rate changed.
    changed: Condvar,
}

/// The time is `anchor_ns + (now - anchor_instant) * scale`, frozen at
/// `anchor_ns` while held or on the virtual clock.
struct State {
    anchor_ns: u64,
    anchor_instant: Instant,
    scale: f64,
    /// The reasons to keep the time frozen: each live [`PauseHold`], and the
    /// host's [`Timer::pause`] as one more while it is on.
    holds: usize,
    /// The host's [`Timer::pause`] is on (and counted in `holds`).
    paused: bool,
    source: Source,
}

impl State {
    fn is_frozen(&self) -> bool {
        self.holds > 0 || self.source == Source::Virtual
    }

    fn ns_at(&self, instant: Instant) -> u64 {
        if self.is_frozen() {
            return self.anchor_ns;
        }
        let real_ns = instant.duration_since(self.anchor_instant).as_nanos() as f64;
        self.anchor_ns.saturating_add((real_ns * self.scale) as u64)
    }

    fn now_ns(&self) -> u64 {
        self.ns_at(Instant::now())
    }

    /// Moves the anchor to now, before the rate of the time changes.
    fn reanchor(&mut self) {
        let now = Instant::now();
        self.anchor_ns = self.ns_at(now);
        self.anchor_instant = now;
    }
}

impl Timer {
    /// A running timer at `start` and scale 1, moved by `source`.
    pub fn new(source: Source, start: Timestamp) -> Self {
        Self {
            state: Mutex::new(State {
                anchor_ns: start.as_ns(),
                anchor_instant: Instant::now(),
                scale: 1.0,
                holds: 0,
                paused: false,
                source,
            }),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        // Nothing panics while holding the lock, so a poisoned state is still valid.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Changes the state under the lock and wakes every waiter.
    fn update<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        let result = f(&mut self.lock());
        self.changed.notify_all();
        result
    }

    pub fn time_ns(&self) -> u64 {
        self.lock().now_ns()
    }

    /// Calls `f` with the time now, under the timer's lock: what `f` does is
    /// ordered with every other read of the time.
    fn at_now<T>(&self, f: impl FnOnce(u64) -> T) -> T {
        let s = self.lock();
        f(s.now_ns())
    }

    pub fn now(&self) -> Timestamp {
        Timestamp(self.time_ns())
    }

    pub fn time_scale(&self) -> f64 {
        self.lock().scale
    }

    /// Sets how many seconds of this timer pass per real second. The virtual
    /// clock keeps it but does not apply it.
    pub fn set_time_scale(&self, scale: f64) -> Result<(), TimerError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(TimerError::InvalidScale(scale));
        }
        self.update(|s| {
            s.reanchor();
            s.scale = scale;
        });
        Ok(())
    }

    /// Freezes the time until [`Timer::resume`]. For the host.
    ///
    /// This is one switch, not a count: pausing twice needs one resume. It
    /// counts as one hold while it is on, so the time stays frozen while it or
    /// any hold of a runtime is left, and `resume` never undoes a hold.
    pub fn pause(&self) {
        self.update(|s| {
            if !s.paused {
                s.reanchor();
                s.paused = true;
                s.holds += 1;
            }
        });
    }

    /// Turns off the switch of [`Timer::pause`]. The time runs again from
    /// where it was frozen if no hold is left.
    pub fn resume(&self) {
        self.update(|s| {
            if s.paused {
                s.reanchor();
                s.paused = false;
                s.holds -= 1;
            }
        });
    }

    /// Freezes the time until the hold is dropped (and no other is left).
    ///
    /// Each controller of a running system (a runtime starting, or pausing to
    /// take the state) takes its own hold, so that ending one freeze cannot let
    /// the time run during another one: several systems may run on one timer
    /// (all on the global one by default).
    pub(crate) fn hold(self: &Arc<Self>) -> PauseHold {
        self.update(|s| {
            s.reanchor();
            s.holds += 1;
        });
        PauseHold {
            timer: Arc::clone(self),
        }
    }

    /// Whether the time is frozen by [`Timer::pause`] or a hold. The
    /// virtual clock alone does not count as paused.
    pub fn is_paused(&self) -> bool {
        self.lock().holds > 0
    }

    /// Moves the time forward by `dt_ns`, at once. The scale does not apply.
    pub fn advance_ns(&self, dt_ns: u64) -> Result<(), TimerError> {
        self.update(|s| {
            if s.holds > 0 {
                return Err(TimerError::Paused);
            }
            s.anchor_ns = s.anchor_ns.saturating_add(dt_ns);
            Ok(())
        })
    }

    /// Blocks until the time reaches `target_ns` or `timeout` of real time
    /// passes, and returns whether the target was reached.
    pub fn wait_until_ns(&self, target_ns: u64, timeout: Duration) -> bool {
        self.wait_until_ns_or(target_ns, timeout, || false) == Wait::Reached
    }

    /// Blocks until the time reaches `target_ns`, `interrupted` returns true, or
    /// `timeout` of real time passes.
    ///
    /// `interrupted` is checked under the timer's lock, before every wait. A
    /// waker that makes it true and then calls [`Timer::interrupt_waiters`]
    /// is therefore never missed: either the waiter sees the condition before it
    /// sleeps, or it is already sleeping when the notification comes. Because it
    /// runs under the timer's lock, `interrupted` must not take a lock that is
    /// held while calling into the timer (read an atomic instead).
    pub(crate) fn wait_until_ns_or(
        &self,
        target_ns: u64,
        timeout: Duration,
        interrupted: impl Fn() -> bool,
    ) -> Wait {
        // A timeout too long to represent never expires.
        let deadline = Instant::now().checked_add(timeout);
        let mut s = self.lock();
        loop {
            let now_ns = s.now_ns();
            if now_ns >= target_ns {
                return Wait::Reached;
            }
            if interrupted() {
                return Wait::Interrupted;
            }
            let mut wait = remaining(deadline);
            if wait.is_zero() {
                return Wait::TimedOut;
            }
            if !s.is_frozen() {
                let real_ns = ((target_ns - now_ns) as f64 / s.scale).ceil();
                wait = wait.min(Duration::from_nanos(real_ns as u64));
            }
            s = self
                .changed
                .wait_timeout(s, wait)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Wakes every [`Timer::wait_until_ns_or`] so that it checks its
    /// `interrupted` condition again.
    pub(crate) fn interrupt_waiters(&self) {
        // Taking the lock orders this after any waiter's check of its condition.
        let _s = self.lock();
        self.changed.notify_all();
    }
}

/// A reason to keep the time frozen; dropping it removes the reason.
#[must_use = "the time runs again as soon as the hold is dropped"]
pub(crate) struct PauseHold {
    timer: Arc<Timer>,
}

impl Drop for PauseHold {
    fn drop(&mut self) {
        self.timer.update(|s| {
            s.reanchor();
            s.holds -= 1;
        });
    }
}

/// The time of one run: its timer's time until the run ends, and the time of
/// the end from then on.
///
/// Everything of a run reads the time here: the Δt and the last updates of
/// its nodes, and the times of its messages. So nothing of a run is ever
/// later than its end, and a state taken after the end is of the end. Only
/// the runtime ends it.
pub struct RunClock {
    timer: Arc<Timer>,
    /// The time of the end, `u64::MAX` until then.
    end_ns: AtomicU64,
}

impl RunClock {
    pub fn new(timer: Arc<Timer>) -> Self {
        Self {
            timer,
            end_ns: AtomicU64::new(u64::MAX),
        }
    }

    /// The timer's time, but never later than the end. The end is set and read
    /// under the timer's lock, so no time is later than it, read after the end
    /// or before: the timer's time never goes back.
    pub fn now_ns(&self) -> u64 {
        self.timer
            .at_now(|now_ns| now_ns.min(self.end_ns.load(Ordering::Relaxed)))
    }

    pub fn now(&self) -> Timestamp {
        Timestamp(self.now_ns())
    }

    pub fn time_scale(&self) -> f64 {
        self.timer.time_scale()
    }

    /// Stops the time of the run at the timer's time now, once.
    pub(crate) fn end(&self) {
        self.timer
            .at_now(|now_ns| self.end_ns.fetch_min(now_ns, Ordering::Relaxed));
    }

    pub(crate) fn hold(&self) -> PauseHold {
        self.timer.hold()
    }

    /// [`Timer::wait_until_ns_or`] on the timer.
    pub(crate) fn wait_until_ns_or(
        &self,
        target_ns: u64,
        timeout: Duration,
        interrupted: impl Fn() -> bool,
    ) -> Wait {
        self.timer.wait_until_ns_or(target_ns, timeout, interrupted)
    }

    pub(crate) fn interrupt_waiters(&self) {
        self.timer.interrupt_waiters();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::thread;

    use super::*;

    fn virtual_timer() -> Arc<Timer> {
        Arc::new(Timer::new(Source::Virtual, Timestamp::from_ns(0)))
    }

    mod hold {
        use super::*;

        #[test]
        fn the_time_runs_again_only_when_every_hold_is_dropped() {
            let timer = Arc::new(Timer::new(Source::Real, Timestamp::from_ns(0)));
            let first = timer.hold();
            let second = timer.hold();
            let frozen = timer.time_ns();
            thread::sleep(Duration::from_millis(2));
            drop(first);
            assert!(timer.is_paused());
            assert_eq!(timer.time_ns(), frozen);
            drop(second);
            assert!(!timer.is_paused());
        }

        #[test]
        fn resume_does_not_undo_a_hold() {
            let timer = virtual_timer();
            let hold = timer.hold();
            timer.resume();
            assert!(timer.is_paused());
            drop(hold);
            assert!(!timer.is_paused());
        }

        #[test]
        fn dropping_a_hold_does_not_undo_pause() {
            let timer = virtual_timer();
            timer.pause();
            drop(timer.hold());
            assert!(timer.is_paused());
            timer.resume();
            assert!(!timer.is_paused());
        }

        #[test]
        fn a_held_timer_cannot_advance() {
            let timer = virtual_timer();
            let _hold = timer.hold();
            assert!(matches!(timer.advance_ns(1), Err(TimerError::Paused)));
        }

        #[test]
        fn a_hold_freezes_only_its_own_timer() {
            let (a, b) = (virtual_timer(), virtual_timer());
            let _hold = a.hold();
            assert!(a.is_paused());
            assert!(!b.is_paused());
            b.advance_ns(1).unwrap();
            assert!(matches!(a.advance_ns(1), Err(TimerError::Paused)));
        }
    }

    mod wait {
        use super::*;

        #[test]
        fn wakes_on_interrupt() {
            let timer = virtual_timer();
            let flag = Arc::new(AtomicBool::new(false));
            let waiter = {
                let (timer, flag) = (Arc::clone(&timer), Arc::clone(&flag));
                thread::spawn(move || {
                    timer.wait_until_ns_or(1, Duration::MAX, || flag.load(Ordering::SeqCst))
                })
            };
            thread::sleep(Duration::from_millis(5));
            flag.store(true, Ordering::SeqCst);
            timer.interrupt_waiters();
            assert_eq!(waiter.join().unwrap(), Wait::Interrupted);
        }

        #[test]
        fn wakes_at_the_target() {
            let timer = virtual_timer();
            let waiter = {
                let timer = Arc::clone(&timer);
                thread::spawn(move || timer.wait_until_ns_or(5, Duration::MAX, || false))
            };
            timer.advance_ns(5).unwrap();
            assert_eq!(waiter.join().unwrap(), Wait::Reached);
            assert_eq!(
                timer.wait_until_ns_or(6, Duration::from_millis(5), || false),
                Wait::TimedOut
            );
        }
    }

    mod run_clock {
        use super::*;

        #[test]
        fn follows_the_timer_until_the_end_and_stays_there() {
            let timer = virtual_timer();
            let clock = RunClock::new(Arc::clone(&timer));
            timer.advance_ns(5).unwrap();
            assert_eq!(clock.now_ns(), 5);
            clock.end();
            timer.advance_ns(3).unwrap();
            assert_eq!((clock.now_ns(), timer.time_ns()), (5, 8));
            // A second end does not move it.
            clock.end();
            assert_eq!(clock.now(), Timestamp::from_ns(5));
        }

        #[test]
        fn holds_its_timer() {
            let timer = virtual_timer();
            let clock = RunClock::new(Arc::clone(&timer));
            let hold = clock.hold();
            assert!(timer.is_paused());
            drop(hold);
            assert!(!timer.is_paused());
        }

        #[test]
        fn no_time_read_while_it_ends_is_later_than_the_end() {
            let clock = Arc::new(RunClock::new(Arc::new(Timer::new(
                Source::Real,
                Timestamp::from_ns(0),
            ))));
            let ended = Arc::new(AtomicBool::new(false));
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let (clock, ended) = (Arc::clone(&clock), Arc::clone(&ended));
                    thread::spawn(move || {
                        let mut latest = 0;
                        // Some reads after the end too.
                        for _ in 0..1000 {
                            latest = latest.max(clock.now_ns());
                        }
                        while !ended.load(Ordering::SeqCst) {
                            latest = latest.max(clock.now_ns());
                        }
                        for _ in 0..1000 {
                            latest = latest.max(clock.now_ns());
                        }
                        latest
                    })
                })
                .collect();
            thread::sleep(Duration::from_millis(1));
            clock.end();
            ended.store(true, Ordering::SeqCst);
            let end_ns = clock.now_ns();
            for reader in readers {
                assert!(reader.join().unwrap() <= end_ns);
            }
        }
    }
}
