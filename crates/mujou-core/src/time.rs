//! Timers: the time axes the nodes run on.
//!
//! Every node discretizes its dynamics with the elapsed time measured on the
//! timer of its run, so the time can be scaled, paused and restored without
//! changing the dynamics (`specs/execution-model.md`). A run uses the global
//! timer ([`global`]) unless it is given a [`Timer`] of its own; systems that
//! should not stop each other's time run on separate timers.
//!
//! The time is held as nanoseconds (`u64`); seconds (`f64`) are derived from it.
//! It starts at 0 when the timer is created and saturates at `u64::MAX` (about
//! 584 years).
//!
//! For deterministic tests the time source can be switched to a virtual clock
//! that only moves when [`Timer::advance_ns`] is called.

use std::fmt;
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The process-wide default timer, used by every run not given its own.
pub fn global() -> &'static Arc<Timer> {
    static TIMER: LazyLock<Arc<Timer>> = LazyLock::new(|| Arc::new(Timer::new()));
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

/// What is needed to continue the time from where it was taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimerState {
    pub time_ns: u64,
    pub scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimerError {
    /// The scale must be finite and positive. Use pause to stop the time.
    InvalidScale(f64),
    /// A duration in seconds must be finite and non-negative.
    InvalidDuration(f64),
    /// Only the virtual clock can be advanced by hand.
    NotVirtual,
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
            Self::NotVirtual => write!(f, "only the virtual clock can be advanced"),
            Self::Paused => write!(f, "cannot advance the time while paused"),
        }
    }
}

impl std::error::Error for TimerError {}

/// How [`Timer::wait_until_ns_or`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wait {
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

fn check_scale(scale: f64) -> Result<(), TimerError> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err(TimerError::InvalidScale(scale))
    }
}

/// A time axis that can be scaled, paused and restored. Shared as
/// `Arc<Timer>` by everything that runs on it.
pub struct Timer {
    state: Mutex<State>,
    /// Notified whenever the time may have jumped or its rate changed.
    changed: Condvar,
}

/// The time is `anchor_ns + (now - anchor_instant) * scale`, frozen at
/// `anchor_ns` while paused or on the virtual clock.
struct State {
    anchor_ns: u64,
    anchor_instant: Instant,
    scale: f64,
    /// Paused by [`Timer::pause`].
    paused: bool,
    /// The number of live [`PauseHold`]s.
    holds: usize,
    is_virtual: bool,
}

impl State {
    fn is_paused(&self) -> bool {
        self.paused || self.holds > 0
    }

    fn is_frozen(&self) -> bool {
        self.is_paused() || self.is_virtual
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

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

impl Timer {
    /// A running timer on the real clock at time 0 and scale 1.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                anchor_ns: 0,
                anchor_instant: Instant::now(),
                scale: 1.0,
                paused: false,
                holds: 0,
                is_virtual: false,
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

    pub fn now(&self) -> Timestamp {
        Timestamp(self.time_ns())
    }

    pub fn time(&self) -> f64 {
        self.time_ns() as f64 / 1e9
    }

    pub fn scale(&self) -> f64 {
        self.lock().scale
    }

    /// Sets how many seconds of this timer pass per real second.
    pub fn set_scale(&self, scale: f64) -> Result<(), TimerError> {
        check_scale(scale)?;
        self.update(|s| {
            s.reanchor();
            s.scale = scale;
        });
        Ok(())
    }

    /// Freezes the time until [`Timer::resume`]. For the host and tests.
    ///
    /// This is one switch, not a count: pausing twice needs one resume. It is
    /// independent of the [`PauseHold`]s, so the time stays frozen while this
    /// switch or any hold is on, and `resume` never undoes a hold. Pausing while
    /// paused leaves the time as it is.
    pub fn pause(&self) {
        self.update(|s| {
            s.reanchor();
            s.paused = true;
        });
    }

    /// Turns off the switch of [`Timer::pause`]. The time runs again from
    /// where it was frozen if no [`PauseHold`] is left.
    pub fn resume(&self) {
        self.update(|s| {
            s.reanchor();
            s.paused = false;
        });
    }

    /// Freezes the time until every hold taken so far is dropped (and the
    /// switch of [`Timer::pause`] is off).
    ///
    /// Each controller of a running system (a runtime starting, or pausing to
    /// take the state) takes its own hold, so that ending one freeze cannot let
    /// the time run during another one: several systems may run on one timer
    /// (all on the global one by default).
    pub fn hold(self: &Arc<Self>) -> PauseHold {
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
        self.lock().is_paused()
    }

    /// The time and scale. Whether paused and which clock is used are not part
    /// of the state: the state is taken while paused, and the clock is a test
    /// setting.
    pub fn state(&self) -> TimerState {
        let s = self.lock();
        TimerState {
            time_ns: s.now_ns(),
            scale: s.scale,
        }
    }

    /// Continues the time from `state`.
    pub fn load_state(&self, state: TimerState) -> Result<(), TimerError> {
        check_scale(state.scale)?;
        self.update(|s| {
            s.anchor_ns = state.time_ns;
            s.anchor_instant = Instant::now();
            s.scale = state.scale;
        });
        Ok(())
    }

    /// Switches between the real clock and the virtual clock, keeping the time
    /// continuous.
    pub fn use_virtual_clock(&self, enabled: bool) {
        self.update(|s| {
            s.reanchor();
            s.is_virtual = enabled;
        });
    }

    /// Moves the virtual clock forward. The scale does not apply.
    pub fn advance_ns(&self, dt_ns: u64) -> Result<(), TimerError> {
        self.update(|s| {
            if !s.is_virtual {
                return Err(TimerError::NotVirtual);
            }
            if s.is_paused() {
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
    pub fn wait_until_ns_or(
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
            let mut wait = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => Duration::MAX,
            };
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
    pub fn interrupt_waiters(&self) {
        // Taking the lock orders this after any waiter's check of its condition.
        let _s = self.lock();
        self.changed.notify_all();
    }
}

/// A reason to keep the time frozen; dropping it removes the reason.
#[must_use = "the time runs again as soon as the hold is dropped"]
pub struct PauseHold {
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
