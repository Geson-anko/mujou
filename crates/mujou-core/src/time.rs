//! The global timer: the single time axis of the whole system.
//!
//! Every node discretizes its dynamics with the elapsed time measured on this
//! timer, so the time can be scaled, paused and restored without changing the
//! dynamics (`docs/execution-model.md`). The time is held as nanoseconds (`u64`);
//! seconds (`f64`) are derived from it. It starts at 0 when the timer is created.
//!
//! For deterministic tests the time source can be switched to a virtual clock
//! that only moves when [`GlobalTimer::advance_ns`] is called.

use std::fmt;
use std::sync::{Condvar, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// The process-wide timer.
pub fn global() -> &'static GlobalTimer {
    static TIMER: LazyLock<GlobalTimer> = LazyLock::new(GlobalTimer::new);
    &TIMER
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

pub struct GlobalTimer {
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
    paused: bool,
    is_virtual: bool,
}

impl State {
    fn is_frozen(&self) -> bool {
        self.paused || self.is_virtual
    }

    fn now_ns(&self) -> u64 {
        if self.is_frozen() {
            return self.anchor_ns;
        }
        let real_ns = self.anchor_instant.elapsed().as_nanos() as f64;
        self.anchor_ns + (real_ns * self.scale) as u64
    }

    /// Moves the anchor to now, before the rate of the time changes.
    fn reanchor(&mut self) {
        self.anchor_ns = self.now_ns();
        self.anchor_instant = Instant::now();
    }
}

impl Default for GlobalTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalTimer {
    /// A running timer on the real clock at time 0 and scale 1.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(State {
                anchor_ns: 0,
                anchor_instant: Instant::now(),
                scale: 1.0,
                paused: false,
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

    pub fn time(&self) -> f64 {
        self.time_ns() as f64 * 1e-9
    }

    pub fn scale(&self) -> f64 {
        self.lock().scale
    }

    /// Sets how many global seconds pass per real second.
    pub fn set_scale(&self, scale: f64) -> Result<(), TimerError> {
        check_scale(scale)?;
        self.update(|s| {
            s.reanchor();
            s.scale = scale;
        });
        Ok(())
    }

    /// Freezes the time. Does nothing if already paused.
    pub fn pause(&self) {
        self.update(|s| {
            s.reanchor();
            s.paused = true;
        });
    }

    /// Lets the time run again from where it was frozen.
    pub fn resume(&self) {
        self.update(|s| {
            s.reanchor();
            s.paused = false;
        });
    }

    pub fn is_paused(&self) -> bool {
        self.lock().paused
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
            if s.paused {
                return Err(TimerError::Paused);
            }
            s.anchor_ns += dt_ns;
            Ok(())
        })
    }

    /// Blocks until the time reaches `target_ns` or `timeout` of real time
    /// passes, and returns whether the target was reached.
    pub fn wait_until_ns(&self, target_ns: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut s = self.lock();
        loop {
            let now_ns = s.now_ns();
            if now_ns >= target_ns {
                return true;
            }
            let Some(mut wait) = deadline.checked_duration_since(Instant::now()) else {
                return false;
            };
            if wait.is_zero() {
                return false;
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

    /// Blocks for `dt_ns` of global time. Waits through pauses.
    pub fn sleep_ns(&self, dt_ns: u64) {
        let target_ns = self.time_ns() + dt_ns;
        while !self.wait_until_ns(target_ns, Duration::from_secs(3600)) {}
    }
}
