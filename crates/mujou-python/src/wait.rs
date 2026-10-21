//! Waiting on the host, and the seconds it waits for.

use std::time::{Duration, Instant};

use mujou_core::time::seconds_to_ns;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// How long a wait without the GIL lasts before checking for signals (Ctrl-C).
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);

/// Waits without the GIL until `attempt` returns `Some` or `deadline` passes
/// (never if `None`), checking for signals in between: `attempt` is given
/// how long it may block, at most `SIGNAL_CHECK_INTERVAL`, and returns `None`
/// if what it waits for did not come in that time.
pub(crate) fn wait_interruptibly<T: Send>(
    py: Python<'_>,
    deadline: Option<Instant>,
    mut attempt: impl FnMut(Duration) -> Option<T> + Send,
) -> PyResult<Option<T>> {
    loop {
        let remaining = deadline.map_or(Duration::MAX, |d| {
            d.saturating_duration_since(Instant::now())
        });
        let step = remaining.min(SIGNAL_CHECK_INTERVAL);
        if let Some(result) = py.detach(|| attempt(step)) {
            return Ok(Some(result));
        }
        if remaining <= SIGNAL_CHECK_INTERVAL {
            return Ok(None);
        }
        py.check_signals()?;
    }
}

/// `seconds` in nanoseconds, rounded to the nearest.
///
/// Raises `ValueError` if `seconds` is negative or not finite.
pub(crate) fn to_ns(seconds: f64) -> PyResult<u64> {
    seconds_to_ns(seconds).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// `seconds` of real time to wait, rounded to the nearest nanosecond.
/// Infinity is `Duration::MAX`, which never expires.
///
/// Raises `ValueError` if `seconds` is negative or NaN.
pub(crate) fn to_duration(seconds: f64) -> PyResult<Duration> {
    if seconds == f64::INFINITY {
        return Ok(Duration::MAX);
    }
    to_ns(seconds).map(Duration::from_nanos)
}
