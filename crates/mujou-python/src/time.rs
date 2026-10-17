//! Bindings of `mujou_core::time`: timers and timestamps.

use std::sync::Arc;

use mujou_core::time::{self as core, Timer, TimerState, Timestamp, seconds_to_ns};
use pyo3::exceptions::PyValueError;
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{PyClassInitializer, PyTypeInfo};

use crate::{SIGNAL_CHECK_INTERVAL, timer_error};

/// A time axis that nodes discretize their dynamics on.
///
/// `Timer()` is a new timer at time 0 on the real clock at scale 1. Systems
/// whose pauses should not stop each other's time run on separate timers;
/// everything not given a timer runs on the global one, `timer`.
///
/// The time is held in nanoseconds and saturates at about 584 years.
#[pyclass(subclass, frozen, module = "mujou._core._native", name = "Timer")]
pub struct PyTimer(Arc<Timer>);

impl PyTimer {
    pub fn timer(&self) -> &Arc<Timer> {
        &self.0
    }
}

/// The timer given as an optional argument, or the global timer.
pub fn timer_or_global(timer: Option<PyRef<'_, PyTimer>>) -> Arc<Timer> {
    timer.map_or_else(|| Arc::clone(core::global()), |t| Arc::clone(t.timer()))
}

#[pymethods]
impl PyTimer {
    #[new]
    fn new() -> Self {
        Self(Arc::new(Timer::new()))
    }

    /// The time in seconds.
    fn time(&self) -> f64 {
        self.0.time()
    }

    /// The time in nanoseconds.
    fn time_ns(&self) -> u64 {
        self.0.time_ns()
    }

    /// The time as a `Timestamp`.
    fn now(&self) -> PyTimestamp {
        PyTimestamp(self.0.now())
    }

    /// Blocks for `seconds` of this timer's time, following the scale and
    /// pauses: while paused, it does not return until resumed. Waits without
    /// the GIL and checks for signals (Ctrl-C) every 50 ms.
    ///
    /// Raises:
    ///     ValueError: `seconds` is negative or not finite.
    fn sleep(&self, py: Python<'_>, seconds: f64) -> PyResult<()> {
        let dt_ns = seconds_to_ns(seconds).map_err(timer_error)?;
        let target_ns = self.0.time_ns().saturating_add(dt_ns);
        while !py.detach(|| self.0.wait_until_ns(target_ns, SIGNAL_CHECK_INTERVAL)) {
            py.check_signals()?;
        }
        Ok(())
    }

    /// Seconds of this timer per real second.
    fn time_scale(&self) -> f64 {
        self.0.scale()
    }

    /// A `TimerReader` of this timer: it reads the time and the scale and has
    /// no way to change them.
    fn reader(&self) -> PyTimerReader {
        PyTimerReader(Arc::clone(&self.0))
    }

    /// Sets how many seconds of this timer pass per real second.
    ///
    /// Raises:
    ///     ValueError: `scale` is not finite and > 0. Use `pause` to stop the
    ///         time.
    fn set_time_scale(&self, scale: f64) -> PyResult<()> {
        self.0.set_scale(scale).map_err(timer_error)
    }

    /// Freezes the time until `resume`. For hosts and tests: the runtime
    /// freezes the time with its own holds, which this switch neither ends nor
    /// is ended by. Pausing twice needs one resume.
    fn pause(&self) {
        self.0.pause();
    }

    /// Turns off the switch set by `pause`. The time runs again only when no
    /// runtime holds it frozen either.
    fn resume(&self) {
        self.0.resume();
    }

    /// Whether the time is frozen by `pause` or by a runtime. The virtual clock
    /// alone does not count as paused.
    fn is_paused(&self) -> bool {
        self.0.is_paused()
    }

    /// `{"time_ns": int, "scale": float}`: what is needed to continue the time
    /// from where it was taken. Take it while paused.
    #[pyo3(signature = () -> "dict[str, int | float]")]
    fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = self.0.state();
        let dict = PyDict::new(py);
        dict.set_item("time_ns", state.time_ns)?;
        dict.set_item("scale", state.scale)?;
        Ok(dict)
    }

    /// Continues the time from a `state_dict`.
    ///
    /// Raises:
    ///     ValueError: The scale is not finite and > 0.
    #[pyo3(signature = (state: "dict[str, int | float]"))]
    fn load_state_dict(&self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        self.0
            .load_state(extract_state(state)?)
            .map_err(timer_error)
    }

    /// Switches to the virtual clock, which moves only by `advance`, or back to
    /// the real clock. The time stays continuous. For tests.
    fn use_virtual_clock(&self, enabled: bool) {
        self.0.use_virtual_clock(enabled);
    }

    /// Moves the virtual clock forward by `seconds`. The scale does not apply.
    ///
    /// Raises:
    ///     ValueError: `seconds` is negative or not finite.
    ///     RuntimeError: The real clock is in use, or the time is paused.
    fn advance(&self, seconds: f64) -> PyResult<()> {
        let dt_ns = seconds_to_ns(seconds).map_err(timer_error)?;
        self.0.advance_ns(dt_ns).map_err(timer_error)
    }
}

/// Reads `{"time_ns": int, "scale": float}`.
pub fn extract_state(state: &Bound<'_, PyAny>) -> PyResult<TimerState> {
    Ok(TimerState {
        time_ns: state.get_item("time_ns")?.extract()?,
        scale: state.get_item("scale")?.extract()?,
    })
}

/// The global timer, which every run not given a timer of its own uses. The
/// only instance is `timer`.
#[pyclass(extends = PyTimer, frozen, module = "mujou._core._native")]
pub struct GlobalTimer;

// `#[pyclass]` does not implement this for a subclass of another pyclass,
// whose instance needs the base's value too. It lets `timer` be a module const.
impl<'py> IntoPyObject<'py> for GlobalTimer {
    type Target = GlobalTimer;
    type Output = Bound<'py, GlobalTimer>;
    type Error = PyErr;
    const OUTPUT_TYPE: PyStaticExpr = <GlobalTimer as PyTypeInfo>::TYPE_HINT;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        let base = PyTimer(Arc::clone(core::global()));
        Bound::new(py, PyClassInitializer::from(base).add_subclass(self))
    }
}

/// The reading side of a `Timer`: the time and the scale, and nothing that
/// changes them. Made by `Timer.reader()`.
#[pyclass(frozen, module = "mujou._core._native", name = "TimerReader")]
pub struct PyTimerReader(Arc<Timer>);

#[pymethods]
impl PyTimerReader {
    /// The time as a `Timestamp`.
    fn now(&self) -> PyTimestamp {
        PyTimestamp(self.0.now())
    }

    /// Seconds of the timer per real second.
    fn time_scale(&self) -> f64 {
        self.0.scale()
    }
}

/// A time on a timer.
///
/// `float(ts)` is the time in seconds and `int(ts)` its whole seconds; `ts.ns`
/// is the exact time in nanoseconds. Timestamps compare and hash by time, so
/// compare only those of one timer. `later - earlier` is the time between them,
/// as a `Timestamp`. Timestamps pickle and copy.
#[pyclass(
    frozen,
    eq,
    ord,
    hash,
    module = "mujou._core._native",
    name = "Timestamp"
)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyTimestamp(pub Timestamp);

#[pymethods]
impl PyTimestamp {
    /// Takes the time in nanoseconds, by keyword only: a bare number could
    /// mean seconds.
    ///
    /// Raises:
    ///     ValueError: `ns` is negative or does not fit in 64 bits.
    #[new]
    #[pyo3(signature = (*, ns))]
    fn new(ns: i128) -> PyResult<Self> {
        let ns = u64::try_from(ns).map_err(|_| {
            PyValueError::new_err(format!(
                "a timestamp is from 0 to 2**64 - 1 nanoseconds, got {ns}"
            ))
        })?;
        Ok(Self(Timestamp::from_ns(ns)))
    }

    /// The time in nanoseconds.
    #[getter]
    fn ns(&self) -> u64 {
        self.0.as_ns()
    }

    fn __float__(&self) -> f64 {
        self.0.as_secs_f64()
    }

    fn __int__(&self) -> u64 {
        self.0.whole_secs()
    }

    /// The time from `earlier` to this one.
    ///
    /// Raises:
    ///     ValueError: `earlier` is later than this one.
    fn __sub__(&self, earlier: PyRef<'_, Self>) -> PyResult<Self> {
        self.0.checked_sub(earlier.0).map(Self).ok_or_else(|| {
            PyValueError::new_err(format!(
                "cannot subtract the later {:?} from {:?}",
                earlier.__repr__(),
                self.__repr__()
            ))
        })
    }

    fn __repr__(&self) -> String {
        format!("Timestamp(ns={})", self.0.as_ns())
    }

    /// Pickles and copies as `Timestamp(ns=...)`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        // `copyreg.__newobj_ex__(cls, args, kwargs)` is `cls.__new__(cls, *args,
        // **kwargs)`: the constructor takes `ns` by keyword only.
        let new = py.import("copyreg")?.getattr("__newobj_ex__")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("ns", slf.get().0.as_ns())?;
        let args = (slf.get_type(), PyTuple::empty(py), kwargs);
        Ok((new, args).into_pyobject(py)?.into_any())
    }
}
