//! Bindings of `mujou_core::time`: timers and timestamps, which `mujou.time`
//! shows as they are.

use std::sync::Arc;

use mujou_core::time::{self as core, RunClock, Source, Timer, TimerError, Timestamp};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::inspect::PyStaticExpr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{PyClassInitializer, PyTypeInfo};

use crate::current::in_hook;
use crate::wait::{to_ns, wait_interruptibly};

/// A time axis that a run's nodes measure their `dt` on.
///
/// `Timer()` starts at `start` and runs with the real clock, `time_scale`
/// seconds of the timer per real second. Give it to
/// `mujou.run(..., timer=...)` to run a network on a time of its own: a pause
/// of that run (as `Run.state_dict` takes) then stops only this timer, not
/// `global_timer`, which every other run uses by default. Every time of the
/// run is a `Timestamp` on it.
///
/// Only the host controls a timer: `sleep`, `pause`, `resume`,
/// `set_time_scale` (and `VirtualTimer.advance`) raise `RuntimeError` in a
/// node's hook, where a node would wait for, or change the `dt` of, the other
/// nodes. A node reads the time of its run with `self.timer`.
///
/// `pause` stops the time of every run on the timer at one time T until
/// `resume`. Take the states of several runs on one timer while it is paused,
/// so that all of them are of the same time T.
///
/// The time is held in nanoseconds and saturates at about 584 years.
///
/// Raises:
///     ValueError: `time_scale` is not finite and > 0.
#[pyclass(subclass, frozen, module = "mujou.time", name = "Timer")]
pub struct PyTimer(pub(crate) Arc<Timer>);

/// Raises `RuntimeError` in a node's hook, where `what` must not run.
fn host_only(what: &str) -> PyResult<()> {
    if in_hook() {
        return Err(PyRuntimeError::new_err(format!(
            "a node's hook cannot call {what}: only the host controls the time; a \
             node reads it with self.timer.now()"
        )));
    }
    Ok(())
}

#[pymethods]
impl PyTimer {
    #[new]
    #[pyo3(signature = (*, start = PyTimestamp(Timestamp::from_ns(0)), time_scale = 1.0))]
    fn new(start: PyTimestamp, time_scale: f64) -> PyResult<Self> {
        let timer = Timer::new(Source::Real, start.0);
        timer.set_time_scale(time_scale).map_err(timer_error)?;
        Ok(Self(Arc::new(timer)))
    }

    /// The time now.
    fn now(&self) -> PyTimestamp {
        PyTimestamp(self.0.now())
    }

    /// Blocks for `seconds` of this timer's time, following the scale and the
    /// pauses: while paused, it does not return until resumed. Waits without
    /// the GIL and checks for signals (Ctrl-C) in between.
    ///
    /// Raises:
    ///     ValueError: `seconds` is negative or not finite.
    ///     RuntimeError: This is called from a node's hook.
    fn sleep(&self, py: Python<'_>, seconds: f64) -> PyResult<()> {
        host_only("Timer.sleep")?;
        let target_ns = self.0.time_ns().saturating_add(to_ns(seconds)?);
        wait_interruptibly(py, None, |step| {
            self.0.wait_until_ns(target_ns, step).then_some(())
        })?;
        Ok(())
    }

    /// How many seconds of this timer pass per real second.
    fn time_scale(&self) -> f64 {
        self.0.time_scale()
    }

    /// Sets how many seconds of this timer pass per real second.
    ///
    /// Raises:
    ///     ValueError: `scale` is not finite and > 0. Use `pause` to stop the
    ///         time.
    ///     RuntimeError: This is called from a node's hook.
    fn set_time_scale(&self, scale: f64) -> PyResult<()> {
        host_only("Timer.set_time_scale")?;
        self.0.set_time_scale(scale).map_err(timer_error)
    }

    /// Stops the time until `resume`.
    ///
    /// While paused, the nodes of the runs on this timer do not update (the
    /// first `dt` after `resume` does not count the pause), and `sleep` does
    /// not return until resumed. `Run.state_dict` still works. This is a
    /// switch, not a count: pausing twice needs one `resume`. A run that takes
    /// its state holds the time paused on its own, apart from this switch, so
    /// `resume` does not end that, and its end does not end this pause.
    ///
    /// Raises:
    ///     RuntimeError: This is called from a node's hook.
    fn pause(&self) -> PyResult<()> {
        host_only("Timer.pause")?;
        self.0.pause();
        Ok(())
    }

    /// Turns off the switch of `pause`: the time moves again from where it
    /// stopped, unless a run on this timer is taking its state.
    ///
    /// Raises:
    ///     RuntimeError: This is called from a node's hook.
    fn resume(&self) -> PyResult<()> {
        host_only("Timer.resume")?;
        self.0.resume();
        Ok(())
    }

    /// Whether the time is stopped, by `pause` or by a run on this timer that
    /// is starting or taking its state. The clock of a `VirtualTimer` standing
    /// still does not count as paused.
    fn is_paused(&self) -> bool {
        self.0.is_paused()
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().qualname()?;
        Ok(format!(
            "{name}(now={})",
            PyTimestamp(slf.get().0.now()).__repr__()
        ))
    }
}

fn timer_error(err: TimerError) -> PyErr {
    match err {
        TimerError::InvalidScale(_) | TimerError::InvalidDuration(_) => {
            PyValueError::new_err(err.to_string())
        }
        TimerError::Paused => PyRuntimeError::new_err(err.to_string()),
    }
}

/// A timer for tests: its time moves only when `advance` moves it.
///
/// It starts at `start`. Its time scale does not apply to `advance`.
/// `mujou.testing.run` runs a network on a new one unless given one, and
/// `ManualRun.step(dt)` advances it.
#[pyclass(extends = PyTimer, frozen, module = "mujou.time", name = "VirtualTimer")]
pub struct PyVirtualTimer;

#[pymethods]
impl PyVirtualTimer {
    #[new]
    #[pyo3(signature = (*, start = PyTimestamp(Timestamp::from_ns(0))))]
    fn new(start: PyTimestamp) -> PyClassInitializer<Self> {
        let timer = PyTimer(Arc::new(Timer::new(Source::Virtual, start.0)));
        PyClassInitializer::from(timer).add_subclass(Self)
    }

    /// Moves the time forward by `seconds`, at once.
    ///
    /// Raises:
    ///     ValueError: `seconds` is negative or not finite.
    ///     RuntimeError: The timer is paused (by `pause`, or by a run on it
    ///         that is starting or taking its state), or this is called from a
    ///         node's hook.
    fn advance(slf: PyRef<'_, Self>, seconds: f64) -> PyResult<()> {
        host_only("VirtualTimer.advance")?;
        let dt_ns = to_ns(seconds)?;
        slf.as_super().0.advance_ns(dt_ns).map_err(timer_error)
    }
}

/// The global timer, a `Timer` at 0 when the process first uses it: every
/// run not given a timer of its own runs on it.
pub struct GlobalTimer;

// The module constant `global_timer`: a `Timer` built once, when the module is
// made.
impl<'py> IntoPyObject<'py> for GlobalTimer {
    type Target = PyTimer;
    type Output = Bound<'py, PyTimer>;
    type Error = PyErr;
    const OUTPUT_TYPE: PyStaticExpr = <PyTimer as PyTypeInfo>::TYPE_HINT;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, PyTimer(Arc::clone(core::global())))
    }
}

/// The reading side of a timer: its time and its scale, and nothing that
/// changes them. What a node sees of its run's timer as `Node.timer`.
///
/// `ReadOnlyTimer(timer)` reads `timer`. The one a run gives its nodes stops
/// at the time the run ends: every time of the run is at most that time.
#[pyclass(frozen, module = "mujou.time", name = "ReadOnlyTimer")]
pub struct PyReadOnlyTimer(pub(crate) Arc<RunClock>);

#[pymethods]
impl PyReadOnlyTimer {
    #[new]
    fn new(timer: PyRef<'_, PyTimer>) -> Self {
        Self(Arc::new(RunClock::new(Arc::clone(&timer.0))))
    }

    /// The time now.
    fn now(&self) -> PyTimestamp {
        PyTimestamp(self.0.now())
    }

    /// How many seconds of the timer pass per real second.
    fn time_scale(&self) -> f64 {
        self.0.time_scale()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReadOnlyTimer(now={})",
            PyTimestamp(self.0.now()).__repr__()
        )
    }
}

/// A time on a timer.
///
/// `float(ts)` is the time in seconds and `int(ts)` its whole seconds; `ts.ns`
/// is the exact time in nanoseconds. Timestamps compare and hash by time, so
/// compare only those of one timer: a timestamp does not say which timer it is
/// on. `a - b` is the time from `b` to `a` in seconds, a `float` that is
/// negative when `b` is later. Timestamps pickle and copy.
#[pyclass(
    frozen,
    eq,
    ord,
    hash,
    from_py_object,
    module = "mujou.time",
    name = "Timestamp"
)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyTimestamp(pub Timestamp);

#[pymethods]
impl PyTimestamp {
    /// Takes the time in nanoseconds, by keyword only: a bare number could
    /// mean seconds.
    #[new]
    #[pyo3(signature = (*, ns))]
    fn new(ns: u64) -> Self {
        Self(Timestamp::from_ns(ns))
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

    /// The time from `other` to this one, in seconds: `(self.ns - other.ns) /
    /// 1e9`, negative when `other` is later.
    fn __sub__(&self, other: PyRef<'_, Self>) -> f64 {
        self.0.secs_since(other.0)
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
