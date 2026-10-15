//! Python bindings of the native core, imported as the private module `mujou._core`.

use std::time::Duration;

use mujou_core::time::{self, TimerError, TimerState, seconds_to_ns};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// How long `sleep` waits without the GIL before checking for signals (Ctrl-C).
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);

fn to_py_err(err: TimerError) -> PyErr {
    match err {
        TimerError::InvalidScale(_) | TimerError::InvalidDuration(_) => {
            PyValueError::new_err(err.to_string())
        }
        TimerError::NotVirtual | TimerError::Paused => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Handle to the process-wide global timer. The only instance is `mujou._core.timer`.
#[pyclass(frozen, module = "mujou._core")]
struct GlobalTimer;

#[pymethods]
impl GlobalTimer {
    /// The global time in seconds.
    fn time(&self) -> f64 {
        time::global().time()
    }

    /// The global time in nanoseconds.
    fn time_ns(&self) -> u64 {
        time::global().time_ns()
    }

    /// Blocks for `seconds` of global time, following the scale and pauses.
    fn sleep(&self, py: Python<'_>, seconds: f64) -> PyResult<()> {
        let timer = time::global();
        let dt_ns = seconds_to_ns(seconds).map_err(to_py_err)?;
        let target_ns = timer.time_ns().saturating_add(dt_ns);
        while !py.detach(|| timer.wait_until_ns(target_ns, SIGNAL_CHECK_INTERVAL)) {
            py.check_signals()?;
        }
        Ok(())
    }

    /// Global seconds per real second.
    fn time_scale(&self) -> f64 {
        time::global().scale()
    }

    /// Sets how many global seconds pass per real second (finite and > 0).
    fn set_time_scale(&self, scale: f64) -> PyResult<()> {
        time::global().set_scale(scale).map_err(to_py_err)
    }

    /// Freezes the global time. For the framework's pause and state capture.
    fn pause(&self) {
        time::global().pause();
    }

    /// Lets the global time run again from where it was frozen.
    fn resume(&self) {
        time::global().resume();
    }

    fn is_paused(&self) -> bool {
        time::global().is_paused()
    }

    /// `{"time_ns": int, "scale": float}`.
    fn state_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let state = time::global().state();
        let dict = PyDict::new(py);
        dict.set_item("time_ns", state.time_ns)?;
        dict.set_item("scale", state.scale)?;
        Ok(dict)
    }

    fn load_state_dict(&self, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let state = TimerState {
            time_ns: state.get_item("time_ns")?.extract()?,
            scale: state.get_item("scale")?.extract()?,
        };
        time::global().load_state(state).map_err(to_py_err)
    }

    /// Switches to the virtual clock, which moves only by `advance`. For tests.
    fn use_virtual_clock(&self, enabled: bool) {
        time::global().use_virtual_clock(enabled);
    }

    /// Moves the virtual clock forward by `seconds`.
    fn advance(&self, seconds: f64) -> PyResult<()> {
        let dt_ns = seconds_to_ns(seconds).map_err(to_py_err)?;
        time::global().advance_ns(dt_ns).map_err(to_py_err)
    }
}

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<GlobalTimer>()?;
    m.add("timer", GlobalTimer)?;
    Ok(())
}
