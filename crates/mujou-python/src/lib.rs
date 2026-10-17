//! Python bindings of the native core, imported as the private module `mujou._core`.
//!
//! The module is declared inline (`#[pymodule] mod`) so that pyo3 can introspect
//! it: `just stub` writes `src/mujou/_core.pyi` from the signatures and doc
//! comments here. Its full name is set once, in the `#[pymodule]` attribute
//! below; the classes repeat it in their `module = "..."` (pyo3 takes only a
//! literal there), and `pyproject.toml` (`module-name`) must match.

mod hint;
mod runtime;
mod slot;
mod time;

use std::time::Duration;

use mujou_core::time::TimerError;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// How long a wait without the GIL lasts before checking for signals (Ctrl-C).
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);

fn timer_error(err: TimerError) -> PyErr {
    match err {
        TimerError::InvalidScale(_) | TimerError::InvalidDuration(_) => {
            PyValueError::new_err(err.to_string())
        }
        TimerError::NotVirtual | TimerError::Paused => PyRuntimeError::new_err(err.to_string()),
    }
}

/// The native core of mujou: timers, slots and the runtime that runs the nodes.
///
/// Private: the framework wraps it. Values go in and out as tensors over DLPack,
/// always copied.
#[pymodule(module = "mujou", name = "_core", gil_used = false)]
// The constants take their Python names, `__version__` and `timer`.
#[allow(non_upper_case_globals)]
mod core_module {
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    /// The global timer.
    #[pymodule_export]
    const timer: super::time::GlobalTimer = super::time::GlobalTimer;

    #[pymodule_export]
    use super::time::{GlobalTimer, PyTimer, PyTimestamp};

    #[pymodule_export]
    use super::slot::{
        BufferView, PyInputGroup, PyInputSlot, PyMessage, PyOutlet, PySchema, PySlot,
    };

    #[pymodule_export]
    use super::runtime::{NetworkError, Node, PauseTimeoutError, Runtime};
}
