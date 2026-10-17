//! Python bindings of the native core, imported as the private module `mujou._core._native`.
//!
//! The module is declared inline (`#[pymodule] mod`) so that pyo3 can introspect
//! it: `just stub` writes `src/mujou/_core/_native.pyi` from the signatures and doc
//! comments here. Its full name is set once, in the `#[pymodule]` attribute
//! below; the classes repeat it in their `module = "..."` (pyo3 takes only a
//! literal there), and `pyproject.toml` (`module-name`) must match.

mod dlpack;
mod hint;
mod runtime;
mod slot;
mod time;

use std::time::{Duration, Instant};

use pyo3::prelude::*;

/// How long a wait without the GIL lasts before checking for signals (Ctrl-C).
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);

/// Waits without the GIL until `attempt` returns `Some` or `deadline` passes
/// (never if `None`), checking for signals in between: `attempt` is given
/// how long it may block, at most `SIGNAL_CHECK_INTERVAL`, and returns `None`
/// if what it waits for did not come in that time.
fn wait_interruptibly<T: Send>(
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

/// The native core of mujou: timers, slots and the runtime that runs the nodes.
///
/// Private: the framework wraps it. Values go in and out as tensors over DLPack,
/// always copied.
#[pymodule(module = "mujou._core", name = "_native", gil_used = false)]
// The constants take their Python names, `__version__` and `timer`.
#[allow(non_upper_case_globals)]
mod core_module {
    #[pymodule_export]
    const __version__: &str = env!("CARGO_PKG_VERSION");

    /// The global timer.
    #[pymodule_export]
    const timer: super::time::GlobalTimer = super::time::GlobalTimer;

    #[pymodule_export]
    use super::time::{GlobalTimer, PyTimer, PyTimerReader, PyTimestamp};

    #[pymodule_export]
    use super::slot::{
        BufferView, PyInputGroup, PyInputSlot, PyMessage, PyOutlet, PySchema, PySlot,
    };

    #[pymodule_export]
    use super::runtime::{NetworkError, Node, PauseTimeoutError, Runtime};
}
