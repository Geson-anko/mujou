//! Which node's hook the calling thread runs, as the node's driver: the one
//! fact that ports, the node's timer and the host-only operations look at.

use std::cell::RefCell;

use pyo3::prelude::*;

thread_local! {
    /// What `current` returns while this thread runs a hook, else `None`.
    static CURRENT: RefCell<Option<Py<PyAny>>> = const { RefCell::new(None) };
}

/// Runs `f` as a hook of `running`, which `current` returns meanwhile.
pub(crate) fn running<T>(running: &Py<PyAny>, py: Python<'_>, f: impl FnOnce() -> T) -> T {
    /// Puts the outer value back however `f` ends, a panic included. Dropped
    /// with the thread attached (`py`), which Python's refcount needs.
    struct Outer(Option<Py<PyAny>>);

    impl Drop for Outer {
        fn drop(&mut self) {
            drop(CURRENT.replace(self.0.take()));
        }
    }

    let _outer = Outer(CURRENT.replace(Some(running.clone_ref(py))));
    f()
}

/// Whether the calling thread runs a hook.
pub(crate) fn in_hook() -> bool {
    CURRENT.with_borrow(Option::is_some)
}

/// The driver (`mujou._core.driver.Driver`) of the node whose hook the
/// calling thread runs; `None` outside a hook.
#[pyfunction]
#[pyo3(signature = () -> "object | None")]
pub fn current(py: Python<'_>) -> Option<Py<PyAny>> {
    CURRENT.with_borrow(|current| current.as_ref().map(|c| c.clone_ref(py)))
}
