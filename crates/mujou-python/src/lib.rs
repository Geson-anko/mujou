//! Python bindings of the native core, imported as the private module `mujou._native`.
//!
//! The module is declared inline (`#[pymodule] mod`) so that pyo3 can introspect
//! it: `just stub` writes `src/mujou/_native.pyi` from the signatures and doc
//! comments here. Its full name is set once, in the `#[pymodule]` attribute
//! below; the classes repeat it in their `module = "..."` (pyo3 takes only a
//! literal there), and `pyproject.toml` (`module-name`) must match. The timers
//! name `mujou.time` instead, where they are public.

mod current;
mod dlpack;
mod errors;
mod hint;
mod runtime;
mod scalar;
mod slot;
mod time;
mod wait;

use pyo3::prelude::*;

/// The native core of mujou: timers, slots and the runtime that runs the nodes.
///
/// Private: the framework wraps it. Values go in and out as tensors over DLPack,
/// always copied.
#[pymodule(module = "mujou", name = "_native", gil_used = false)]
// The constant takes its Python name, `global_timer`.
#[allow(non_upper_case_globals)]
mod core_module {
    /// The global timer, which every run not given a timer of its own uses.
    #[pymodule_export]
    const global_timer: super::time::GlobalTimer = super::time::GlobalTimer;

    #[pymodule_export]
    use super::time::{PyReadOnlyTimer, PyTimer, PyTimestamp, PyVirtualTimer};

    #[pymodule_export]
    use super::slot::{
        BufferView, PyInputGroup, PyInputSlot, PyMessage, PyOutlet, PySchema, PySlot,
    };

    #[pymodule_export]
    use super::current::current;

    #[pymodule_export]
    use super::runtime::{Runtime, start};
}
