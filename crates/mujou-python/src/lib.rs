//! Python bindings of the native core, imported as the private module `mujou._core`.

use pyo3::prelude::*;

#[pymodule(gil_used = false)]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
