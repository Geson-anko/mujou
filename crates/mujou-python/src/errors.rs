//! The exceptions of mujou that the bindings raise, defined in Python
//! (`mujou.errors`), where they are documented once.

pyo3::import_exception!(mujou.errors, NetworkError);
