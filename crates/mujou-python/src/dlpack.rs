//! The DLPack capsules of the bindings: borrowing a producer's tensor from
//! its `__dlpack__`, and lending a buffer of a message to a consumer. The
//! structs, the copy and the export are in `mujou_core::dlpack`.

use std::ffi::CStr;
use std::sync::Arc;

use mujou_core::buffer::Buffer;
use mujou_core::dlpack::export::export;
use mujou_core::dlpack::ffi::{DLManagedTensorVersioned, DLTensor, VERSION};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{ffi, intern};

const VERSIONED: &CStr = c"dltensor_versioned";

/// A pointer to a producer's tensor, handed to the copy that runs without the GIL.
#[derive(Clone, Copy)]
pub(crate) struct TensorPtr(pub(crate) *const DLTensor);

// SAFETY: the tensor is only read, and only while the capsule that owns it is held
// by the `Borrowed` of the calling frame, which outlives the detached copy. If
// another thread resizes or frees the producer's tensor during the copy, that is
// a data race in the caller's code, as with any read of a tensor another thread
// is changing.
unsafe impl Send for TensorPtr {}
// SAFETY: as above; shared references only read the pointer.
unsafe impl Sync for TensorPtr {}

/// Asks `value` for its DLPack capsule of the 1.x protocol (`torch.Tensor`
/// gives one), or returns `None` if it has no `__dlpack__`.
pub(crate) fn borrow_tensor<'py>(
    value: &Bound<'py, PyAny>,
    at: &str,
) -> PyResult<Option<(Bound<'py, PyAny>, TensorPtr)>> {
    let py = value.py();
    let Some(method) = value.getattr_opt(intern!(py, "__dlpack__"))? else {
        return Ok(None);
    };
    let borrow = || -> PyResult<_> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_version", (VERSION.major, VERSION.minor))?;
        let capsule = method.call((), Some(&kwargs))?;
        let tensor = capsule_tensor(&capsule)?;
        Ok((capsule, TensorPtr(tensor)))
    };
    borrow().map(Some).map_err(|e| {
        // The producer's own errors (such as torch refusing a tensor that requires
        // grad) do not say which value was being written.
        let note = format!("while reading the value of {at} over DLPack");
        match e.value(py).call_method1(intern!(py, "add_note"), (note,)) {
            Ok(_) => e,
            Err(note_error) => note_error,
        }
    })
}

fn capsule_tensor(capsule: &Bound<'_, PyAny>) -> PyResult<*const DLTensor> {
    let invalid = |what: &str| PyTypeError::new_err(format!("__dlpack__ returned {what}"));
    let obj = capsule.as_ptr();
    // SAFETY: `obj` is a live object, held by `capsule`.
    if unsafe { ffi::PyCapsule_CheckExact(obj) } == 0 {
        return Err(invalid("something other than a capsule"));
    }
    // SAFETY: `obj` is a capsule, so its name can be read; the name lives as long
    // as the capsule.
    let name = unsafe { ffi::PyCapsule_GetName(obj) };
    if name.is_null() {
        return Err(
            PyErr::take(capsule.py()).unwrap_or_else(|| invalid("a capsule without a name"))
        );
    }
    // SAFETY: a non-null capsule name is a NUL-terminated string.
    let name = unsafe { CStr::from_ptr(name) };
    if name == VERSIONED {
        // SAFETY: the name matches, so the pointer is the producer's
        // `DLManagedTensorVersioned`, valid until its deleter runs, which the
        // capsule's destructor does only after `capsule` is dropped.
        unsafe {
            let managed = ffi::PyCapsule_GetPointer(obj, VERSIONED.as_ptr())
                .cast::<DLManagedTensorVersioned>();
            if managed.is_null() {
                return Err(PyErr::fetch(capsule.py()));
            }
            if (*managed).version.major != VERSION.major {
                return Err(invalid(&format!(
                    "unsupported DLPack version {:?}",
                    (*managed).version
                )));
            }
            Ok(&raw const (*managed).dl_tensor)
        }
    } else {
        Err(invalid(&format!(
            "a capsule named {name:?}, which is used or not a DLPack tensor"
        )))
    }
}

/// Lends `buffer` as a new DLPack capsule of the 1.x protocol.
pub(crate) fn export_capsule(py: Python<'_>, buffer: Arc<Buffer>) -> PyResult<Bound<'_, PyAny>> {
    let ptr = export(buffer);
    // SAFETY: `ptr` is a fresh DLPack tensor and the name a static string.
    let capsule =
        unsafe { ffi::PyCapsule_New(ptr.cast(), VERSIONED.as_ptr(), Some(destroy_capsule)) };
    if capsule.is_null() {
        // SAFETY: the capsule was not made, so nobody else will delete the
        // tensor.
        unsafe { (*ptr).deleter.expect("set by export")(ptr) };
    }
    // SAFETY: `capsule` is a new reference or null with an exception set.
    unsafe { Bound::from_owned_ptr_or_err(py, capsule) }
}

/// Deletes the tensor of a capsule that no consumer took. A consumer that takes it
/// renames the capsule to `used_dltensor_versioned` and owns the deleter.
unsafe extern "C" fn destroy_capsule(capsule: *mut ffi::PyObject) {
    // SAFETY: called by Python with a live capsule; with the name unchanged the
    // pointer is the tensor `export` made, deleted exactly once here.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, VERSIONED.as_ptr()) == 1 {
            let t = ffi::PyCapsule_GetPointer(capsule, VERSIONED.as_ptr())
                .cast::<DLManagedTensorVersioned>();
            (*t).deleter.expect("set by export")(t);
        }
    }
}
