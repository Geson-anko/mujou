//! The DLPack capsules of the bindings: borrowing a producer's tensor from
//! its `__dlpack__`, and lending a field of a message to a consumer. The
//! structs and the copy are in `mujou_core::dlpack`.

use std::ffi::{CStr, c_void};
use std::sync::Arc;

use mujou_core::dlpack::{self, DLManagedTensor, DLManagedTensorVersioned, DLTensor};
use mujou_core::slot::Message;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{ffi, intern};

const VERSIONED: &CStr = c"dltensor_versioned";
const LEGACY: &CStr = c"dltensor";

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

/// Asks `value` for its DLPack capsule, preferring the 1.x protocol and falling
/// back to the unversioned one for producers that do not take `max_version`.
pub(crate) fn borrow_tensor<'py>(
    value: &Bound<'py, PyAny>,
    at: &str,
) -> PyResult<(Bound<'py, PyAny>, TensorPtr)> {
    let py = value.py();
    let Some(method) = value.getattr_opt(intern!(py, "__dlpack__"))? else {
        return Err(PyTypeError::new_err(format!(
            "{at}: expected a tensor supporting DLPack, such as torch.Tensor, got {}",
            value.get_type().name()?
        )));
    };
    let borrow = || {
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "max_version",
            (dlpack::VERSION.major, dlpack::VERSION.minor),
        )?;
        let capsule = match method.call((), Some(&kwargs)) {
            Ok(capsule) => capsule,
            Err(e) if e.is_instance_of::<PyTypeError>(py) => method.call0()?,
            Err(e) => return Err(e),
        };
        let tensor = capsule_tensor(&capsule)?;
        Ok((capsule, TensorPtr(tensor)))
    };
    borrow().map_err(|e| {
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
            if (*managed).version.major != dlpack::VERSION.major {
                return Err(invalid(&format!(
                    "unsupported DLPack version {:?}",
                    (*managed).version
                )));
            }
            Ok(&raw const (*managed).dl_tensor)
        }
    } else if name == LEGACY {
        // SAFETY: as above, for the unversioned `DLManagedTensor`.
        unsafe {
            let managed = ffi::PyCapsule_GetPointer(obj, LEGACY.as_ptr()).cast::<DLManagedTensor>();
            if managed.is_null() {
                return Err(PyErr::fetch(capsule.py()));
            }
            Ok(&raw const (*managed).dl_tensor)
        }
    } else {
        Err(invalid(&format!(
            "a capsule named {name:?}, which is used or not a DLPack tensor"
        )))
    }
}

/// Lends field `index` of `message` as a new DLPack capsule, of the 1.x
/// protocol if `versioned`, and of the unversioned one otherwise.
pub(crate) fn export_capsule(
    py: Python<'_>,
    message: Arc<Message>,
    index: usize,
    versioned: bool,
) -> PyResult<Bound<'_, PyAny>> {
    let (ptr, name, destructor): (*mut c_void, &CStr, ffi::PyCapsule_Destructor) = if versioned {
        let ptr = dlpack::export_versioned(message, index);
        (ptr.cast(), VERSIONED, destroy_versioned)
    } else {
        let ptr = dlpack::export_legacy(message, index);
        (ptr.cast(), LEGACY, destroy_legacy)
    };
    // SAFETY: `ptr` is a fresh DLPack tensor and `name` a static string.
    let capsule = unsafe { ffi::PyCapsule_New(ptr, name.as_ptr(), Some(destructor)) };
    if capsule.is_null() {
        // SAFETY: the capsule was not made, so nobody else will delete the
        // tensor; `destroy_*` is not usable without a capsule.
        unsafe {
            if name == VERSIONED {
                let t = ptr.cast::<DLManagedTensorVersioned>();
                (*t).deleter.expect("set by export")(t);
            } else {
                let t = ptr.cast::<DLManagedTensor>();
                (*t).deleter.expect("set by export")(t);
            }
        }
    }
    // SAFETY: `capsule` is a new reference or null with an exception set.
    unsafe { Bound::from_owned_ptr_or_err(py, capsule) }
}

/// Deletes the tensor of a capsule that no consumer took. A consumer that takes it
/// renames the capsule to `used_dltensor_versioned` and owns the deleter.
unsafe extern "C" fn destroy_versioned(capsule: *mut ffi::PyObject) {
    // SAFETY: called by Python with a live capsule; with the name unchanged the
    // pointer is the tensor `export_versioned` made, deleted exactly once here.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, VERSIONED.as_ptr()) == 1 {
            let t = ffi::PyCapsule_GetPointer(capsule, VERSIONED.as_ptr())
                .cast::<DLManagedTensorVersioned>();
            (*t).deleter.expect("set by export")(t);
        }
    }
}

unsafe extern "C" fn destroy_legacy(capsule: *mut ffi::PyObject) {
    // SAFETY: as in `destroy_versioned`, for `export_legacy`.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, LEGACY.as_ptr()) == 1 {
            let t = ffi::PyCapsule_GetPointer(capsule, LEGACY.as_ptr()).cast::<DLManagedTensor>();
            (*t).deleter.expect("set by export")(t);
        }
    }
}
