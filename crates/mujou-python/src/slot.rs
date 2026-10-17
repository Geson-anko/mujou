//! Bindings of `mujou_core::slot`.
//!
//! Values come in as any object with `__dlpack__` (in practice `torch.Tensor`) and
//! are copied into pooled buffers; they go out as `BufferView`s, which torch reads
//! with `torch.from_dlpack(view)` and then copies. Copying on both sides keeps every
//! message immutable however the caller later changes its tensors
//! (`specs/messages.md`).

use std::ffi::{CStr, c_void};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::hint::{Values, Views};
use crate::time::{PyTimer, PyTimestamp, timer_or_global};
use mujou_core::dlpack::{self, DLManagedTensor, DLManagedTensorVersioned, DLTensor};
use mujou_core::slot::{self as core, DType, FieldSpec, Message, Schema, SlotError, Target};
use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{ffi, intern};

/// How long `Slot.wait_newer` waits without the GIL before checking for signals.
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(50);

const VERSIONED: &CStr = c"dltensor_versioned";
const LEGACY: &CStr = c"dltensor";

fn to_py_err(err: SlotError) -> PyErr {
    match err {
        SlotError::InvalidSchema(_) | SlotError::Mismatch { .. } => {
            PyValueError::new_err(err.to_string())
        }
        SlotError::UnsupportedDevice { .. } | SlotError::InvalidDLPack { .. } => {
            PyTypeError::new_err(err.to_string())
        }
    }
}

/// A pointer to a producer's tensor, handed to the copy that runs without the GIL.
#[derive(Clone, Copy)]
struct TensorPtr(*const DLTensor);

// SAFETY: the tensor is only read, and only while the capsule that owns it is held
// by the `Borrowed` of the calling frame, which outlives the detached copy. If
// another thread resizes or frees the producer's tensor during the copy, that is
// a data race in the caller's code, as with any read of a tensor another thread
// is changing.
unsafe impl Send for TensorPtr {}
// SAFETY: as above; shared references only read the pointer.
unsafe impl Sync for TensorPtr {}

/// The values of one message, borrowed over DLPack until they are copied. The
/// capsules stay unconsumed and are held here, so the producers' memory lives
/// until the copy is done, and each capsule's destructor calls the deleter when
/// it is dropped.
struct Borrowed<'py> {
    _capsules: Vec<Bound<'py, PyAny>>,
    tensors: Vec<TensorPtr>,
}

/// Asks `value` for its DLPack capsule, preferring the 1.x protocol and falling
/// back to the unversioned one for producers that do not take `max_version`.
fn borrow_tensor<'py>(
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

/// Takes the DLPack capsule of each value, checking the count against `schema`.
/// All Python calls happen here, before any buffer is filled.
fn borrow_values<'py>(
    values: &Bound<'py, PyAny>,
    schema: &Schema,
    slot: &str,
) -> PyResult<Borrowed<'py>> {
    if values.hasattr(intern!(values.py(), "__dlpack__"))? {
        // A tensor is iterable too; iterating it would split it into rows.
        return Err(PyTypeError::new_err(format!(
            "slot '{slot}': expected a sequence of tensors, got a single tensor"
        )));
    }
    let values: Vec<Bound<'py, PyAny>> = values.try_iter()?.collect::<PyResult<_>>()?;
    let fields = schema.fields();
    if values.len() != fields.len() {
        let expected = match fields {
            [field] if field.name.is_empty() => "1 tensor".to_owned(),
            _ => {
                let names: Vec<String> = fields.iter().map(|f| format!("'{}'", f.name)).collect();
                format!("{} tensors (fields {})", fields.len(), names.join(", "))
            }
        };
        return Err(PyValueError::new_err(format!(
            "slot '{slot}': expected {expected}, got {}",
            values.len()
        )));
    }
    let (capsules, tensors) = values
        .iter()
        .zip(fields)
        .map(|(value, field)| borrow_tensor(value, &core::location(slot, &field.name)))
        .collect::<PyResult<Vec<_>>>()?
        .into_iter()
        .unzip();
    Ok(Borrowed {
        _capsules: capsules,
        tensors,
    })
}

/// Copies each borrowed tensor into the buffer of its field. Calls no Python, so
/// it runs without the GIL.
fn copy_into(
    tensors: &[TensorPtr],
    schema: &Schema,
    slot: &str,
    bytes: &mut [&mut [u8]],
) -> Result<(), SlotError> {
    for ((t, field), dst) in tensors.iter().zip(schema.fields()).zip(bytes.iter_mut()) {
        // SAFETY: `t` came from a capsule that the caller's `Borrowed` keeps
        // alive, and the producer guarantees it describes readable memory
        // (checked to be on the host by `copy_from`). `dst` is a pool buffer,
        // distinct from it.
        unsafe { dlpack::copy_from(&*t.0, field, dst) }.map_err(|e| e.at_slot(slot))?;
    }
    Ok(())
}

/// Borrows `values` for `slot` with the GIL, then runs `write` without it, giving
/// it a `fill` that copies the values into the buffers of the new message.
fn write_values<W>(values: &Bound<'_, PyAny>, schema: &Schema, slot: &str, write: W) -> PyResult<()>
where
    W: Send + FnOnce(&dyn Fn(&mut [&mut [u8]]) -> Result<(), SlotError>) -> Result<(), SlotError>,
{
    let borrowed = borrow_values(values, schema, slot)?;
    let tensors = &borrowed.tensors;
    let fill = |bytes: &mut [&mut [u8]]| copy_into(tensors, schema, slot, bytes);
    values.py().detach(|| write(&fill)).map_err(to_py_err)
}

fn check_index(index: isize, len: usize) -> PyResult<usize> {
    usize::try_from(index)
        .ok()
        .filter(|&i| i < len)
        .ok_or_else(|| {
            PyIndexError::new_err(format!(
                "channel index {index} out of range for {len} channels"
            ))
        })
}

/// `(code, bits, lanes)`, as in DLPack.
type DLDataType = (u8, u8, u16);

/// The fields of a message, fixed when the network is built.
///
/// `Schema(fields)` takes `(name, shape, (code, bits, lanes))` per field, the
/// dtype as a DLPack `DLDataType`. A tensor message has one field named `""`;
/// a TypedDict message has one field per key, in key order. Every extent must
/// be at least 1.
///
/// Raises:
///     ValueError: No fields, duplicate names, an extent below 1, a dtype that
///         is not a scalar of whole bytes, or a field too large to allocate.
#[pyclass(frozen, module = "mujou._core", name = "Schema")]
pub struct PySchema(Arc<Schema>);

#[pymethods]
impl PySchema {
    #[new]
    fn new(fields: Vec<(String, Vec<i64>, DLDataType)>) -> PyResult<Self> {
        let fields = fields
            .into_iter()
            .map(|(name, shape, (code, bits, lanes))| FieldSpec {
                name,
                shape,
                dtype: DType { code, bits, lanes },
            })
            .collect();
        Ok(Self(Arc::new(Schema::new(fields).map_err(to_py_err)?)))
    }
}

/// One field of a message, lent over DLPack.
///
/// Read it with `torch.from_dlpack(view)` and copy the result at once: the
/// memory belongs to the core and must not be written. `__dlpack__` raises
/// `BufferError` for `copy=True` or a `dl_device` other than the CPU.
#[pyclass(frozen, module = "mujou._core")]
pub struct BufferView {
    message: Arc<Message>,
    index: usize,
}

#[pymethods]
impl BufferView {
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // The data is on the CPU, where there is no stream to synchronize with.
        let _ = stream;
        if dl_device.is_some_and(|d| d != (dlpack::DEVICE_CPU, 0)) {
            return Err(PyBufferError::new_err(
                "a message buffer can only be read on the CPU",
            ));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "a message buffer cannot be copied by its producer; copy the tensor after from_dlpack",
            ));
        }
        let message = Arc::clone(&self.message);
        let (ptr, name, destructor): (*mut c_void, &CStr, ffi::PyCapsule_Destructor) =
            if max_version.is_some_and(|(major, _)| major >= dlpack::VERSION.major) {
                let ptr = dlpack::export_versioned(message, self.index);
                (ptr.cast(), VERSIONED, destroy_versioned)
            } else {
                let ptr = dlpack::export_legacy(message, self.index);
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

    fn __dlpack_device__(&self) -> (i32, i32) {
        (dlpack::DEVICE_CPU, 0)
    }
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

/// An immutable value of a slot, and the time it was written at on the
/// writer's timer.
#[pyclass(frozen, module = "mujou._core", name = "Message")]
pub struct PyMessage(Arc<Message>);

#[pymethods]
impl PyMessage {
    /// One view per field, in key order.
    #[getter]
    fn buffers<'py>(&self, py: Python<'py>) -> PyResult<Views<'py>> {
        let views = (0..self.0.buffers().len()).map(|index| BufferView {
            message: Arc::clone(&self.0),
            index,
        });
        PyTuple::new(py, views).map(Views)
    }

    /// The time it was written at, in nanoseconds.
    #[getter]
    fn time_ns(&self) -> u64 {
        self.0.time_ns()
    }

    /// The time it was written at.
    #[getter]
    fn timestamp(&self) -> PyTimestamp {
        PyTimestamp(self.0.timestamp())
    }
}

/// The latest value of one writer, read by any number of readers.
#[pyclass(frozen, module = "mujou._core", name = "Slot")]
pub struct PySlot(Arc<core::Slot>);

#[pymethods]
impl PySlot {
    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    fn read(&self) -> PyMessage {
        PyMessage(self.0.load())
    }

    /// Waits without the GIL until the slot holds a message other than `last`.
    ///
    /// Returns `None` after `timeout` seconds of real time. With `last` `None`,
    /// returns the current value at once.
    ///
    /// Only for the host's `subscribe`: a node must never wait for another, so
    /// ports do not expose it.
    ///
    /// Raises:
    ///     ValueError: `timeout` is negative or not finite.
    #[pyo3(signature = (last, timeout))]
    fn wait_newer(
        &self,
        py: Python<'_>,
        last: Option<PyRef<'_, PyMessage>>,
        timeout: f64,
    ) -> PyResult<Option<PyMessage>> {
        let timeout = Duration::try_from_secs_f64(timeout).map_err(|_| {
            PyValueError::new_err(format!(
                "timeout must be finite and >= 0 seconds, got {timeout}"
            ))
        })?;
        let last = last.map(|m| Arc::clone(&m.0));
        // A timeout too long to represent never expires.
        let deadline = Instant::now().checked_add(timeout);
        loop {
            let remaining = deadline.map_or(Duration::MAX, |d| {
                d.saturating_duration_since(Instant::now())
            });
            let step = remaining.min(SIGNAL_CHECK_INTERVAL);
            if let Some(msg) = py.detach(|| self.0.wait_newer(last.as_ref(), step)) {
                return Ok(Some(PyMessage(msg)));
            }
            if remaining <= SIGNAL_CHECK_INTERVAL {
                return Ok(None);
            }
            py.check_signals()?;
        }
    }

    /// Sets the value and its time as they were when the state was taken.
    ///
    /// Call it only while no node runs (before the start, or while every node
    /// is paused): it writes beside the slot's own writer. Raises as
    /// `Outlet.send` does.
    fn restore(&self, values: Values<'_>, time_ns: u64) -> PyResult<()> {
        let slot = &self.0;
        write_values(&values.0, slot.schema(), slot.name(), |fill| {
            slot.restore(time_ns, fill)
        })
    }
}

/// The sending side of one port and the only writer of its slots.
///
/// The caller keeps it the only writer: send from one thread at a time (the
/// owning node's). Concurrent sends stay memory safe but may leave the slots
/// out of time order, or the outward slot and the channels holding different
/// messages.
///
/// Values are copied in without the GIL, so changing them afterwards does not
/// change what readers see. Resizing or freeing a value from another thread
/// while it is being sent is a data race in the caller's code.
///
/// Raises (`send`, `send_to`):
///     ValueError: The number of values, a shape or a dtype differs from the
///         schema. The message names the slot and the field.
///     TypeError: A value does not support DLPack or is not in host memory.
///         Pinned memory is read without synchronizing any CUDA stream, so
///         synchronize the copy that filled it before sending.
///     IndexError: No such channel.
#[pyclass(frozen, module = "mujou._core", name = "Outlet")]
pub struct PyOutlet(core::Outlet);

impl PyOutlet {
    fn send_with(&self, target: Target, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let outlet = &self.0;
        let name = match target {
            Target::All => outlet.outbox().name(),
            Target::Channel(k) => outlet.channel(k).name(),
        };
        write_values(values, outlet.schema(), name, |fill| {
            outlet.send_with(target, fill)
        })
    }
}

#[pymethods]
impl PyOutlet {
    /// Creates the slots `"{name}/out"` and `"{name}/ch{k}"` holding zeros.
    /// Every message is stamped with the time on `timer` (the global timer if
    /// `None`).
    ///
    /// The slots and the host's inputs of one run must use the timer of its
    /// runtime: on another timer, the times of the messages would not be on
    /// the axis the nodes' Δt is measured on, and a pause of the run would not
    /// stop them, so a state taken at T could hold later messages.
    #[new]
    #[pyo3(signature = (schema, channels, name, *, timer = None))]
    fn new(
        schema: PyRef<'_, PySchema>,
        channels: usize,
        name: &str,
        timer: Option<PyRef<'_, PyTimer>>,
    ) -> Self {
        Self(core::Outlet::new(
            Arc::clone(&schema.0),
            channels,
            name,
            timer_or_global(timer),
        ))
    }

    /// The outward slot, which every `send` writes.
    #[getter]
    fn outbox(&self) -> PySlot {
        PySlot(Arc::clone(self.0.outbox()))
    }

    fn channel(&self, index: isize) -> PyResult<PySlot> {
        let k = check_index(index, self.0.channels().len())?;
        Ok(PySlot(Arc::clone(self.0.channel(k))))
    }

    /// Writes one message to the outward slot and every channel.
    fn send(&self, values: Values<'_>) -> PyResult<()> {
        self.send_with(Target::All, &values.0)
    }

    /// Writes one message to channel `index` only.
    fn send_to(&self, index: isize, values: Values<'_>) -> PyResult<()> {
        let k = check_index(index, self.0.channels().len())?;
        self.send_with(Target::Channel(k), &values.0)
    }
}

/// The slots the host writes, switched together by one version counter.
#[pyclass(frozen, module = "mujou._core", name = "InputGroup")]
pub struct PyInputGroup(Arc<core::InputGroup>);

#[pymethods]
impl PyInputGroup {
    /// Every write is stamped with the time on `timer` (the global timer if
    /// `None`).
    ///
    /// The slots and the host's inputs of one run must use the timer of its
    /// runtime: on another timer, the times of the messages would not be on
    /// the axis the nodes' Δt is measured on, and a pause of the run would not
    /// stop them, so a state taken at T could hold later messages.
    #[new]
    #[pyo3(signature = (*, timer = None))]
    fn new(timer: Option<PyRef<'_, PyTimer>>) -> Self {
        Self(core::InputGroup::new(timer_or_global(timer)))
    }

    /// Creates the slot `"{name}/in"` holding zeros.
    fn input(&self, schema: PyRef<'_, PySchema>, name: &str) -> PyInputSlot {
        PyInputSlot(self.0.input(Arc::clone(&schema.0), name))
    }

    /// Writes every item at once, all stamped with the same time.
    ///
    /// No reader sees some of the slots new and others old at any one moment;
    /// a reader that reads two slots one after the other may still see a write
    /// complete in between. If any value is invalid, no slot changes. Raises as
    /// `Outlet.send` does, and `ValueError` for a slot of another group or a
    /// slot given twice.
    fn write(
        &self,
        py: Python<'_>,
        items: Vec<(PyRef<'_, PyInputSlot>, Values<'_>)>,
    ) -> PyResult<()> {
        let mut slots: Vec<&core::InputSlot> = Vec::with_capacity(items.len());
        let mut borrowed = Vec::with_capacity(items.len());
        for (slot, values) in &items {
            let slot = &*slot.0;
            if !slot.belongs_to(&self.0) {
                return Err(PyValueError::new_err(format!(
                    "slot '{}' belongs to another input group",
                    slot.name()
                )));
            }
            if slots.iter().any(|&s| std::ptr::eq(s, slot)) {
                return Err(PyValueError::new_err(format!(
                    "slot '{}' appears more than once in one write",
                    slot.name()
                )));
            }
            borrowed.push(borrow_values(&values.0, slot.schema(), slot.name())?);
            slots.push(slot);
        }
        let tensors: Vec<&[TensorPtr]> = borrowed.iter().map(|b| b.tensors.as_slice()).collect();
        py.detach(|| {
            self.0.write(&slots, |i, bytes| {
                copy_into(tensors[i], slots[i].schema(), slots[i].name(), bytes)
            })
        })
        .map_err(to_py_err)
    }
}

/// A slot the host writes through its `InputGroup`.
#[pyclass(frozen, module = "mujou._core", name = "InputSlot")]
pub struct PyInputSlot(Arc<core::InputSlot>);

#[pymethods]
impl PyInputSlot {
    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    fn read(&self) -> PyMessage {
        PyMessage(self.0.load())
    }

    /// Sets the value and its time, dropping any pending write.
    ///
    /// Call it only while no node runs (before the start, or while every node
    /// is paused). Raises as `Outlet.send` does.
    fn restore(&self, values: Values<'_>, time_ns: u64) -> PyResult<()> {
        let slot = &self.0;
        write_values(&values.0, slot.schema(), slot.name(), |fill| {
            slot.restore(time_ns, fill)
        })
    }
}
