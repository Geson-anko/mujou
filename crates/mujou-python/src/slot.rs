//! Bindings of `mujou_core::slot`.
//!
//! Values come in as any object with `__dlpack__` (in practice `torch.Tensor`) and
//! are copied into pooled buffers; they go out as `BufferView`s, which torch reads
//! with `torch.from_dlpack(view)` and then copies. Copying on both sides keeps every
//! message immutable however the caller later changes its tensors
//! (`specs/messages.md`).

use std::sync::Arc;
use std::time::Instant;

use crate::dlpack::{TensorPtr, borrow_tensor, export_capsule};
use crate::hint::Values;
use crate::time::{PyReadOnlyTimer, PyTimestamp};
use crate::wait::{to_duration, wait_interruptibly};
use mujou_core::buffer::{Buffer, Message};
use mujou_core::dlpack::ffi::{DEVICE_CPU, DLDataType, VERSION};
use mujou_core::dlpack::import::copy_from;
use mujou_core::schema::{FieldError, FieldSpec, Schema, SchemaError};
use mujou_core::slot::{self as core, SlotError, Target};
use mujou_core::time::Timestamp;
use pyo3::exceptions::{PyBufferError, PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;

fn to_py_err(err: SlotError) -> PyErr {
    match err.error {
        FieldError::Mismatch { .. } => PyValueError::new_err(err.to_string()),
        _ => PyTypeError::new_err(err.to_string()),
    }
}

/// The values of one message, borrowed over DLPack until they are copied. The
/// capsules stay unconsumed and are held here, so the producers' memory lives
/// until the copy is done, and each capsule's destructor calls the deleter when
/// it is dropped.
struct Borrowed<'py> {
    _capsules: Vec<Bound<'py, PyAny>>,
    tensors: Vec<TensorPtr>,
}

/// Takes the DLPack capsule of each value, checking the count against `fields`
/// (the fields the write fills). All Python calls happen here, before any buffer
/// is filled.
fn borrow_values<'py>(
    values: &Bound<'py, PyAny>,
    fields: &[FieldSpec],
    slot: &str,
) -> PyResult<Borrowed<'py>> {
    let values: Vec<Bound<'py, PyAny>> = values.try_iter()?.collect::<PyResult<_>>()?;
    // An internal error: the codec gives one value per field, and the copy
    // fills each field from its own value.
    if values.len() != fields.len() {
        return Err(PyValueError::new_err(format!(
            "slot '{slot}': {} values for {} fields",
            values.len(),
            fields.len()
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
    fields: &[FieldSpec],
    slot: &str,
    bytes: &mut [&mut [u8]],
) -> Result<(), SlotError> {
    for ((t, field), dst) in tensors.iter().zip(fields).zip(bytes.iter_mut()) {
        // SAFETY: `t` came from a capsule that the caller's `Borrowed` keeps
        // alive, and the producer guarantees it describes readable memory
        // (checked to be on the host by `copy_from`). `dst` is a pool buffer,
        // distinct from it.
        unsafe { copy_from(&*t.0, field, dst) }.map_err(|error| SlotError {
            slot: slot.to_owned(),
            field: field.name.clone(),
            error,
        })?;
    }
    Ok(())
}

/// Borrows `values` of `fields` for `slot` with the GIL, then runs `write`
/// without it, giving it a `fill` that copies the values into the buffers of
/// those fields.
fn write_values<T, W>(
    values: &Bound<'_, PyAny>,
    fields: &[FieldSpec],
    slot: &str,
    write: W,
) -> PyResult<T>
where
    T: Send,
    W: Send + FnOnce(&dyn Fn(&mut [&mut [u8]]) -> Result<(), SlotError>) -> Result<T, SlotError>,
{
    let borrowed = borrow_values(values, fields, slot)?;
    let tensors = &borrowed.tensors;
    let fill = |bytes: &mut [&mut [u8]]| copy_into(tensors, fields, slot, bytes);
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

/// A [`DLDataType`] as the tuple `(code, bits, lanes)` that Python passes.
type DTypeTuple = (u8, u8, u16);

/// A field of a [`PySchema`] as Python passes it: `(name, shape, dtype)`.
type FieldTuple = (String, Vec<i64>, DTypeTuple);

fn field_specs(fields: Vec<FieldTuple>) -> Vec<FieldSpec> {
    fields
        .into_iter()
        .map(|(name, shape, (code, bits, lanes))| FieldSpec {
            name,
            shape,
            dtype: DLDataType { code, bits, lanes },
        })
        .collect()
}

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
#[pyclass(frozen, module = "mujou._core._native", name = "Schema")]
pub struct PySchema(Arc<Schema>);

#[pymethods]
impl PySchema {
    #[new]
    fn new(fields: Vec<FieldTuple>) -> PyResult<Self> {
        let schema = Schema::new(field_specs(fields))
            .map_err(|e: SchemaError| PyValueError::new_err(e.to_string()))?;
        Ok(Self(Arc::new(schema)))
    }
}

/// One field of a message, lent over DLPack.
///
/// Read it with `torch.from_dlpack(view)` and copy the result at once: the
/// memory belongs to the core and must not be written. `__dlpack__` raises
/// `BufferError` without a `max_version` of 1 or later, for `copy=True` or for
/// a `dl_device` other than the CPU.
#[pyclass(frozen, module = "mujou._core._native")]
pub struct BufferView(Arc<Buffer>);

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
        if dl_device.is_some_and(|d| d != (DEVICE_CPU, 0)) {
            return Err(PyBufferError::new_err(
                "a message buffer can only be read on the CPU",
            ));
        }
        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "a message buffer cannot be copied by its producer; copy the tensor after from_dlpack",
            ));
        }
        if max_version.is_none_or(|(major, _)| major < VERSION.major) {
            return Err(PyBufferError::new_err(
                "a message buffer is lent over DLPack 1.x only; pass max_version=(1, 0) \
                 (torch.from_dlpack does)",
            ));
        }
        export_capsule(py, Arc::clone(&self.0))
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (DEVICE_CPU, 0)
    }
}

/// An immutable value of a slot, and the time its writer wrote it at on the
/// run's clock.
#[pyclass(frozen, module = "mujou._core._native", name = "Message")]
pub struct PyMessage(Arc<Message>);

#[pymethods]
impl PyMessage {
    /// One view per field, in key order.
    #[getter]
    fn values(&self) -> Vec<BufferView> {
        let values = self.0.values().iter();
        values.map(|b| BufferView(Arc::clone(b))).collect()
    }

    /// The time it was written at, or `None` for the value the slot holds
    /// until its writer first writes (zeros, or an initial value).
    #[getter]
    fn timestamp(&self) -> Option<PyTimestamp> {
        self.0
            .time_ns()
            .map(|ns| PyTimestamp(Timestamp::from_ns(ns)))
    }
}

/// The latest value of one writer, read by any number of readers.
#[pyclass(frozen, module = "mujou._core._native", name = "Slot")]
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
    /// Returns `None` after `timeout` seconds of real time.
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
        last: PyRef<'_, PyMessage>,
        timeout: f64,
    ) -> PyResult<Option<PyMessage>> {
        let timeout = to_duration(timeout)?;
        let last = Arc::clone(&last.0);
        // A timeout too long to represent never expires.
        let deadline = Instant::now().checked_add(timeout);
        let msg = wait_interruptibly(py, deadline, |step| self.0.wait_newer(&last, step))?;
        Ok(msg.map(PyMessage))
    }

    /// Sets the value and its time as they were when the state was taken
    /// (`None` for a value its writer had not written).
    ///
    /// Call it only before the run starts: it writes beside the slot's own
    /// writer. Raises as `Outlet.send` does.
    fn restore(&self, values: Values<'_>, time_ns: Option<u64>) -> PyResult<()> {
        let slot = &self.0;
        write_values(&values.0, slot.schema().fields(), slot.name(), |fill| {
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
///         schema (the message names the slot and the field).
///     TypeError: A value does not support DLPack or is not in host memory.
///         Pinned memory is read without synchronizing any CUDA stream, so
///         synchronize the copy that filled it before sending.
///     IndexError: No such channel.
#[pyclass(frozen, module = "mujou._core._native", name = "Outlet")]
pub struct PyOutlet(core::Outlet);

impl PyOutlet {
    fn send_with(&self, target: Target, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let outlet = &self.0;
        let name = match target {
            Target::All => outlet.outbox().name(),
            Target::Channel(k) => outlet.channel(k).name(),
        };
        write_values(values, outlet.schema().fields(), name, |fill| {
            outlet.send(target, fill)
        })
    }
}

#[pymethods]
impl PyOutlet {
    /// Creates the slots `"{name}/out"` and `"{name}/ch{k}"` holding zeros
    /// without a time. Every message sent is stamped with the time on `clock`.
    ///
    /// The slots and the host's inputs of one run must use the clock its
    /// runtime runs on: on another clock, the times of the messages would not
    /// be on the axis the nodes' Δt is measured on, a pause of the run would
    /// not stop them, and its end would not bound them, so a state taken at T
    /// could hold later messages.
    #[new]
    #[pyo3(signature = (schema, channels, name, *, clock))]
    fn new(
        schema: PyRef<'_, PySchema>,
        channels: usize,
        name: &str,
        clock: PyRef<'_, PyReadOnlyTimer>,
    ) -> Self {
        Self(core::Outlet::new(
            Arc::clone(&schema.0),
            channels,
            name,
            Arc::clone(&clock.0),
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
#[pyclass(frozen, module = "mujou._core._native", name = "InputGroup")]
pub struct PyInputGroup(Arc<core::InputGroup>);

#[pymethods]
impl PyInputGroup {
    /// Every write is stamped with the time on `clock`, as `Outlet` says.
    #[new]
    #[pyo3(signature = (*, clock))]
    fn new(clock: PyRef<'_, PyReadOnlyTimer>) -> Self {
        Self(core::InputGroup::new(Arc::clone(&clock.0)))
    }

    /// Creates the slot `"{name}/in"` holding zeros without a time.
    fn input(&self, schema: PyRef<'_, PySchema>, name: &str) -> PyInputSlot {
        PyInputSlot(self.0.input(Arc::clone(&schema.0), name))
    }

    /// Writes every item at once, all stamped with the same time, and returns
    /// that time.
    ///
    /// No reader sees some of the slots new and others old at any one moment;
    /// a reader that reads two slots one after the other may still see a write
    /// complete in between. If any value is invalid, no slot changes. Raises as
    /// `Outlet.send` does. Every slot is one of this group's, given once (the
    /// core panics otherwise).
    fn write(
        &self,
        py: Python<'_>,
        items: Vec<(PyRef<'_, PyInputSlot>, Values<'_>)>,
    ) -> PyResult<PyTimestamp> {
        let mut slots: Vec<&core::Slot> = Vec::with_capacity(items.len());
        let mut borrowed = Vec::with_capacity(items.len());
        for (slot, values) in &items {
            let slot = &*slot.0;
            borrowed.push(borrow_values(
                &values.0,
                slot.schema().fields(),
                slot.name(),
            )?);
            slots.push(slot);
        }
        let tensors: Vec<&[TensorPtr]> = borrowed.iter().map(|b| b.tensors.as_slice()).collect();
        py.detach(|| {
            self.0.write(&slots, |i, bytes| {
                copy_into(
                    tensors[i],
                    slots[i].schema().fields(),
                    slots[i].name(),
                    bytes,
                )
            })
        })
        .map(PyTimestamp)
        .map_err(to_py_err)
    }
}

/// A slot the host writes through its `InputGroup`.
///
/// The same core slot as a `Slot`, as a type of its own so that
/// `InputGroup.write` takes only the slots of an `InputGroup` (never the
/// channel slot of an `Outlet`), and so that only a `Slot` waits for a newer
/// message (`wait_newer`).
#[pyclass(frozen, module = "mujou._core._native", name = "InputSlot")]
pub struct PyInputSlot(Arc<core::Slot>);

#[pymethods]
impl PyInputSlot {
    #[getter]
    fn name(&self) -> &str {
        self.0.name()
    }

    fn read(&self) -> PyMessage {
        PyMessage(self.0.load())
    }

    /// Sets the value and its time (`None` for a value the host had not
    /// written), dropping any pending write.
    ///
    /// Call it only before the run starts, while the host writes nothing.
    /// Raises as `Outlet.send` does.
    fn restore(&self, values: Values<'_>, time_ns: Option<u64>) -> PyResult<()> {
        let slot = &self.0;
        write_values(&values.0, slot.schema().fields(), slot.name(), |fill| {
            slot.restore(time_ns, fill)
        })
    }
}
