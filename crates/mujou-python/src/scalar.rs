//! Python scalars as the values of fields of shape `()`: an `int` of a field of
//! an integer dtype, a `float` of a `float64` field and a `bool` of a `bool`
//! field. The codec decides which fields are scalars (`IntSpec` and the like);
//! here a value without `__dlpack__` is read as one, by the field's dtype.

use mujou_core::buffer::Buffer;
use mujou_core::dlpack::ffi::DLDataType;
use mujou_core::schema::{FieldSpec, format_dtype, format_shape};
use pyo3::exceptions::{PyOverflowError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyBool;

/// The bytes of a scalar, in native byte order, read with the GIL to be copied
/// into a buffer without it.
pub(crate) struct Scalar {
    bytes: [u8; 8],
    len: usize,
}

impl Scalar {
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    fn new(bytes: &[u8]) -> Self {
        let mut scalar = Self {
            bytes: [0; 8],
            len: bytes.len(),
        };
        scalar.bytes[..bytes.len()].copy_from_slice(bytes);
        scalar
    }
}

/// Reads `value` as the scalar of `field`, which `at` names in the errors.
///
/// A value out of the range of the field (an `int` of an integer field, or one
/// too large for a `float64`) raises `OverflowError`; it is never wrapped. A
/// field that is not of shape `()` and of an `int` or `uint` dtype of 8 to 64
/// bits, `float64` or `bool` takes only a tensor.
pub(crate) fn read_scalar(
    value: &Bound<'_, PyAny>,
    field: &FieldSpec,
    at: &str,
) -> PyResult<Scalar> {
    let DLDataType { code, bits, .. } = field.dtype;
    let located = |e: PyErr| located(e, value, field, at);
    if field.shape.is_empty() {
        match (code, bits) {
            (0 | 1, 8 | 16 | 32 | 64) => {
                return read_int(value, code == 0, bits).map_err(located);
            }
            (2, 64) => {
                let v = value.extract::<f64>().map_err(located)?;
                return Ok(Scalar::new(&v.to_ne_bytes()));
            }
            (6, 8) => {
                let v = value.extract::<bool>().map_err(located)?;
                return Ok(Scalar::new(&[u8::from(v)]));
            }
            _ => {}
        }
    }
    Err(PyTypeError::new_err(format!(
        "{at}: expected a tensor supporting DLPack, such as torch.Tensor, got {}",
        value.get_type().name()?
    )))
}

/// Names `at` in an error of reading `value` as the scalar of `field`: an
/// `OverflowError` says the value and the dtype, any other gets a note.
fn located(e: PyErr, value: &Bound<'_, PyAny>, field: &FieldSpec, at: &str) -> PyErr {
    let py = value.py();
    if e.is_instance_of::<PyOverflowError>(py) {
        let dtype = format_dtype(field.dtype);
        return PyOverflowError::new_err(format!("{at}: {value} is out of range for {dtype}"));
    }
    let note = format!("while reading the value of {at} as a scalar");
    match e.value(py).call_method1(intern!(py, "add_note"), (note,)) {
        Ok(_) => e,
        Err(note_error) => note_error,
    }
}

/// Raises `OverflowError` for a value out of the range, which the caller names.
fn read_int(value: &Bound<'_, PyAny>, signed: bool, bits: u8) -> PyResult<Scalar> {
    let (min, max) = if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    };
    let v = value.extract::<i128>()?;
    if !(min..=max).contains(&v) {
        return Err(PyOverflowError::new_err("out of range"));
    }
    // Truncating keeps the low bits: the two's complement of a negative value.
    Ok(match bits {
        8 => Scalar::new(&(v as u8).to_ne_bytes()),
        16 => Scalar::new(&(v as u16).to_ne_bytes()),
        32 => Scalar::new(&(v as u32).to_ne_bytes()),
        _ => Scalar::new(&(v as u64).to_ne_bytes()),
    })
}

/// The value of `buffer` as a Python `int`, `float` or `bool`.
///
/// Raises `TypeError` for a field that `read_scalar` would not read.
pub(crate) fn to_python<'py>(py: Python<'py>, buffer: &Buffer) -> PyResult<Bound<'py, PyAny>> {
    let field = buffer.field();
    let b = buffer.bytes();
    if field.shape.is_empty() {
        let int = |v: i128| -> PyResult<Bound<'py, PyAny>> { Ok(v.into_pyobject(py)?.into_any()) };
        match (field.dtype.code, field.dtype.bits) {
            (0, 8) => return int(i8::from_ne_bytes([b[0]]).into()),
            (0, 16) => return int(i16::from_ne_bytes([b[0], b[1]]).into()),
            (0, 32) => return int(i32::from_ne_bytes(b.try_into().expect("4 bytes")).into()),
            (0, 64) => return int(i64::from_ne_bytes(b.try_into().expect("8 bytes")).into()),
            (1, 8) => return int(b[0].into()),
            (1, 16) => return int(u16::from_ne_bytes([b[0], b[1]]).into()),
            (1, 32) => return int(u32::from_ne_bytes(b.try_into().expect("4 bytes")).into()),
            (1, 64) => return int(u64::from_ne_bytes(b.try_into().expect("8 bytes")).into()),
            (2, 64) => {
                let v = f64::from_ne_bytes(b.try_into().expect("8 bytes"));
                return Ok(v.into_pyobject(py)?.into_any());
            }
            (6, 8) => return Ok(PyBool::new(py, b[0] != 0).to_owned().into_any()),
            _ => {}
        }
    }
    Err(PyTypeError::new_err(format!(
        "a field of shape {} and dtype {} is not a scalar",
        format_shape(&field.shape),
        format_dtype(field.dtype)
    )))
}
