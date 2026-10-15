//! DLPack: copying values in from, and lending them out to, other libraries.
//!
//! The core has no tensor type or dtype of its own. Values cross the boundary as
//! DLPack tensors (<https://dmlc.github.io/dlpack/latest/>), and are always copied:
//! into a pooled buffer when written, and out of it by the reader (`docs/messages.md`).
//! The structs here mirror `dlpack.h` (version 1.x) field by field.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use crate::slot::{FieldSpec, Message, SlotError};

/// `kDLCPU`: ordinary host memory.
pub const DEVICE_CPU: i32 = 1;
/// `kDLCUDAHost`: pinned host memory, readable from the CPU as well.
pub const DEVICE_CUDA_HOST: i32 = 3;

/// The DLPack version of the tensors this module exports.
pub const VERSION: DLPackVersion = DLPackVersion { major: 1, minor: 0 };

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDevice {
    pub device_type: i32,
    pub device_id: i32,
}

/// The element type: `code` is the kind (0 int, 1 uint, 2 float, 4 bfloat, 6 bool,
/// ...), `bits` the size of one lane and `lanes` the vector width.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Debug)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    /// `ndim` extents.
    pub shape: *mut i64,
    /// `ndim` strides in elements, or null for a compact row-major tensor.
    pub strides: *mut i64,
    /// Bytes from `data` to the first element.
    pub byte_offset: u64,
}

/// The tensor of the unversioned (pre-1.0) protocol, capsule name `"dltensor"`.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}

/// The tensor of the 1.x protocol, capsule name `"dltensor_versioned"`.
#[repr(C)]
#[derive(Debug)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}

/// Copies the tensor `t` into `dst` as compact row-major bytes, after checking it
/// against `field`. Any strides are accepted, negative and zero (broadcast) ones
/// included.
///
/// Pinned host memory (`kDLCUDAHost`) is read without synchronizing any CUDA
/// stream, since no stream is passed to the producer. The writer must have
/// synchronized the device-to-host copy that filled it before handing it over.
///
/// The error names the field but not the slot; the caller adds it with
/// [`SlotError::at_slot`].
///
/// # Safety
///
/// `t` must describe a live tensor as DLPack requires: `shape` (and `strides`,
/// unless null) point to `ndim` values, and every element addressed by `data`,
/// `byte_offset`, `shape` and `strides` is readable host memory if the device is
/// the CPU or pinned host memory. `dst` must not overlap the tensor.
pub unsafe fn copy_from(t: &DLTensor, field: &FieldSpec, dst: &mut [u8]) -> Result<(), SlotError> {
    if t.device.device_type != DEVICE_CPU && t.device.device_type != DEVICE_CUDA_HOST {
        return Err(SlotError::UnsupportedDevice {
            slot: String::new(),
            field: field.name.clone(),
            device_type: t.device.device_type,
        });
    }
    if t.dtype != field.dtype {
        return Err(SlotError::Mismatch {
            slot: String::new(),
            field: field.name.clone(),
            what: "dtype",
            expected: format_dtype(field.dtype),
            actual: format_dtype(t.dtype),
        });
    }
    let ndim = usize::try_from(t.ndim).map_err(|_| invalid(field, "negative ndim"))?;
    let shape = if ndim == 0 {
        &[][..]
    } else {
        // SAFETY: the caller guarantees `shape` points to `ndim` values.
        unsafe { std::slice::from_raw_parts(t.shape, ndim) }
    };
    if shape != field.shape.as_slice() {
        return Err(SlotError::Mismatch {
            slot: String::new(),
            field: field.name.clone(),
            what: "shape",
            expected: format_shape(&field.shape),
            actual: format_shape(shape),
        });
    }
    if t.data.is_null() {
        return Err(invalid(field, "null data pointer"));
    }
    debug_assert_eq!(dst.len(), field.nbytes());
    let elem = field.elem_bytes();
    // SAFETY: the caller guarantees `byte_offset` stays inside the tensor's memory.
    let base = unsafe { (t.data as *const u8).add(t.byte_offset as usize) };
    let strides = if t.strides.is_null() || ndim == 0 {
        None
    } else {
        // SAFETY: the caller guarantees non-null `strides` points to `ndim` values.
        Some(unsafe { std::slice::from_raw_parts(t.strides, ndim) })
    };
    match strides {
        Some(strides) if !is_row_major(shape, strides) => {
            // SAFETY: the shape matches `field`, so `dst` holds exactly the
            // elements that `shape` and `strides` address, which the caller
            // guarantees are readable.
            unsafe { copy_strided(base, shape, strides, elem, dst) };
        }
        // SAFETY: a compact row-major tensor of this shape spans exactly
        // `dst.len()` readable bytes from `base`.
        _ => unsafe { ptr::copy_nonoverlapping(base, dst.as_mut_ptr(), dst.len()) },
    }
    Ok(())
}

fn invalid(field: &FieldSpec, reason: &str) -> SlotError {
    SlotError::InvalidDLPack {
        slot: String::new(),
        field: field.name.clone(),
        reason: reason.to_owned(),
    }
}

/// Whether the elements lie compactly in row-major order. The stride of an
/// extent-1 dimension never moves the address, so producers may set it freely.
fn is_row_major(shape: &[i64], strides: &[i64]) -> bool {
    let mut expected = 1;
    for (&n, &s) in shape.iter().zip(strides).rev() {
        if n != 1 && s != expected {
            return false;
        }
        expected *= n;
    }
    true
}

/// Gathers a strided tensor row by row, where a row is the last dimension.
///
/// # Safety
///
/// Every element `base + elem * sum_d(index[d] * strides[d])` for `index` within
/// `shape` must be readable, and `dst` must hold `elem * prod(shape)` bytes.
unsafe fn copy_strided(
    base: *const u8,
    shape: &[i64],
    strides: &[i64],
    elem: usize,
    dst: &mut [u8],
) {
    let last = shape.len() - 1;
    let row_len = shape[last] as usize;
    let row_stride = strides[last] as isize * elem as isize;
    // The index over every dimension but the last, advanced like an odometer.
    let mut index = vec![0i64; last];
    for row in dst.chunks_exact_mut(row_len * elem) {
        let offset: i64 = index.iter().zip(strides).map(|(i, s)| i * s).sum();
        // SAFETY: `index` is within `shape`, so this is the first element of a
        // row that the caller guarantees is readable.
        let src = unsafe { base.offset(offset as isize * elem as isize) };
        if row_stride == elem as isize {
            // SAFETY: the row is compact, so its `row.len()` bytes are readable.
            unsafe { ptr::copy_nonoverlapping(src, row.as_mut_ptr(), row.len()) };
        } else {
            for (j, out) in row.chunks_exact_mut(elem).enumerate() {
                // SAFETY: `j < row_len`, so this is an element of the row.
                unsafe {
                    ptr::copy_nonoverlapping(
                        src.offset(j as isize * row_stride),
                        out.as_mut_ptr(),
                        elem,
                    )
                };
            }
        }
        for d in (0..last).rev() {
            index[d] += 1;
            if index[d] < shape[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// `(2, 3)`, `(5,)` or `()`, as Python prints a shape.
pub(crate) fn format_shape(shape: &[i64]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => {
            let dims: Vec<String> = shape.iter().map(i64::to_string).collect();
            format!("({})", dims.join(", "))
        }
    }
}

/// The name torch and NumPy use for the scalar dtypes, so that error messages
/// read naturally; the raw code otherwise.
pub(crate) fn format_dtype(dtype: DLDataType) -> String {
    let DLDataType { code, bits, lanes } = dtype;
    match (code, lanes) {
        (0, 1) => format!("int{bits}"),
        (1, 1) => format!("uint{bits}"),
        (2, 1) => format!("float{bits}"),
        (4, 1) => format!("bfloat{bits}"),
        (6, 1) if bits == 8 => "bool".to_owned(),
        _ => format!("DLDataType(code={code}, bits={bits}, lanes={lanes})"),
    }
}

/// What a lent tensor keeps alive: the message, so that its buffer cannot go back
/// to the pool, and the shape and strides that the tensor points to.
struct Lent<T> {
    managed: T,
    _message: Arc<Message>,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

/// Explicit strides, since DLPack 1.2 asks producers not to leave them null.
fn row_major_strides(shape: &[i64]) -> Box<[i64]> {
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides.into_boxed_slice()
}

/// Boxes the managed tensor `make` builds around buffer `i` of `msg`, and points
/// it at the shape, the strides and the box itself (its `manager_ctx`). `parts`
/// picks the tensor and the context out of the managed struct.
fn lend<T>(
    msg: Arc<Message>,
    i: usize,
    make: impl FnOnce(DLTensor) -> T,
    parts: fn(&mut T) -> (&mut DLTensor, &mut *mut c_void),
) -> *mut T {
    let buffer = &msg.buffers()[i];
    let shape: Box<[i64]> = buffer.field().shape.clone().into_boxed_slice();
    let tensor = DLTensor {
        // The pointer is mutable only because DLPack says so: the buffer is
        // immutable, and every reader copies it out instead of writing to it.
        // READ_ONLY is not set, because consumers may refuse such tensors.
        data: buffer.bytes().as_ptr() as *mut c_void,
        device: DLDevice {
            device_type: DEVICE_CPU,
            device_id: 0,
        },
        ndim: shape.len() as i32,
        dtype: buffer.field().dtype,
        // Set below, once the box is in place.
        shape: ptr::null_mut(),
        strides: ptr::null_mut(),
        byte_offset: 0,
    };
    let lent = Box::into_raw(Box::new(Lent {
        managed: make(tensor),
        strides: row_major_strides(&shape),
        shape,
        _message: msg,
    }));
    // SAFETY: `lent` was just allocated and is not shared yet. The pointers are
    // taken after `into_raw`, from the slices the box owns, so they stay valid
    // until the deleter drops the box.
    unsafe {
        let shape = (*lent).shape.as_mut_ptr();
        let strides = (*lent).strides.as_mut_ptr();
        let (tensor, ctx) = parts(&mut (*lent).managed);
        tensor.shape = shape;
        tensor.strides = strides;
        *ctx = lent.cast();
        &raw mut (*lent).managed
    }
}

/// Lends buffer `i` of `msg` as a 1.x DLPack tensor. The message stays alive
/// until the consumer calls the tensor's deleter.
pub fn export_versioned(msg: Arc<Message>, i: usize) -> *mut DLManagedTensorVersioned {
    let make = |dl_tensor| DLManagedTensorVersioned {
        version: VERSION,
        manager_ctx: ptr::null_mut(),
        deleter: Some(delete_versioned),
        flags: 0,
        dl_tensor,
    };
    lend(msg, i, make, |m| (&mut m.dl_tensor, &mut m.manager_ctx))
}

/// Lends buffer `i` of `msg` as an unversioned DLPack tensor, for consumers that
/// do not know the 1.x protocol.
pub fn export_legacy(msg: Arc<Message>, i: usize) -> *mut DLManagedTensor {
    let make = |dl_tensor| DLManagedTensor {
        dl_tensor,
        manager_ctx: ptr::null_mut(),
        deleter: Some(delete_legacy),
    };
    lend(msg, i, make, |m| (&mut m.dl_tensor, &mut m.manager_ctx))
}

unsafe extern "C" fn delete_versioned(t: *mut DLManagedTensorVersioned) {
    // SAFETY: DLPack calls the deleter once, with the tensor `export_versioned`
    // returned, whose `manager_ctx` is the `Lent` box that owns it.
    unsafe {
        drop(Box::from_raw(
            (*t).manager_ctx.cast::<Lent<DLManagedTensorVersioned>>(),
        ))
    };
}

unsafe extern "C" fn delete_legacy(t: *mut DLManagedTensor) {
    // SAFETY: as in `delete_versioned`, for `export_legacy`.
    unsafe {
        drop(Box::from_raw(
            (*t).manager_ctx.cast::<Lent<DLManagedTensor>>(),
        ))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: DLDataType = DLDataType {
        code: 2,
        bits: 32,
        lanes: 1,
    };

    fn field(shape: &[i64]) -> FieldSpec {
        FieldSpec {
            name: "x".to_owned(),
            shape: shape.to_vec(),
            dtype: F32,
        }
    }

    /// A tensor over `data` (as f32), keeping its shape and strides alive.
    struct Tensor {
        data: Vec<f32>,
        shape: Vec<i64>,
        strides: Option<Vec<i64>>,
        byte_offset: u64,
        device_type: i32,
    }

    impl Tensor {
        fn new(data: Vec<f32>, shape: &[i64], strides: Option<&[i64]>) -> Self {
            Self {
                data,
                shape: shape.to_vec(),
                strides: strides.map(<[i64]>::to_vec),
                byte_offset: 0,
                device_type: DEVICE_CPU,
            }
        }

        fn dl(&mut self) -> DLTensor {
            DLTensor {
                data: self.data.as_mut_ptr().cast(),
                device: DLDevice {
                    device_type: self.device_type,
                    device_id: 0,
                },
                ndim: self.shape.len() as i32,
                dtype: F32,
                shape: self.shape.as_mut_ptr(),
                strides: self
                    .strides
                    .as_mut()
                    .map_or(ptr::null_mut(), |s| s.as_mut_ptr()),
                byte_offset: self.byte_offset,
            }
        }

        fn copy(&mut self, field: &FieldSpec) -> Result<Vec<f32>, SlotError> {
            let mut dst = vec![0u8; field.nbytes()];
            let t = self.dl();
            // SAFETY: `t` describes `self.data`, which outlives the call.
            unsafe { copy_from(&t, field, &mut dst) }?;
            Ok(dst
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect())
        }
    }

    fn range(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn copies_contiguous_with_and_without_strides() {
        let f = field(&[2, 3]);
        assert_eq!(
            Tensor::new(range(6), &[2, 3], None).copy(&f).unwrap(),
            range(6)
        );
        let strided = Tensor::new(range(6), &[2, 3], Some(&[3, 1])).copy(&f);
        assert_eq!(strided.unwrap(), range(6));
    }

    #[test]
    fn copies_transposed() {
        // The transpose of [[0, 1, 2], [3, 4, 5]].
        let mut t = Tensor::new(range(6), &[3, 2], Some(&[1, 3]));
        assert_eq!(t.copy(&field(&[3, 2])).unwrap(), [0., 3., 1., 4., 2., 5.]);
    }

    #[test]
    fn copies_negative_strides_and_byte_offset() {
        // [5, 4, 3, 2, 1, 0] starting at the last element and walking backwards.
        let mut t = Tensor::new(range(6), &[6], Some(&[-1]));
        t.byte_offset = 5 * 4;
        assert_eq!(t.copy(&field(&[6])).unwrap(), [5., 4., 3., 2., 1., 0.]);
    }

    #[test]
    fn copies_every_other_column_of_3d() {
        // x[:, :, ::2] of a (2, 2, 4) tensor.
        let mut t = Tensor::new(range(16), &[2, 2, 2], Some(&[8, 4, 2]));
        let expected = [0., 2., 4., 6., 8., 10., 12., 14.];
        assert_eq!(t.copy(&field(&[2, 2, 2])).unwrap(), expected);
    }

    #[test]
    fn copies_zero_dimensional() {
        assert_eq!(
            Tensor::new(vec![7.0], &[], None).copy(&field(&[])).unwrap(),
            [7.0]
        );
    }

    #[test]
    fn copies_broadcast_dimensions() {
        // [0, 1, 2] repeated over rows (stride 0 outside), and each of
        // [[0], [1]] repeated along a row (stride 0 inside).
        let mut rows = Tensor::new(range(3), &[2, 3], Some(&[0, 1]));
        assert_eq!(
            rows.copy(&field(&[2, 3])).unwrap(),
            [0., 1., 2., 0., 1., 2.]
        );
        let mut cols = Tensor::new(range(2), &[2, 3], Some(&[1, 0]));
        assert_eq!(
            cols.copy(&field(&[2, 3])).unwrap(),
            [0., 0., 0., 1., 1., 1.]
        );
    }

    #[test]
    fn ignores_strides_of_extent_one_dimensions() {
        let mut t = Tensor::new(range(3), &[1, 3], Some(&[99, 1]));
        assert_eq!(t.copy(&field(&[1, 3])).unwrap(), range(3));
    }

    #[test]
    fn rejects_device_memory() {
        let mut t = Tensor::new(range(6), &[6], None);
        t.device_type = 2; // kDLCUDA
        let err = t.copy(&field(&[6])).unwrap_err();
        assert!(matches!(
            err,
            SlotError::UnsupportedDevice { device_type: 2, .. }
        ));
    }

    #[test]
    fn accepts_pinned_host_memory() {
        let mut t = Tensor::new(range(6), &[6], None);
        t.device_type = DEVICE_CUDA_HOST;
        assert_eq!(t.copy(&field(&[6])).unwrap(), range(6));
    }

    #[test]
    fn rejects_shape_and_dtype_mismatch() {
        let err = Tensor::new(range(6), &[3, 2], None)
            .copy(&field(&[2, 3]))
            .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("(2, 3)") && message.contains("(3, 2)"),
            "{message}"
        );

        let mut f = field(&[6]);
        f.dtype.bits = 64;
        let err = Tensor::new(range(6), &[6], None).copy(&f).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("float64") && message.contains("float32"),
            "{message}"
        );
    }
}
