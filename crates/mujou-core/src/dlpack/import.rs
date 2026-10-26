//! Copying a producer's DLPack tensor into the buffer of a field.

use std::ptr;

use crate::dlpack::ffi::{DEVICE_CPU, DEVICE_CUDA_HOST, DLTensor};
use crate::schema::{FieldError, FieldSpec, format_dtype, format_shape};

/// Checks the dtype and the shape of the tensor `t` against `field`, wherever
/// its memory is: only the tensor's description is read.
///
/// The error does not name the field, which the caller knows.
///
/// # Safety
///
/// `t` must describe a live tensor as DLPack requires: `shape` points to `ndim`
/// values.
pub unsafe fn check(t: &DLTensor, field: &FieldSpec) -> Result<(), FieldError> {
    if t.dtype != field.dtype {
        return Err(FieldError::Mismatch {
            what: "dtype",
            expected: format_dtype(field.dtype),
            actual: format_dtype(t.dtype),
        });
    }
    let ndim = usize::try_from(t.ndim).map_err(|_| invalid("negative ndim"))?;
    let shape = if ndim == 0 {
        &[][..]
    } else {
        // SAFETY: the caller guarantees `shape` points to `ndim` values.
        unsafe { std::slice::from_raw_parts(t.shape, ndim) }
    };
    if shape != field.shape.as_slice() {
        return Err(FieldError::Mismatch {
            what: "shape",
            expected: format_shape(&field.shape),
            actual: format_shape(shape),
        });
    }
    Ok(())
}

/// Copies the tensor `t` into `dst` as compact row-major bytes, after checking it
/// against `field` ([`check`]). Any strides are accepted, negative and zero
/// (broadcast) ones included.
///
/// Pinned host memory (`kDLCUDAHost`) is read without synchronizing any CUDA
/// stream, since no stream is passed to the producer. The writer must have
/// synchronized the device-to-host copy that filled it before handing it over.
///
/// The error does not name the field, which the caller knows.
///
/// # Safety
///
/// `t` must describe a live tensor as DLPack requires: `shape` (and `strides`,
/// unless null) point to `ndim` values, and every element addressed by `data`,
/// `byte_offset`, `shape` and `strides` is readable host memory if the device is
/// the CPU or pinned host memory. `dst` must not overlap the tensor.
pub unsafe fn copy_from(t: &DLTensor, field: &FieldSpec, dst: &mut [u8]) -> Result<(), FieldError> {
    if t.device.device_type != DEVICE_CPU && t.device.device_type != DEVICE_CUDA_HOST {
        return Err(FieldError::UnsupportedDevice {
            device_type: t.device.device_type,
        });
    }
    // SAFETY: the caller's guarantee covers the check's.
    unsafe { check(t, field) }?;
    // Checked to be the tensor's own.
    let shape = field.shape.as_slice();
    let ndim = shape.len();
    if t.data.is_null() {
        return Err(invalid("null data pointer"));
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

fn invalid(reason: &str) -> FieldError {
    FieldError::InvalidDLPack {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlpack::ffi::{DEVICE_CUDA, DLDataType, DLDevice};

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

        fn copy(&mut self, field: &FieldSpec) -> Result<Vec<f32>, FieldError> {
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
        t.device_type = DEVICE_CUDA;
        let err = t.copy(&field(&[6])).unwrap_err();
        assert_eq!(
            err,
            FieldError::UnsupportedDevice {
                device_type: DEVICE_CUDA
            }
        );
    }

    #[test]
    fn checks_device_memory_without_reading_it() {
        let mut t = Tensor::new(Vec::new(), &[2, 3], None);
        t.device_type = DEVICE_CUDA;
        let mut dl = t.dl();
        dl.data = ptr::null_mut();
        // SAFETY: `dl` describes its shape, which `t` keeps alive.
        unsafe { check(&dl, &field(&[2, 3])) }.unwrap();
        // SAFETY: as above.
        let err = unsafe { check(&dl, &field(&[3, 2])) }.unwrap_err();
        assert!(matches!(err, FieldError::Mismatch { what: "shape", .. }));
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
