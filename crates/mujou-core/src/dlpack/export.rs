//! Lending the buffer of a message to a consumer as a DLPack tensor.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use crate::buffer::Buffer;
use crate::dlpack::ffi::{
    DEVICE_CPU, DLDevice, DLManagedTensor, DLManagedTensorVersioned, DLTensor, VERSION,
};

/// What a lent tensor keeps alive: the buffer, so that it cannot go back to its
/// pool, and the shape and strides that the tensor points to.
struct Lent<T> {
    managed: T,
    _buffer: Arc<Buffer>,
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

/// Boxes the managed tensor `make` builds around `buffer`, and points it at the
/// shape, the strides and the box itself (its `manager_ctx`). `parts` picks the
/// tensor and the context out of the managed struct.
fn lend<T>(
    buffer: Arc<Buffer>,
    make: impl FnOnce(DLTensor) -> T,
    parts: fn(&mut T) -> (&mut DLTensor, &mut *mut c_void),
) -> *mut T {
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
        _buffer: buffer,
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

/// Lends `buffer` as a 1.x DLPack tensor. The buffer stays alive (out of its
/// pool) until the consumer calls the tensor's deleter.
pub fn export_versioned(buffer: Arc<Buffer>) -> *mut DLManagedTensorVersioned {
    let make = |dl_tensor| DLManagedTensorVersioned {
        version: VERSION,
        manager_ctx: ptr::null_mut(),
        deleter: Some(delete_versioned),
        flags: 0,
        dl_tensor,
    };
    lend(buffer, make, |m| (&mut m.dl_tensor, &mut m.manager_ctx))
}

/// Lends `buffer` as an unversioned DLPack tensor, for consumers that do not
/// know the 1.x protocol.
pub fn export_legacy(buffer: Arc<Buffer>) -> *mut DLManagedTensor {
    let make = |dl_tensor| DLManagedTensor {
        dl_tensor,
        manager_ctx: ptr::null_mut(),
        deleter: Some(delete_legacy),
    };
    lend(buffer, make, |m| (&mut m.dl_tensor, &mut m.manager_ctx))
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
    use crate::buffer::Messages;
    use crate::dlpack::ffi::DLDataType;
    use crate::dlpack::import::copy_from;
    use crate::schema::{FieldSpec, Schema};

    /// Fills the field of shape (2, 3) with `k`, `k + 1`, ...
    fn fill_range(k: i64) -> impl FnOnce(&mut [&mut [u8]]) -> Result<(), ()> {
        move |bytes| {
            for (i, chunk) in bytes[0].chunks_exact_mut(8).enumerate() {
                chunk.copy_from_slice(&(k + i as i64).to_ne_bytes());
            }
            Ok(())
        }
    }

    #[test]
    fn lent_tensor_keeps_buffer_out_of_its_pool_until_deleted() {
        let field = FieldSpec {
            name: "x".to_owned(),
            shape: vec![2, 3],
            dtype: DLDataType {
                code: 0,
                bits: 64,
                lanes: 1,
            },
        };
        let messages = Messages::new(Arc::new(Schema::new(vec![field]).unwrap()));
        let pool = Arc::clone(&messages.pools()[0]);
        for (k, versioned) in [(0, true), (100, false)] {
            let msg = messages.build(None, fill_range(k)).unwrap();
            let buffer = Arc::clone(&msg.values()[0]);
            let field = buffer.field().clone();
            let expected = buffer.bytes().to_vec();
            let before = Arc::strong_count(&buffer);
            let (tensor, delete): (&DLTensor, Box<dyn FnOnce()>) = if versioned {
                let t = export_versioned(Arc::clone(&buffer));
                // SAFETY: `t` stays valid until its deleter is called below.
                unsafe {
                    assert_eq!((*t).version, VERSION);
                    (&(*t).dl_tensor, Box::new(move || (*t).deleter.unwrap()(t)))
                }
            } else {
                let t = export_legacy(Arc::clone(&buffer));
                // SAFETY: as above.
                unsafe { (&(*t).dl_tensor, Box::new(move || (*t).deleter.unwrap()(t))) }
            };
            assert_eq!(Arc::strong_count(&buffer), before + 1);
            // Drop the message: only the tensor holds the buffer now, so later
            // messages must not reuse it.
            drop((buffer, msg));
            for j in 1..=3 {
                drop(messages.build(None, fill_range(-j)).unwrap());
            }
            let mut copy = vec![0u8; field.nbytes()];
            // SAFETY: the exported tensor describes the live buffer.
            unsafe { copy_from(tensor, &field, &mut copy) }.unwrap();
            assert_eq!(copy, expected);
            // SAFETY: the strides point to `ndim` values.
            let strides = unsafe { std::slice::from_raw_parts(tensor.strides, 2) };
            assert_eq!(strides, [3, 1]);
            let free = pool.free();
            delete();
            assert_eq!(pool.free(), free + 1, "back in its pool");
        }
    }
}
