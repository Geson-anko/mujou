//! The structs of `dlpack.h` (version 1.x), mirrored field by field.

use std::ffi::c_void;

/// `kDLCPU`: ordinary host memory.
pub const DEVICE_CPU: i32 = 1;
/// `kDLCUDAHost`: pinned host memory, readable from the CPU as well.
pub(crate) const DEVICE_CUDA_HOST: i32 = 3;

/// The DLPack version of the tensors the core exports.
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
