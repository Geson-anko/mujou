//! DLPack: copying values in from, and lending them out to, other libraries.
//!
//! The core has no tensor type or dtype of its own. Values cross the boundary as
//! DLPack tensors (<https://dmlc.github.io/dlpack/latest/>). A tensor in host
//! memory is always copied: into a pooled buffer when written ([`import`]), and
//! out of it by the reader, which borrows the buffer ([`export`]). A tensor on a
//! CUDA device is only checked ([`import::check`]): the bindings hold a copy that
//! the writer made on the device as a [`crate::buffer::Held`] value
//! (`specs/messages.md`).

pub mod export;
pub mod ffi;
pub mod import;
