//! The native core of mujou, independent of Python.
//!
//! Python bindings live in the `mujou-python` crate, so that this crate can be
//! built and tested without linking libpython.

use std::sync::{Mutex, MutexGuard};

pub mod buffer;
pub mod dlpack;
pub mod runtime;
pub mod schema;
pub mod slot;
pub mod time;

/// Locks `m` for the buffers and the slots. Nothing panics while holding their
/// locks, so a poisoned value is still valid.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}
