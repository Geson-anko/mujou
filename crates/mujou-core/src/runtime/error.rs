//! What the runtime reports: the errors of the hooks and of the requests.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, OnceLock};

/// An error raised by a hook, shared because it is reported to several places.
pub type SharedError = Arc<dyn Error + Send + Sync>;

/// A hook failed. `hook` names the hook as the node's author knows it.
#[derive(Debug, Clone)]
pub struct HookError {
    pub hook: &'static str,
    pub source: SharedError,
}

/// The first hook error, which ended the system.
#[derive(Debug, Clone)]
pub struct NodeFailure {
    pub node: usize,
    pub hook: &'static str,
    pub error: SharedError,
}

#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// The system stopped because of a hook error (or panic).
    Failed(NodeFailure),
    /// A capture gave up because these nodes did not pause in time. The system
    /// keeps running.
    PauseTimeout {
        nodes: Vec<String>,
    },
    /// Stepping by hand needs the manual mode.
    NotManual,
    /// The system has already stopped normally.
    Finished,
    InvalidArgument(String),
    /// A thread of this runtime (a hook or a callback of the runtime) asked it
    /// to do something it would have to wait for, which would never finish.
    Reentrant,
    /// The runtime could not do what was asked: a thread could not be spawned,
    /// or a callback (the `build` of a capture) panicked.
    Internal(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed(failure) => write!(
                f,
                "node #{} raised in {}: {}",
                failure.node, failure.hook, failure.error
            ),
            Self::PauseTimeout { nodes } => write!(
                f,
                "pausing timed out; these nodes did not stop: {}",
                nodes.join(", ")
            ),
            Self::NotManual => write!(f, "step needs the manual mode"),
            Self::Finished => write!(f, "the runtime has finished"),
            Self::InvalidArgument(message) => f.write_str(message),
            Self::Reentrant => write!(
                f,
                "cannot wait for the runtime from one of its own threads (a hook or a callback)"
            ),
            Self::Internal(message) => f.write_str(message),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Failed(failure) => Some(failure.error.as_ref()),
            _ => None,
        }
    }
}

/// A hook panicked. Treated like an exception (`specs/execution-model.md`).
#[derive(Debug)]
pub(super) struct Panicked(pub(super) String);

impl fmt::Display for Panicked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "panicked: {}", self.0)
    }
}

impl Error for Panicked {}

pub(super) fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map_or_else(|| "unknown payload".to_owned(), |s| (*s).to_owned()),
    }
}

pub(super) fn finished_error(failure: &OnceLock<NodeFailure>) -> RuntimeError {
    match failure.get() {
        Some(failure) => RuntimeError::Failed(failure.clone()),
        None => RuntimeError::Finished,
    }
}

/// What a request returns when the system has already ended.
pub(super) type Ended<T> = fn(&OnceLock<NodeFailure>) -> Result<T, RuntimeError>;

pub(super) fn unavailable<T>(failure: &OnceLock<NodeFailure>) -> Result<T, RuntimeError> {
    Err(finished_error(failure))
}

/// A stop request succeeds on a system that has stopped, unless by a failure.
pub(super) fn stopped(failure: &OnceLock<NodeFailure>) -> Result<(), RuntimeError> {
    match failure.get() {
        Some(failure) => Err(RuntimeError::Failed(failure.clone())),
        None => Ok(()),
    }
}
