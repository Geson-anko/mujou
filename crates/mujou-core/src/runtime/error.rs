//! What the runtime reports: the errors of the hooks and of the requests.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, OnceLock};

/// A hook of a node. It is displayed as the node's author knows it
/// (`on_update` for [`Hook::Update`], ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hook {
    Start,
    Restore,
    Update,
    Pause,
    Resume,
    Exception,
    Stop,
    Capture,
}

impl Hook {
    pub fn name(self) -> &'static str {
        match self {
            Self::Start => "on_start",
            Self::Restore => "load_state_dict",
            Self::Update => "on_update",
            Self::Pause => "on_pause",
            Self::Resume => "on_resume",
            Self::Exception => "on_exception",
            Self::Stop => "on_stop",
            Self::Capture => "state_dict",
        }
    }
}

impl fmt::Display for Hook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An error raised by a hook, shared because it is reported to several places.
type SharedError = Arc<dyn Error + Send + Sync>;

/// A hook failed.
#[derive(Debug, Clone)]
pub struct HookError {
    pub hook: Hook,
    pub source: SharedError,
}

/// The first hook error, which ended the system.
#[derive(Debug, Clone)]
pub struct NodeFailure {
    /// The node's name, as given in its [`NodeSpec`](super::NodeSpec).
    pub name: String,
    pub hook: Hook,
    pub error: SharedError,
}

#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// The system stopped because of a hook error (or panic).
    Failed(NodeFailure),
    /// A capture after a normal end failed in a node's hook. Nothing else
    /// changed: the system had already ended normally, and stays so.
    CaptureFailed(NodeFailure),
    /// A capture gave up because these nodes did not pause in time. The system
    /// keeps running.
    PauseTimeout { nodes: Vec<String> },
    /// The system has already stopped normally.
    Finished,
    /// A thread of this runtime (a hook or a callback of the runtime) asked it
    /// to do something it would have to wait for, which would never finish.
    Reentrant,
    /// The runtime could not do what was asked: a thread could not be spawned,
    /// a callback (the `build` of a capture) panicked, or the caller broke the
    /// contract of a request. A bug of the caller, not of the nodes.
    Internal(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed(failure) | Self::CaptureFailed(failure) => write!(
                f,
                "node '{}' raised in {}: {}",
                failure.name, failure.hook, failure.error
            ),
            Self::PauseTimeout { nodes } => write!(
                f,
                "pausing timed out; these nodes did not stop: {}",
                nodes.join(", ")
            ),
            Self::Finished => write!(f, "the runtime has finished"),
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
            Self::Failed(failure) | Self::CaptureFailed(failure) => Some(failure.error.as_ref()),
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

/// What a request gets once the system has ended.
pub(super) fn finished_error(failure: &OnceLock<NodeFailure>) -> RuntimeError {
    match failure.get() {
        Some(failure) => RuntimeError::Failed(failure.clone()),
        None => RuntimeError::Finished,
    }
}

/// Fails with the failure, if one was recorded.
pub(super) fn stopped(failure: &OnceLock<NodeFailure>) -> Result<(), RuntimeError> {
    match failure.get() {
        Some(failure) => Err(RuntimeError::Failed(failure.clone())),
        None => Ok(()),
    }
}
