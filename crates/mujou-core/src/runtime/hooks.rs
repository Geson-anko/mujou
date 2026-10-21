//! What the runtime runs: the hooks of a node ([`NodeHooks`]), the whole
//! contract between the runtime and what the nodes compute, and how a node is
//! given to it ([`NodeSpec`]).

use std::num::NonZeroU64;

use super::error::HookError;

/// The lifecycle hooks of a node, called by the runtime.
///
/// Every hook except `capture` runs on the thread that runs the node: its
/// dedicated thread, or any worker of the pool (a different one from call to
/// call). The hooks of a node never overlap, and each one sees what the
/// previous one wrote. `capture` runs on the control thread while the node is
/// paused, or after a normal end on the thread that asked for it, so it never
/// overlaps with the other hooks either. It is not called after an exception.
///
/// A hook that panics is treated like one that returned an error, under the
/// hook's [`Hook`](super::Hook). The hooks with a default do nothing.
pub trait NodeHooks: Send + 'static {
    type State: Send + 'static;

    /// Runs the whole body of a runtime thread (a dedicated thread, a worker
    /// or the control thread). The bindings of a language with per-thread
    /// state wrap it here, so that the state lives as long as the thread
    /// instead of one hook call.
    /// The body waits a lot, so the wrapper must not hold a lock others need.
    fn on_thread<R: Send>(body: impl FnOnce() -> R + Send) -> R {
        body()
    }

    /// Told that a hook panicked, with the error that will stand for it, so
    /// that the node can log it. The node may be in an inconsistent state.
    fn panicked(&mut self, _error: &HookError) {}

    /// Prepares the node on its thread before the first update. The time is
    /// frozen meanwhile, so the time this takes does not enter the first Δt.
    fn start(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    /// Restores the node's state, if it has one to restore, right after
    /// `start` and with the time still frozen.
    fn restore(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    /// Advances the node by `dt` seconds of the runtime's timer (always > 0).
    fn update(&mut self, dt: f64) -> Result<(), HookError>;

    fn pause(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    fn resume(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    /// Called once on every node when the system stops because of an exception,
    /// before `stop`. Its error is ignored.
    fn exception(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    /// Called once on every node at the end. Its error is ignored.
    fn stop(&mut self) -> Result<(), HookError> {
        Ok(())
    }

    /// Takes the node's state, while it is paused or after the end.
    fn capture(&mut self) -> Result<Self::State, HookError>;
}

/// One node of a runtime.
pub struct NodeSpec<N> {
    pub hooks: N,
    /// How the errors and the node's thread name it.
    pub name: String,
    /// The period on the timer when running freely: the node updates at
    /// targets a period apart, so once every period on average while its
    /// update and its wake-up fit in the period. A period of 1 ns updates as
    /// soon as any time has passed. The manual mode ignores it.
    pub period_ns: NonZeroU64,
    /// Runs on a thread of its own instead of the pool.
    pub dedicated_thread: bool,
    /// Where the first Δt starts: the time at the start if `None`.
    pub last_update_ns: Option<u64>,
}
