//! The control thread, which serves the host's requests one at a time.

use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::error::{finished_error, panic_message, stopped};
use super::freeze::FreezeReason;
use super::node::Command;
use super::{Control, NodeHooks, RUNTIME_OF_THREAD, RuntimeError, Shared, remaining};

/// A request from the host, run on the control thread.
pub(super) type Job<N> = Box<dyn FnOnce(&mut Controller<N>) + Send>;

pub(super) enum Message<N: NodeHooks> {
    Job(Job<N>),
    /// A node recorded the first failure; the control thread ends the system.
    Failed,
}

/// The control thread: serves the host's requests one at a time and ends the
/// system on a failure.
pub(super) struct Controller<N: NodeHooks> {
    pub(super) shared: Arc<Shared<N>>,
    pub(super) threads: Vec<JoinHandle<()>>,
    pub(super) pause_timeout: Duration,
    pub(super) done: bool,
}

impl<N: NodeHooks> Controller<N> {
    pub(super) fn run(
        mut self,
        messages: Receiver<Message<N>>,
        started: Sender<Result<(), RuntimeError>>,
    ) {
        RUNTIME_OF_THREAD.set(self.shared.id());
        let _ = started.send(self.startup());
        while !self.done {
            // `shared` holds a sender, so this does not disconnect.
            match messages.recv() {
                Ok(Message::Job(job)) => job(&mut self),
                Ok(Message::Failed) => self.finish(),
                Err(_) => break,
            }
        }
        // Dropping `messages` makes the pending and later requests fail fast.
    }

    fn startup(&mut self) -> Result<(), RuntimeError> {
        let s = Arc::clone(&self.shared);
        let mut c = s.lock();
        while !c.failed && !c.nodes.iter().all(|r| r.started) {
            c = s.wait(c);
        }
        if c.failed {
            drop(c);
            return Err(self.end());
        }
        // The timer was frozen at T before the nodes started, so `start` (and
        // restoring) took no time on it. Resume the time before letting the
        // nodes go, as after a pause, so that the first Δt is not 0.
        c.freeze.release(FreezeReason::Startup);
        c.go = true;
        s.changed.notify_all();
        Ok(())
    }

    /// Ends the system if a failure was recorded, and fails if it has ended.
    fn ensure_running(&mut self) -> Result<(), RuntimeError> {
        if self.done || self.shared.lock().failed {
            return Err(self.end());
        }
        Ok(())
    }

    pub(super) fn step(&mut self, order: Vec<usize>) -> Result<(), RuntimeError> {
        self.ensure_running()?;
        let s = Arc::clone(&self.shared);
        for i in order {
            let mut c = s.lock();
            if !c.failed {
                c.grant = Some(i);
                s.changed.notify_all();
            }
            while c.grant.is_some() && !c.failed {
                c = s.wait(c);
            }
            if c.failed {
                drop(c);
                return Err(self.end());
            }
        }
        Ok(())
    }

    pub(super) fn capture<R>(
        &mut self,
        build: impl FnOnce(Vec<N::State>) -> R,
    ) -> Result<R, RuntimeError> {
        self.ensure_running()?;
        let s = Arc::clone(&self.shared);
        let mut c = s.lock();
        // Raise the flag, then freeze the time right away instead of after the
        // nodes stop: otherwise the wait between a node's last update and the
        // freeze would enter that node's next Δt.
        s.set_command(&mut c, Command::Pause);
        let epoch = c.epoch;
        c.freeze.hold(FreezeReason::Capture, &s.timer);
        let deadline = Instant::now().checked_add(self.pause_timeout);
        while !c.failed && !c.nodes.iter().all(|r| r.paused_at == Some(epoch)) {
            let remaining = remaining(deadline);
            if remaining.is_zero() {
                let nodes = (c.nodes.iter().zip(&s.names))
                    .filter(|(r, _)| r.paused_at != Some(epoch))
                    .map(|(_, name)| name.clone())
                    .collect();
                // The nodes that did pause get `resume`.
                resume_after_pause(&s, &mut c);
                return Err(RuntimeError::PauseTimeout { nodes });
            }
            c = s.wait_timeout(c, remaining);
        }
        let failed = c.failed;
        drop(c);
        if failed {
            return Err(self.end());
        }
        // Every node waits at its loop boundary without holding its lock, and
        // the time stands at T: the states are consistent at T.
        let states = (0..s.hooks.len()).map(|i| s.call(i, "state_dict", N::state_dict));
        let Ok(states) = states.collect::<Result<Vec<_>, ()>>() else {
            return Err(self.end());
        };
        let result = panic::catch_unwind(AssertUnwindSafe(|| build(states)));
        // Resume even if `build` panicked, so the system is not left paused.
        resume_after_pause(&s, &mut s.lock());
        result.map_err(|payload| {
            let message = panic_message(payload);
            RuntimeError::Internal(format!("the build of a capture panicked: {message}"))
        })
    }

    /// Stops every node and ends the system. Does nothing once ended.
    pub(super) fn finish(&mut self) {
        if self.done {
            return;
        }
        let s = Arc::clone(&self.shared);
        let mut c = s.lock();
        s.set_command(&mut c, Command::Stop);
        // No timeout: a node in a long update is waited for (Ctrl-C on the host
        // still works, since the host waits in steps).
        while !c.nodes.iter().all(|r| r.ready_to_stop(c.failed)) {
            c = s.wait(c);
        }
        c.stop_allowed = true;
        s.changed.notify_all();
        drop(c);
        for thread in self.threads.drain(..) {
            // A node thread does not panic: hook panics are caught.
            let _ = thread.join();
        }
        s.mark_finished(s.lock());
        self.done = true;
    }

    /// Ends the system (if not yet ended) and returns how it ended.
    fn end(&mut self) -> RuntimeError {
        self.finish();
        finished_error(&self.shared.failure)
    }

    pub(super) fn shutdown(&mut self) -> Result<(), RuntimeError> {
        self.finish();
        stopped(&self.shared.failure)
    }
}

/// Ends a pause: resume the time first, then lower the flag, so that the first
/// Δt after the pause is not 0 and no wait enters it.
fn resume_after_pause<N: NodeHooks>(s: &Shared<N>, c: &mut Control) {
    c.freeze.release(FreezeReason::Capture);
    // A no-op if an exception already turned the command to `Stop`.
    s.set_command(c, Command::Run);
}
