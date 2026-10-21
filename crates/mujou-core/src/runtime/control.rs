//! The control thread, which serves the host's requests one at a time.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use super::error::finished_error;
use super::lifecycle::Command;
use super::{Control, NodeHooks, RuntimeError, Shared, build_states};
use crate::time::remaining;

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
    pub(super) done: bool,
}

impl<N: NodeHooks> Controller<N> {
    pub(super) fn run(
        mut self,
        messages: Receiver<Message<N>>,
        started: Sender<Result<(), RuntimeError>>,
    ) {
        let _ = started.send(self.startup());
        while !self.done {
            // `shared` holds a sender, so this does not disconnect.
            match messages.recv() {
                Ok(Message::Job(job)) => job(&mut self),
                Ok(Message::Failed) => self.finish(),
                Err(_) => break,
            }
        }
        // Serve what was sent before the end, then drop `messages`, which makes
        // the later requests fail fast.
        for message in messages.try_iter() {
            if let Message::Job(job) = message {
                job(&mut self);
            }
        }
    }

    fn startup(&mut self) -> Result<(), RuntimeError> {
        let s = Arc::clone(&self.shared);
        let mut c = s.lock();
        while !c.life.failed() && !c.life.all_started() {
            c = s.wait(c);
        }
        if c.life.failed() {
            drop(c);
            return Err(self.end());
        }
        // The timer was frozen at T before the nodes started, so `start` (and
        // restoring) took no time on it. Resume the time before letting the
        // nodes go, as after a pause, so that the first Δt is not 0.
        c.startup = None;
        c.life.go();
        s.wake_all(&mut c);
        Ok(())
    }

    /// Ends the system if a failure was recorded, and fails if it has ended.
    fn ensure_running(&mut self) -> Result<(), RuntimeError> {
        if self.done || self.shared.lock().life.failed() {
            return Err(self.end());
        }
        Ok(())
    }

    pub(super) fn step(&mut self, order: Vec<usize>) -> Result<(), RuntimeError> {
        self.ensure_running()?;
        let s = Arc::clone(&self.shared);
        for i in order {
            let mut c = s.lock();
            if !c.life.failed() {
                c.life.grant(i);
                s.wake(&mut c, i);
            }
            while c.life.granted() && !c.life.failed() {
                c = s.wait(c);
            }
            if c.life.failed() {
                drop(c);
                return Err(self.end());
            }
        }
        Ok(())
    }

    pub(super) fn capture<R>(
        &mut self,
        build: impl FnOnce(Vec<N::State>) -> R,
        timeout: Duration,
    ) -> Result<R, RuntimeError> {
        if self.done {
            return self.shared.capture_ended(build);
        }
        self.ensure_running()?;
        let s = Arc::clone(&self.shared);
        let mut c = s.lock();
        // Raise the flag, then freeze the time right away instead of after the
        // nodes stop: otherwise the wait between a node's last update and the
        // freeze would enter that node's next Δt.
        s.set_command(&mut c, Command::Pause);
        let epoch = c.life.epoch();
        c.capture = Some(s.clock.hold());
        let deadline = Instant::now().checked_add(timeout);
        let n = s.nodes.len();
        while !c.life.failed() && !(0..n).all(|i| c.life.paused(i, epoch)) {
            let remaining = remaining(deadline);
            if remaining.is_zero() {
                let nodes = (0..n)
                    .filter(|&i| !c.life.paused(i, epoch))
                    .map(|i| s.nodes[i].name.clone())
                    .collect();
                // The nodes that did pause get `resume`.
                resume_after_pause(&s, &mut c);
                return Err(RuntimeError::PauseTimeout { nodes });
            }
            c = s.wait_timeout(c, remaining);
        }
        let failed = c.life.failed();
        drop(c);
        if failed {
            return Err(self.end());
        }
        // Every node is parked without holding its lock, and the time stands
        // at T: the states are consistent at T.
        let states = (0..n).map(|i| s.capture(i));
        let Ok(states) = states.collect::<Result<Vec<_>, ()>>() else {
            return Err(self.end());
        };
        let result = build_states(states, build);
        // Resume even if `build` panicked, so the system is not left paused.
        resume_after_pause(&s, &mut s.lock());
        result
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
        while !c.life.ready_to_stop() {
            c = s.wait(c);
        }
        c.life.allow_stop();
        s.wake_all(&mut c);
        drop(c);
        for thread in self.threads.drain(..) {
            // A runtime thread does not panic: hook panics are caught.
            let _ = thread.join();
        }
        s.mark_finished(&mut s.lock());
        self.done = true;
    }

    /// Ends the system (if not yet ended) and returns how it ended.
    fn end(&mut self) -> RuntimeError {
        self.finish();
        finished_error(&self.shared.failure)
    }
}

/// Ends a pause: resume the time first, then lower the flag, so that the first
/// Δt after the pause is not 0 and no wait enters it.
fn resume_after_pause<N: NodeHooks>(s: &Shared<N>, c: &mut Control) {
    c.capture = None;
    // A no-op if an exception already turned the command to `Stop`.
    s.set_command(c, Command::Run);
}
