//! Running nodes on worker threads or their own threads, and controlling them
//! from the host.
//!
//! The nodes of a runtime share a pool of worker threads (M nodes on N
//! workers), except those given a dedicated thread, which is a worker of their
//! own. A worker runs one hook of one node at a time and then picks the next
//! work, the updates earliest target first; a node never runs on two threads
//! at once, and its hooks keep their order. Which thread runs a node does not
//! change what the node sees.
//!
//! The nodes never wait for each other; the only barriers are the control
//! operations (pausing to capture the state, stepping by hand, and stopping),
//! which `specs/execution-model.md` allows because the time of the runtime's
//! timer stands still while they hold.
//!
//! All control goes through a single control thread, as in pamiq-core's
//! `ThreadController`. The host submits requests to it and waits for the reply;
//! a node that catches an exception notifies it. Serializing the control on one
//! thread keeps the requests and the end after an exception from interleaving.
//! The timer is frozen through holds of its own, so that neither this runtime's
//! freezes nor those of the host or another runtime undo each other.
//!
//! Every time of the runtime is read on its [`RunClock`]: the nodes' Δt and
//! last updates, and the times of the messages of its slots, which share the
//! clock. The runtime ends the clock when it is asked to stop or catches a
//! failure, so nothing of a run is ever later than its end.
//!
//! The runtime knows nothing about what the hooks compute or what the nodes
//! exchange: [`NodeHooks`] is the whole contract, so any update rule fits.
//!
//! This module holds the host's side: the [`Runtime`] handle, its [`Reply`],
//! and the start, which spawns the threads. The rest is in submodules, each
//! depending only on those after it: `control` is the control thread, `worker`
//! the threads that run the nodes, `shared` the state every thread of a runtime
//! shares and the operations on it, `sched` the queues of the workers,
//! `lifecycle` the state machine of a node, `hooks` what a node gives the
//! runtime, and `error` what the runtime reports.

mod control;
mod error;
mod hooks;
mod lifecycle;
mod sched;
mod shared;
mod worker;

use std::num::NonZeroUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::time::{RunClock, remaining};

pub use error::{Hook, HookError, NodeFailure, RuntimeError};
pub use hooks::{NodeHooks, NodeSpec};

use control::{Controller, Message};
use error::finished_error;
use lifecycle::Command;
use shared::Shared;
use worker::Worker;

pub struct RunConfig<N> {
    pub nodes: Vec<NodeSpec<N>>,
    /// Run the nodes only when [`Runtime::step`] allows them, one at a time.
    pub manual: bool,
    /// The number of worker threads for the nodes in the pool, never more
    /// than the nodes in the pool, so none if every node has a dedicated
    /// thread.
    pub workers: NonZeroUsize,
}

/// The pending reply to a request to the runtime.
pub struct Reply<T> {
    rx: Receiver<Result<T, RuntimeError>>,
    failure: Arc<OnceLock<NodeFailure>>,
}

impl<T> Reply<T> {
    fn ready(result: Result<T, RuntimeError>, failure: Arc<OnceLock<NodeFailure>>) -> Self {
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(result);
        Self { rx, failure }
    }

    /// Blocks until the reply comes.
    pub fn wait(self) -> Result<T, RuntimeError> {
        self.rx
            .recv()
            .unwrap_or_else(|_| Err(finished_error(&self.failure)))
    }

    /// Blocks until the reply comes or `timeout` passes (`None`), so that the
    /// caller can check for signals in between.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Result<T, RuntimeError>> {
        match self.rx.recv_timeout(timeout) {
            Ok(result) => Some(result),
            Err(RecvTimeoutError::Timeout) => None,
            // The control thread ended before serving the request.
            Err(RecvTimeoutError::Disconnected) => Some(Err(finished_error(&self.failure))),
        }
    }
}

/// The host's handle to the running nodes.
///
/// The requests are served by the control thread; they return a [`Reply`] so
/// that the host can wait for it in steps. Dropping the handle asks the system
/// to stop without waiting.
pub struct Runtime<N: NodeHooks> {
    shared: Arc<Shared<N>>,
    to_controller: Sender<Message<N>>,
}

impl<N: NodeHooks> Runtime<N> {
    /// Starts every node, on the worker pool or on a dedicated thread. The
    /// reply comes after every node started and the timer runs again; if a
    /// node failed to start, it comes after the system ended.
    ///
    /// The time is frozen first, so that it does not run while the nodes
    /// start and restore their states.
    ///
    /// The nodes run on `clock`: it measures their Δt, and its timer is the one
    /// frozen. Create the slots and the host's inputs of this run with the same
    /// clock ([`crate::slot::Outlet::new`]).
    pub fn start(
        config: RunConfig<N>,
        clock: Arc<RunClock>,
    ) -> Result<(Self, Reply<()>), RuntimeError> {
        let RunConfig {
            nodes,
            manual,
            workers,
        } = config;
        let lanes = lanes(&nodes);
        let pool = lanes.iter().filter(|&&lane| lane == 0).count();
        let workers = workers.get().min(pool);
        let last_update_ns: Vec<_> = nodes.iter().map(|spec| spec.last_update_ns).collect();
        let shared = Arc::new(Shared::new(clock, nodes, &lanes, workers, manual));
        let (to_controller, messages) = mpsc::channel();
        shared.lock().startup = Some(shared.clock.hold());
        // Where each first Δt starts: set before any thread exists, so that it
        // is right even if the system ends while starting, and stays right since
        // the time is frozen.
        let now_ns = shared.clock.now_ns();
        for (node, last_ns) in shared.nodes.iter().zip(last_update_ns) {
            (node.last_update_ns).store(last_ns.unwrap_or(now_ns), Ordering::SeqCst);
        }
        let threads = spawn_workers(&shared, &lanes, &to_controller)?;
        let (started_tx, started_rx) = mpsc::channel();
        let control = move |shared: &Arc<Shared<N>>| {
            let controller = Controller {
                shared: Arc::clone(shared),
                threads,
            };
            controller.run(messages, started_tx);
        };
        if let Err(e) = spawn(&shared, "mujou-control".to_owned(), control) {
            // The node threads' handles were dropped with `control`.
            return Err(abandon(&shared, "the control", &e));
        }
        let reply = Reply {
            rx: started_rx,
            failure: Arc::clone(&shared.failure),
        };
        let runtime = Self {
            shared,
            to_controller,
        };
        Ok((runtime, reply))
    }

    fn submit<R: Send + 'static>(
        &self,
        job: impl FnOnce(&mut Controller<N>) -> Result<R, RuntimeError> + Send + 'static,
    ) -> Reply<R> {
        let failure = Arc::clone(&self.shared.failure);
        if self.shared.on_own_thread() {
            return Reply::ready(Err(RuntimeError::Reentrant), failure);
        }
        let (tx, rx) = mpsc::channel();
        let job = Box::new(move |controller: &mut Controller<N>| {
            let _ = tx.send(job(controller));
        });
        // If the control thread has ended, the job is dropped with its sender
        // and the reply reports how the system ended.
        let _ = self.to_controller.send(Message::Job(job));
        Reply { rx, failure }
    }

    /// Manual mode: lets the nodes in `order` (all in index order if `None`)
    /// run one update each, one after another. A node whose time did not move
    /// since its last update does not update.
    pub fn step(&self, order: Option<&[usize]>) -> Reply<()> {
        let n = self.shared.nodes.len();
        let order = order.map_or_else(|| (0..n).collect(), <[usize]>::to_vec);
        if !self.shared.lock().life.manual() || order.iter().any(|&i| i >= n) {
            // A step on a free-running runtime would never end.
            let error = RuntimeError::Internal(format!(
                "a step needs the manual mode and indices of the {n} nodes, got {order:?}"
            ));
            return Reply::ready(Err(error), Arc::clone(&self.shared.failure));
        }
        self.submit(move |controller| controller.step(order))
    }

    /// Pauses every node, takes their states and passes them to `build` on the
    /// control thread while paused, then resumes. Gives up after `timeout`
    /// (real time) if some node does not pause.
    ///
    /// After a normal end, takes the states as the nodes left them and builds
    /// on the calling thread instead, at once. A capture that ended the system
    /// meanwhile does the same on the control thread.
    pub fn capture<R: Send + 'static>(
        &self,
        build: impl FnOnce(Vec<N::State>) -> R + Send + 'static,
        timeout: Duration,
    ) -> Reply<R> {
        let s = &self.shared;
        let c = s.lock();
        if c.finished {
            drop(c);
            return Reply::ready(s.capture_ended(build), Arc::clone(&s.failure));
        }
        // Sent under the control lock: the control thread marks the system
        // finished under it, and serves what was sent before it ends.
        self.submit(move |controller| controller.capture(build, timeout))
    }

    /// Asks every node to stop and returns without waiting; [`Runtime::join`]
    /// waits for the end. Does nothing once the system has ended.
    pub fn stop(&self) {
        // Not through `submit`: there is no reply to wait for, so a node's own
        // thread may ask too.
        let job = Box::new(|controller: &mut Controller<N>| controller.finish());
        let _ = self.to_controller.send(Message::Job(job));
    }

    /// Waits until the system has ended (every node thread joined), for at most
    /// `timeout`. Returns whether it has ended; fails if it ended because of an
    /// exception.
    pub fn join(&self, timeout: Option<Duration>) -> Result<bool, RuntimeError> {
        let s = &self.shared;
        let mut c = s.lock();
        if !c.finished && s.on_own_thread() {
            return Err(RuntimeError::Reentrant);
        }
        let deadline = timeout.and_then(|t| Instant::now().checked_add(t));
        while !c.finished {
            let remaining = remaining(deadline);
            if remaining.is_zero() {
                return Ok(false);
            }
            c = s.wait_timeout(c, remaining);
        }
        s.check_failure().map(|()| true)
    }

    /// The system has ended.
    pub fn finished(&self) -> bool {
        self.shared.lock().finished
    }

    /// Fails with the first failure as soon as it is recorded (possibly before
    /// the system has ended).
    pub fn check(&self) -> Result<(), RuntimeError> {
        self.shared.check_failure()
    }

    /// The number of workers of the pool.
    pub fn workers(&self) -> usize {
        self.shared.workers
    }

    /// The time of each node's last update on the runtime's timer (where its
    /// next Δt starts).
    pub fn last_update_ns(&self) -> Vec<u64> {
        let nodes = &self.shared.nodes;
        (nodes.iter())
            .map(|n| n.last_update_ns.load(Ordering::SeqCst))
            .collect()
    }
}

/// Spawns a thread of the runtime of `s`, which marks itself as one
/// (`Shared::own_thread`) and runs `body` inside [`NodeHooks::on_thread`].
fn spawn<N: NodeHooks>(
    s: &Arc<Shared<N>>,
    name: String,
    body: impl FnOnce(&Arc<Shared<N>>) + Send + 'static,
) -> std::io::Result<JoinHandle<()>> {
    let shared = Arc::clone(s);
    thread::Builder::new().name(name).spawn(move || {
        N::on_thread(move || {
            shared.own_thread();
            body(&shared);
        })
    })
}

/// The lane of each node: the pool is lane 0, and each node with a dedicated
/// thread has a lane of its own, from 1 in index order.
fn lanes<N>(nodes: &[NodeSpec<N>]) -> Vec<usize> {
    let mut dedicated = 0;
    (nodes.iter())
        .map(|spec| {
            if spec.dedicated_thread {
                dedicated += 1;
                dedicated
            } else {
                0
            }
        })
        .collect()
}

/// Spawns the worker of each dedicated lane and the workers of the pool,
/// which tell the control thread through `to_controller` when a hook fails.
fn spawn_workers<N: NodeHooks>(
    s: &Arc<Shared<N>>,
    lanes: &[usize],
    to_controller: &Sender<Message<N>>,
) -> Result<Vec<JoinHandle<()>>, RuntimeError> {
    let dedicated = (lanes.iter().enumerate())
        .filter(|&(_, &lane)| lane != 0)
        .map(|(i, &lane)| (format!("mujou-node-{}", s.nodes[i].name), lane, "a node"));
    let pool = (0..s.workers).map(|k| (format!("mujou-worker-{k}"), 0, "a worker"));
    let mut threads = Vec::with_capacity(s.lanes.len() - 1 + s.workers);
    for (name, lane, which) in dedicated.chain(pool) {
        let to_controller = to_controller.clone();
        let body = move |s: &Arc<Shared<N>>| {
            let worker = Worker {
                shared: s,
                lane,
                to_controller,
            };
            worker.run();
        };
        match spawn(s, name, body) {
            Ok(thread) => threads.push(thread),
            Err(e) => return Err(abandon(s, which, &e)),
        }
    }
    Ok(threads)
}

/// Gives up a start whose threads could not all be spawned: lets the time run
/// and lets the threads already spawned end on their own (they call `stop`
/// after `start`, also for the pool nodes if a worker was spawned). Does not
/// wait for them, since the caller may hold a lock they need.
fn abandon<N: NodeHooks>(s: &Shared<N>, which: &str, e: &std::io::Error) -> RuntimeError {
    let mut c = s.lock();
    s.set_command(&mut c, Command::Stop);
    c.life.allow_stop();
    // The workers already spawned run their nodes to their end.
    s.wake_all(&mut c);
    s.mark_finished(&mut c);
    RuntimeError::Internal(format!("failed to spawn {which} thread: {e}"))
}

impl<N: NodeHooks> Drop for Runtime<N> {
    fn drop(&mut self) {
        // Do not wait: the caller may hold a lock the nodes need (the GIL).
        self.stop();
    }
}
