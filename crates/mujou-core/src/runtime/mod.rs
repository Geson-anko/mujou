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
//! This module holds the host's side (the [`Runtime`] handle and its
//! [`Reply`]) and the state shared by every thread of a runtime, with the
//! operations on it. The rest is in submodules: `lifecycle` is the state
//! machine of a node, `sched` the threads that run the nodes, `control` the
//! control thread, and `error` what the runtime reports.

mod control;
mod error;
mod lifecycle;
mod sched;

use std::cell::Cell;
use std::num::{NonZeroU64, NonZeroUsize};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::time::{PauseHold, RunClock, remaining};

pub use error::{Hook, HookError, NodeFailure, RuntimeError};

use control::{Controller, Message};
use error::{Panicked, finished_error, panic_message, stopped};
use lifecycle::{Command, Lifecycle, Next, Work};
use sched::{Lane, Sched, Worker};

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
/// hook's [`Hook`]. The hooks with a default do nothing.
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

pub struct RunConfig<N> {
    pub nodes: Vec<NodeSpec<N>>,
    /// Run the nodes only when [`Runtime::step`] allows them, one at a time.
    pub manual: bool,
    /// The number of worker threads for the nodes in the pool. Defaults to the
    /// available parallelism; never more than the nodes in the pool, so none
    /// if every node has a dedicated thread.
    pub workers: Option<NonZeroUsize>,
}

thread_local! {
    /// The runtime (the address of its shared state) whose thread this is, or 0.
    static RUNTIME_OF_THREAD: Cell<usize> = const { Cell::new(0) };
}

/// The state shared by the control thread and the nodes, under one lock.
struct Control {
    life: Lifecycle,
    sched: Sched,
    /// Freezes the time from before the nodes start until all of them are
    /// ready to update.
    startup: Option<PauseHold>,
    /// Freezes the time of a capture, from the pause request until the state
    /// is built.
    capture: Option<PauseHold>,
    finished: bool,
}

/// A node as the threads of its runtime hold it.
struct Member<N> {
    name: String,
    period_ns: u64,
    /// The start of the node's next Δt. Written only by the thread that runs
    /// the node, after an update.
    last_update_ns: AtomicU64,
    hooks: Mutex<N>,
}

struct Shared<N: NodeHooks> {
    clock: Arc<RunClock>,
    nodes: Vec<Member<N>>,
    /// The signals of each lane of [`Control::sched`].
    lanes: Vec<Lane>,
    /// The number of workers of the pool.
    workers: usize,
    control: Mutex<Control>,
    /// Notified on a change the control thread waits for
    /// ([`Lifecycle::take_changed`], a failure) and at the end.
    changed: Condvar,
    /// [`Lifecycle::word`], stored under the control lock. A node running
    /// freely reads it without taking the lock, and while it waits on the
    /// timer (which checks the interrupt condition under the timer's lock, and
    /// the control lock is taken before the timer's elsewhere).
    word: AtomicU64,
    failure: Arc<OnceLock<NodeFailure>>,
    to_controller: Sender<Message<N>>,
}

impl<N: NodeHooks> Shared<N> {
    /// The state of a runtime of `nodes` in `lanes` (see [`lanes`]), with
    /// `workers` workers in the pool, and the receiving end of its control
    /// thread's messages.
    fn new(
        clock: Arc<RunClock>,
        nodes: Vec<NodeSpec<N>>,
        lanes: &[usize],
        workers: usize,
        manual: bool,
    ) -> (Self, Receiver<Message<N>>) {
        let n_lanes = lanes.iter().max().map_or(1, |&last| last + 1);
        let life = Lifecycle::new(nodes.len(), manual);
        let (to_controller, messages) = mpsc::channel();
        let members = (nodes.into_iter())
            .map(|spec| Member {
                name: spec.name,
                period_ns: spec.period_ns.get(),
                // Set once the time is frozen for the start.
                last_update_ns: AtomicU64::new(0),
                hooks: Mutex::new(spec.hooks),
            })
            .collect();
        let shared = Self {
            clock,
            nodes: members,
            lanes: (0..n_lanes).map(|_| Lane::default()).collect(),
            workers,
            word: AtomicU64::new(life.word()),
            control: Mutex::new(Control {
                life,
                sched: Sched::new(lanes, n_lanes),
                startup: None,
                capture: None,
                finished: false,
            }),
            changed: Condvar::new(),
            failure: Arc::new(OnceLock::new()),
            to_controller,
        };
        (shared, messages)
    }

    /// Identifies this runtime for [`RUNTIME_OF_THREAD`].
    fn id(&self) -> usize {
        self as *const Self as usize
    }

    fn on_own_thread(&self) -> bool {
        RUNTIME_OF_THREAD.get() == self.id()
    }

    fn lock(&self) -> MutexGuard<'_, Control> {
        // Hooks run outside this lock, so a poisoned state is still consistent.
        self.control.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, Control>) -> MutexGuard<'a, Control> {
        self.changed.wait(guard).unwrap_or_else(|e| e.into_inner())
    }

    fn wait_timeout<'a>(
        &self,
        guard: MutexGuard<'a, Control>,
        timeout: Duration,
    ) -> MutexGuard<'a, Control> {
        self.changed
            .wait_timeout(guard, timeout)
            .unwrap_or_else(|e| e.into_inner())
            .0
    }

    /// Changes the command and wakes every node, including one waiting on the
    /// timer for its next update. Returns whether it changed. The stop ends
    /// the clock.
    fn set_command(&self, c: &mut Control, command: Command) -> bool {
        if !c.life.set_command(command) {
            return false;
        }
        if command == Command::Stop {
            self.clock.end();
        }
        self.word.store(c.life.word(), Ordering::SeqCst);
        self.clock.interrupt_waiters();
        self.wake_all(c);
        true
    }

    /// Re-evaluates every node after a change of the control state, and wakes
    /// the timekeepers once if any node needs them.
    fn wake_all(&self, c: &mut Control) {
        let mut look = false;
        for i in 0..self.nodes.len() {
            look |= self.schedule(i, c);
        }
        if look {
            for lane in 0..self.lanes.len() {
                self.interrupt_timekeeper(lane, c);
            }
        }
    }

    /// Re-evaluates node `i` after a change for it.
    fn wake(&self, c: &mut Control, i: usize) {
        if self.schedule(i, c) {
            self.interrupt_timekeeper(c.sched.lane(i), c);
        }
    }

    /// Puts node `i` where it waits, unless a worker has it or will decide
    /// when it takes it, and wakes a worker of its lane if it may have work
    /// now. Returns whether the timekeeper must look again.
    ///
    /// Only the control side wakes anyone here. A worker that puts a node back
    /// looks at the queues itself right after, and wakes another worker only
    /// when it leaves work behind ([`Worker::run`]): waking one for the node it
    /// is about to take again would make the workers chase each other. Waking
    /// the timekeeper wakes every waiter on the timer (the other runs'
    /// timekeepers, the nodes waiting alone, the host), so a node of period
    /// 1 ns putting itself back after each update must not either.
    fn schedule(&self, i: usize, c: &mut Control) -> bool {
        if c.sched.busy(i) {
            return false;
        }
        let next = self.advance(i, c);
        let placed = c.sched.place(i, next, false);
        if placed.notify {
            self.lanes[placed.lane].changed.notify_one();
        }
        placed.look
    }

    /// What node `i` does next ([`Lifecycle::advance`]).
    fn advance(&self, i: usize, c: &mut Control) -> Next {
        let node = &self.nodes[i];
        let last_ns = node.last_update_ns.load(Ordering::SeqCst);
        // Under the control lock, the freeze cannot come in between.
        let next = c
            .life
            .advance(i, last_ns, node.period_ns, || self.clock.now_ns());
        self.notify_control(c);
        next
    }

    /// Records that `work` for node `i` returned ([`Lifecycle::record`]).
    fn record(&self, i: usize, work: Work, ok: bool, c: &mut Control) {
        let node = &self.nodes[i];
        if let (Work::Update { now_ns }, true) = (work, ok) {
            node.last_update_ns.store(now_ns, Ordering::SeqCst);
        }
        if matches!(work, Work::Stop) && c.sched.finish(i) {
            // Let the idle workers of the lane end too.
            self.lanes[c.sched.lane(i)].changed.notify_all();
        }
        c.life.record(i, work, ok, node.period_ns);
        self.notify_control(c);
    }

    /// Wakes the timekeeper of `lane`, if any, so that it looks at the queues
    /// again.
    fn interrupt_timekeeper(&self, lane: usize, c: &Control) {
        if c.sched.has_timekeeper(lane) {
            self.lanes[lane].epoch.fetch_add(1, Ordering::SeqCst);
            self.clock.interrupt_waiters();
        }
    }

    /// Notifies the control thread of a change it waits for.
    fn notify_control(&self, c: &mut Control) {
        if c.life.take_changed() {
            self.changed.notify_all();
        }
    }

    /// The Δt of an update of node `i` at `now_ns`.
    fn dt(&self, i: usize, now_ns: u64) -> f64 {
        let last_ns = self.nodes[i].last_update_ns.load(Ordering::SeqCst);
        lifecycle::dt(now_ns, last_ns)
    }

    /// Calls a hook on the thread that runs node `i`, with the node locked,
    /// turning a panic into an error.
    fn invoke<T>(
        &self,
        i: usize,
        hook: Hook,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, HookError> {
        let hooks = &self.nodes[i].hooks;
        let mut node = match hooks.try_lock() {
            Ok(node) => node,
            // Hooks run outside the runtime's locks, so the node is consistent.
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => {
                // The scheduling never lets two threads run a node at once;
                // only the captures after the end may meet here.
                debug_assert!(self.lock().finished, "node #{i} is already running");
                hooks.lock().unwrap_or_else(|e| e.into_inner())
            }
        };
        guarded(&mut *node, hook, f)
    }

    /// Calls a hook whose error ends the system.
    fn call<T>(
        &self,
        i: usize,
        hook: Hook,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, ()> {
        self.invoke(i, hook, f).map_err(|e| self.fail(i, e))
    }

    /// Calls `capture` on the control thread while the nodes are paused.
    fn capture(&self, i: usize) -> Result<N::State, ()> {
        self.call(i, Hook::Capture, N::capture)
    }

    /// Takes every node's state after a normal end, on the calling thread,
    /// and builds the result. A failure of `capture` is reported to the
    /// caller only: the system has already ended.
    fn capture_ended<R>(&self, build: impl FnOnce(Vec<N::State>) -> R) -> Result<R, RuntimeError> {
        stopped(&self.failure)?;
        let states = (0..self.nodes.len()).map(|i| {
            self.invoke(i, Hook::Capture, N::capture)
                .map_err(|e| RuntimeError::CaptureFailed(self.failure_of(i, e)))
        });
        build_states(states.collect::<Result<_, _>>()?, build)
    }

    fn failure_of(&self, i: usize, e: HookError) -> NodeFailure {
        NodeFailure {
            name: self.nodes[i].name.clone(),
            hook: e.hook,
            error: e.source,
        }
    }

    /// Records the failure of node `i` if it is the first and asks everything
    /// to stop, right where the error was caught: the nodes see the stop at
    /// their next loop boundary without the control thread having to poll.
    fn fail(&self, i: usize, e: HookError) {
        // Only the first failure is reported; the hooks log every error.
        if self.failure.set(self.failure_of(i, e)).is_err() {
            return;
        }
        let mut c = self.lock();
        c.life.fail();
        // The nodes that already left their loops call `exception` too: wake
        // them even if the system was already stopping.
        if !self.set_command(&mut c, Command::Stop) {
            self.wake_all(&mut c);
        }
        // The control thread may wait for the failure.
        self.changed.notify_all();
        drop(c);
        // The control thread is gone only after the system finished.
        let _ = self.to_controller.send(Message::Failed);
    }

    /// Lets the time run and marks the system finished.
    fn mark_finished(&self, c: &mut Control) {
        // Let the time run again even after an exception, or the host's waits
        // on the timer would never return.
        c.startup = None;
        c.capture = None;
        c.finished = true;
        self.changed.notify_all();
    }
}

/// Calls a hook of `node`, turning a panic into an error of `hook`.
fn guarded<N: NodeHooks, T>(
    node: &mut N,
    hook: Hook,
    f: impl FnOnce(&mut N) -> Result<T, HookError>,
) -> Result<T, HookError> {
    panic::catch_unwind(AssertUnwindSafe(|| f(node))).unwrap_or_else(|payload| {
        let error = HookError {
            hook,
            source: Arc::new(Panicked(panic_message(payload))),
        };
        node.panicked(&error);
        Err(error)
    })
}

/// Builds the result of a capture from `states`, turning a panic of `build`
/// into an error.
fn build_states<S, R>(states: Vec<S>, build: impl FnOnce(Vec<S>) -> R) -> Result<R, RuntimeError> {
    panic::catch_unwind(AssertUnwindSafe(|| build(states))).map_err(|payload| {
        let message = panic_message(payload);
        RuntimeError::Internal(format!("the build of a capture panicked: {message}"))
    })
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
        let workers = (workers.or_else(|| thread::available_parallelism().ok()))
            .map_or(1, NonZeroUsize::get)
            .min(pool);
        let last_update_ns: Vec<_> = nodes.iter().map(|spec| spec.last_update_ns).collect();
        let (shared, messages) = Shared::new(clock, nodes, &lanes, workers, manual);
        let shared = Arc::new(shared);
        shared.lock().startup = Some(shared.clock.hold());
        // Where each first Δt starts: set before any thread exists, so that it
        // is right even if the system ends while starting, and stays right since
        // the time is frozen.
        let now_ns = shared.clock.now_ns();
        for (node, last_ns) in shared.nodes.iter().zip(last_update_ns) {
            (node.last_update_ns).store(last_ns.unwrap_or(now_ns), Ordering::SeqCst);
        }
        let threads = spawn_workers(&shared, &lanes)?;
        let (started_tx, started_rx) = mpsc::channel();
        let control = move |shared: &Arc<Shared<N>>| {
            let controller = Controller {
                shared: Arc::clone(shared),
                threads,
                done: false,
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
        Ok((Self { shared }, reply))
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
        let _ = self.shared.to_controller.send(Message::Job(job));
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
        let _ = self.shared.to_controller.send(Message::Job(job));
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
        stopped(&s.failure).map(|()| true)
    }

    /// The system has ended.
    pub fn finished(&self) -> bool {
        self.shared.lock().finished
    }

    /// Fails with the first failure as soon as it is recorded (possibly before
    /// the system has ended).
    pub fn check(&self) -> Result<(), RuntimeError> {
        stopped(&self.shared.failure)
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
/// ([`RUNTIME_OF_THREAD`]) and runs `body` inside [`NodeHooks::on_thread`].
fn spawn<N: NodeHooks>(
    s: &Arc<Shared<N>>,
    name: String,
    body: impl FnOnce(&Arc<Shared<N>>) + Send + 'static,
) -> std::io::Result<JoinHandle<()>> {
    let shared = Arc::clone(s);
    thread::Builder::new().name(name).spawn(move || {
        N::on_thread(move || {
            RUNTIME_OF_THREAD.set(shared.id());
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

/// Spawns the worker of each dedicated lane and the workers of the pool.
fn spawn_workers<N: NodeHooks>(
    s: &Arc<Shared<N>>,
    lanes: &[usize],
) -> Result<Vec<JoinHandle<()>>, RuntimeError> {
    let mut threads = Vec::with_capacity(s.lanes.len() - 1 + s.workers);
    for (i, &lane) in lanes.iter().enumerate().filter(|&(_, &lane)| lane != 0) {
        let name = format!("mujou-node-{}", s.nodes[i].name);
        match spawn(s, name, move |s| Worker { shared: s, lane }.run()) {
            Ok(thread) => threads.push(thread),
            Err(e) => return Err(abandon(s, "a node", &e)),
        }
    }
    for k in 0..s.workers {
        let name = format!("mujou-worker-{k}");
        match spawn(s, name, |s| Worker { shared: s, lane: 0 }.run()) {
            Ok(thread) => threads.push(thread),
            Err(e) => return Err(abandon(s, "a worker", &e)),
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
