//! Running nodes on their own threads, and controlling them from the host.
//!
//! Each node runs on a dedicated thread that loops over its update hook. The
//! nodes never wait for each other; the only barriers are the control operations
//! (pausing to capture the state, stepping by hand, and stopping), which
//! `specs/execution-model.md` allows because the time of the runtime's timer
//! stands still while they hold.
//!
//! All control goes through a single control thread, as in pamiq-core's
//! `ThreadController`. The host submits requests to it and waits for the reply;
//! a node that catches an exception notifies it. Serializing the control on one
//! thread keeps the requests and the end after an exception from interleaving.
//! The timer is frozen through [`PauseHold`](crate::time::PauseHold)s, so
//! that neither this runtime's freezes nor those of the host or another
//! runtime undo each other.
//!
//! The runtime knows nothing about what the hooks compute or what the nodes
//! exchange: [`NodeHooks`] is the whole contract, so any update rule fits.
//!
//! This module holds the host's side (the [`Runtime`] handle and its
//! [`Reply`]) and the state shared by every thread of a runtime. The rest is in
//! submodules: `node` is the loop of a node's thread, `control` the
//! control thread, `freeze` how the timer is held frozen, and `error` what the
//! runtime reports.

mod control;
mod error;
mod freeze;
mod node;

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::time::{Timer, TimerState};

pub use error::{HookError, NodeFailure, RuntimeError, SharedError};

use control::{Controller, Message};
use error::{Ended, Panicked, panic_message, stopped, unavailable};
use freeze::{Freeze, FreezeReason, Live, lock_live};
use node::{Command, Report, control_word};

/// The lifecycle hooks of a node, called by the runtime.
///
/// Every hook except `state_dict` runs on the node's own thread. `state_dict`
/// runs on the control thread while the node is paused, so it never overlaps
/// with the other hooks. It is not called after an exception.
///
/// A hook that panics is treated like one that returned an error, under the
/// hook's name as the node's author knows it (`on_update` for `update`, ...).
pub trait NodeHooks: Send + 'static {
    type State: Send + 'static;

    /// Runs the whole body of a runtime thread (a node's or the control
    /// thread). The bindings of a language with per-thread state wrap it here,
    /// so that the state lives as long as the thread instead of one hook call.
    /// The body waits a lot, so the wrapper must not hold a lock others need.
    fn on_thread<R: Send>(body: impl FnOnce() -> R + Send) -> R {
        body()
    }

    /// Told that a hook panicked, with the error that will stand for it, so
    /// that the node can log it. The node may be in an inconsistent state.
    fn panicked(&mut self, _error: &HookError) {}

    /// Prepares the node on its thread before the first update, and restores
    /// its state if there is one. The time is frozen meanwhile, so the time this
    /// takes does not enter the first Δt.
    fn start(&mut self) -> Result<(), HookError>;
    /// Advances the node by `dt` seconds of the runtime's timer (always > 0).
    fn update(&mut self, dt: f64) -> Result<(), HookError>;
    fn pause(&mut self) -> Result<(), HookError>;
    fn resume(&mut self) -> Result<(), HookError>;
    /// Called once on every node when the system stops because of an exception,
    /// before `stop`. Its error is ignored.
    fn exception(&mut self) -> Result<(), HookError>;
    /// Called once on every node at the end. Its error is ignored.
    fn stop(&mut self) -> Result<(), HookError>;
    fn state_dict(&mut self) -> Result<Self::State, HookError>;
}

#[derive(Default)]
pub struct RuntimeConfig {
    /// Run the nodes only when [`Runtime::step`] allows them, one at a time.
    pub manual: bool,
    /// Where each node's first Δt starts. Defaults to the time at the start.
    pub last_update_ns: Option<Vec<u64>>,
    /// Each node's period on the timer, in ns (> 0): a free-running node
    /// updates at targets a period apart (see `node_loop`), so once every
    /// period on average while its update and its wake-up fit in the period.
    /// `None` (for all nodes or one) updates as soon as any time has passed.
    /// The manual mode ignores it.
    pub period_ns: Option<Vec<Option<u64>>>,
    /// The timer state to continue from.
    pub timer_state: Option<TimerState>,
}

thread_local! {
    /// The runtime (the address of its shared state) whose thread this is, or 0.
    static RUNTIME_OF_THREAD: Cell<usize> = const { Cell::new(0) };
}

/// The state shared by the control thread and the nodes, under one lock.
struct Control {
    command: Command,
    /// Bumped on every change of `command`, so that a paused node can tell the
    /// end of its pause from a new pause request that came before it woke up.
    epoch: u64,
    /// The epoch of the last change to `Run`.
    run_epoch: u64,
    failed: bool,
    /// The nodes may enter their loops (set after every node started).
    go: bool,
    /// Manual mode: the node allowed to run one step. The node clears it.
    grant: Option<usize>,
    nodes: Vec<Report>,
    /// Every node may call `stop` (set once every node is ready to stop).
    stop_allowed: bool,
    freeze: Freeze,
    /// Counts this runtime as live on its timer until it finishes.
    live: Option<Live>,
    finished: bool,
}

struct Shared<N: NodeHooks> {
    timer: Arc<Timer>,
    names: Vec<String>,
    manual: bool,
    control: Mutex<Control>,
    /// Notified on every change of `control`, in both directions.
    changed: Condvar,
    /// `control_word(Control::epoch, Control::command)`, stored under the
    /// control lock. A free-running node reads it at its loop boundary without
    /// taking the lock, and while it waits on the timer (which checks the
    /// interrupt condition under the timer's lock, and the control lock is taken
    /// before the timer's elsewhere).
    word: AtomicU64,
    last_update_ns: Vec<AtomicU64>,
    /// Each node's period in ns when running freely (1 without a period: the
    /// next update as soon as Δt > 0).
    period_ns: Vec<u64>,
    failure: Arc<OnceLock<NodeFailure>>,
    hooks: Vec<Mutex<N>>,
    to_controller: Sender<Message<N>>,
}

impl<N: NodeHooks> Shared<N> {
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
    /// timer for its Δt to become positive.
    fn set_command(&self, c: &mut Control, command: Command) {
        if c.command == Command::Stop || c.command == command {
            return;
        }
        c.command = command;
        c.epoch += 1;
        if command == Command::Run {
            c.run_epoch = c.epoch;
        }
        self.word
            .store(control_word(c.epoch, command), Ordering::SeqCst);
        self.changed.notify_all();
        self.timer.interrupt_waiters();
    }

    /// Calls a hook with the node locked, turning a panic into an error.
    fn invoke<T>(
        &self,
        i: usize,
        hook: &'static str,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, HookError> {
        let mut node = self.hooks[i].lock().unwrap_or_else(|e| e.into_inner());
        panic::catch_unwind(AssertUnwindSafe(|| f(&mut node))).unwrap_or_else(|payload| {
            let error = HookError {
                hook,
                source: Arc::new(Panicked(panic_message(payload))),
            };
            node.panicked(&error);
            Err(error)
        })
    }

    /// Calls a hook whose error ends the system.
    fn call<T>(
        &self,
        i: usize,
        hook: &'static str,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, ()> {
        self.invoke(i, hook, f).map_err(|e| {
            self.fail(NodeFailure {
                node: i,
                hook: e.hook,
                error: e.source,
            })
        })
    }

    /// Records the first failure and asks everything to stop, right where the
    /// error was caught: the nodes see the stop at their next loop boundary
    /// without the control thread having to poll.
    fn fail(&self, failure: NodeFailure) {
        // Only the first failure is reported; the hooks log every error.
        if self.failure.set(failure).is_err() {
            return;
        }
        let mut c = self.lock();
        c.failed = true;
        self.set_command(&mut c, Command::Stop);
        // Wake the nodes that already left their loops, so they call
        // `exception` too (`set_command` does not notify if already stopping).
        self.changed.notify_all();
        drop(c);
        // The control thread is gone only after the system finished.
        let _ = self.to_controller.send(Message::Failed);
    }

    /// Freezes the timer for the startup, loads `timer_state` into it and
    /// counts this runtime as live on it. Called by the host before any thread
    /// of this runtime exists, so the host may touch the timer.
    fn freeze_for_startup(&self, timer_state: Option<TimerState>) -> Result<(), RuntimeError> {
        // Checking for other live runtimes, loading the time and counting this
        // runtime happen under one lock, so two starts cannot both pass.
        let mut live = lock_live();
        let timer_id = Arc::as_ptr(&self.timer) as usize;
        if timer_state.is_some() && live.contains_key(&timer_id) {
            // The time is shared by every runtime on the timer: moving it would
            // make the running ones jump (or stall) by the difference in their
            // next Δt.
            return Err(RuntimeError::InvalidArgument(
                "cannot set the timer back to the time of the state while another run is \
                 running on it; give this run a timer of its own"
                    .to_owned(),
            ));
        }
        self.lock().freeze.hold(FreezeReason::Startup, &self.timer);
        if let Some(state) = timer_state {
            if let Err(e) = self.timer.load_state(state) {
                self.lock().freeze.release_all();
                return Err(RuntimeError::InvalidArgument(e.to_string()));
            }
        }
        *live.entry(timer_id).or_insert(0) += 1;
        drop(live);
        self.lock().live = Some(Live(Arc::clone(&self.timer)));
        Ok(())
    }

    /// Lets the time run, uncounts the runtime on its timer and marks it
    /// finished. Takes the control lock held by the caller.
    fn mark_finished(&self, mut c: MutexGuard<'_, Control>) {
        // Let the time run again even after an exception, or the host's waits
        // on the timer would never return.
        c.freeze.release_all();
        // Uncount before `finished`, so that a start after `join` can restore
        // the time; and outside the control lock, since `start` takes the
        // count's lock before a control lock.
        let live = c.live.take();
        drop(c);
        drop(live);
        let mut c = self.lock();
        c.finished = true;
        self.changed.notify_all();
    }
}

/// What is left of a wait until `deadline` (no end if `None`).
fn remaining(deadline: Option<Instant>) -> Duration {
    deadline.map_or(Duration::MAX, |d| {
        d.saturating_duration_since(Instant::now())
    })
}

/// The pending reply to a request to the runtime.
pub struct Reply<T> {
    rx: Receiver<Result<T, RuntimeError>>,
    failure: Arc<OnceLock<NodeFailure>>,
    ended: Ended<T>,
}

impl<T> Reply<T> {
    fn ready(result: Result<T, RuntimeError>, failure: Arc<OnceLock<NodeFailure>>) -> Self {
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(result);
        Self {
            rx,
            failure,
            ended: unavailable,
        }
    }

    /// Blocks until the reply comes.
    pub fn wait(self) -> Result<T, RuntimeError> {
        self.rx
            .recv()
            .unwrap_or_else(|_| (self.ended)(&self.failure))
    }

    /// Blocks until the reply comes or `timeout` passes (`None`), so that the
    /// caller can check for signals in between.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<Result<T, RuntimeError>> {
        match self.rx.recv_timeout(timeout) {
            Ok(result) => Some(result),
            Err(RecvTimeoutError::Timeout) => None,
            // The control thread ended before serving the request.
            Err(RecvTimeoutError::Disconnected) => Some((self.ended)(&self.failure)),
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
    /// Starts every node on its own thread. The reply comes after every node
    /// started and the timer runs again; if a node failed to start, it comes
    /// after the system ended.
    ///
    /// The timer is frozen first and then loaded from `config.timer_state`, so
    /// that the time is restored before the nodes restore their states and
    /// does not run while they start.
    ///
    /// The nodes run on `timer`: it measures their Δt and is the one frozen.
    /// Create the slots and the host's inputs of this run with the same timer
    /// ([`crate::slot::Outlet::new`]).
    pub fn start(
        nodes: Vec<N>,
        names: Vec<String>,
        timer: Arc<Timer>,
        config: RuntimeConfig,
    ) -> Result<(Self, Reply<()>), RuntimeError> {
        let n = nodes.len();
        let period_ns = check_start(n, names.len(), &config)?;
        let (to_controller, messages) = mpsc::channel();
        let shared = Arc::new(Shared {
            timer,
            names,
            manual: config.manual,
            control: Mutex::new(Control {
                command: Command::Run,
                epoch: 0,
                run_epoch: 0,
                failed: false,
                go: false,
                grant: None,
                nodes: (0..n).map(|_| Report::default()).collect(),
                stop_allowed: false,
                freeze: Freeze::default(),
                live: None,
                finished: false,
            }),
            changed: Condvar::new(),
            word: AtomicU64::new(control_word(0, Command::Run)),
            last_update_ns: (0..n).map(|_| AtomicU64::new(0)).collect(),
            period_ns,
            failure: Arc::new(OnceLock::new()),
            hooks: nodes.into_iter().map(Mutex::new).collect(),
            to_controller,
        });
        shared.freeze_for_startup(config.timer_state)?;
        // Where each first Δt starts: set before any thread exists, so that it
        // is right even if the system ends while starting, and stays right since
        // the time is frozen.
        let now_ns = shared.timer.time_ns();
        for (i, last) in shared.last_update_ns.iter().enumerate() {
            let ns = config.last_update_ns.as_ref().map_or(now_ns, |v| v[i]);
            last.store(ns, Ordering::SeqCst);
        }

        let mut threads = Vec::with_capacity(n);
        for i in 0..n {
            let node_shared = Arc::clone(&shared);
            let spawned = thread::Builder::new()
                .name(format!("mujou-node-{}", shared.names[i]))
                .spawn(move || N::on_thread(move || node_shared.run_node(i)));
            match spawned {
                Ok(thread) => threads.push(thread),
                Err(e) => return Err(abandon(&shared, "a node", &e)),
            }
        }
        let controller = Controller {
            shared: Arc::clone(&shared),
            threads,
            done: false,
        };
        let (started_tx, started_rx) = mpsc::channel();
        let spawned = thread::Builder::new()
            .name("mujou-control".to_owned())
            .spawn(move || N::on_thread(move || controller.run(messages, started_tx)));
        if let Err(e) = spawned {
            // The controller (with the node threads' handles) was dropped.
            return Err(abandon(&shared, "the control", &e));
        }
        let reply = Reply {
            rx: started_rx,
            failure: Arc::clone(&shared.failure),
            ended: unavailable,
        };
        Ok((Self { shared }, reply))
    }

    fn submit<R: Send + 'static>(
        &self,
        ended: Ended<R>,
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
        Reply { rx, failure, ended }
    }

    /// Manual mode: lets the nodes in `order` (all in index order if `None`)
    /// run one update each, one after another. A node whose time did not move
    /// since its last update does not update.
    pub fn step(&self, order: Option<&[usize]>) -> Reply<()> {
        let failure = Arc::clone(&self.shared.failure);
        if !self.shared.manual {
            return Reply::ready(Err(RuntimeError::NotManual), failure);
        }
        let n = self.shared.hooks.len();
        let order = order.map_or_else(|| (0..n).collect(), <[usize]>::to_vec);
        if let Some(&i) = order.iter().find(|&&i| i >= n) {
            let message = format!("node index {i} is out of range for {n} nodes");
            return Reply::ready(Err(RuntimeError::InvalidArgument(message)), failure);
        }
        self.submit(unavailable, move |controller| controller.step(order))
    }

    /// Pauses every node, takes their states and passes them to `build` on the
    /// control thread while paused, then resumes. Gives up after `timeout`
    /// (real time) if some node does not pause.
    pub fn capture<R: Send + 'static>(
        &self,
        build: impl FnOnce(Vec<N::State>) -> R + Send + 'static,
        timeout: Duration,
    ) -> Reply<R> {
        self.submit(unavailable, move |controller| {
            controller.capture(build, timeout)
        })
    }

    /// Stops every node and ends the system. Succeeds if the system has
    /// already stopped normally; fails with the failure if it ended because of
    /// one.
    pub fn shutdown(&self) -> Reply<()> {
        self.submit(stopped, |controller| controller.shutdown())
    }

    /// Asks every node to stop and returns without waiting; [`Runtime::join`]
    /// waits for the end. Does nothing once the system has ended.
    pub fn request_stop(&self) {
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

    /// The first failure, as soon as it is recorded (possibly before the system
    /// has ended).
    pub fn failure(&self) -> Option<&NodeFailure> {
        self.shared.failure.get()
    }

    pub fn is_finished(&self) -> bool {
        self.shared.lock().finished
    }

    /// The time of each node's last update on the runtime's timer (where its
    /// next Δt starts).
    pub fn last_update_ns(&self) -> Vec<u64> {
        let last = &self.shared.last_update_ns;
        last.iter().map(|ns| ns.load(Ordering::SeqCst)).collect()
    }
}

/// Checks the arguments of [`Runtime::start`] for `n` nodes and returns each
/// node's period in ns when running freely (1 without a period).
fn check_start(n: usize, names: usize, config: &RuntimeConfig) -> Result<Vec<u64>, RuntimeError> {
    if names != n {
        return Err(RuntimeError::InvalidArgument(format!(
            "got {n} nodes but {names} names"
        )));
    }
    if let Some(last) = &config.last_update_ns {
        if last.len() != n {
            return Err(RuntimeError::InvalidArgument(format!(
                "got {n} nodes but {} last update times",
                last.len()
            )));
        }
    }
    match &config.period_ns {
        None => Ok(vec![1; n]),
        Some(periods) if periods.len() != n => Err(RuntimeError::InvalidArgument(format!(
            "got {n} nodes but {} periods",
            periods.len()
        ))),
        Some(periods) => periods
            .iter()
            .map(|period| match *period {
                None => Ok(1),
                Some(0) => Err(RuntimeError::InvalidArgument(
                    "a period must be > 0 ns".to_owned(),
                )),
                Some(ns) => Ok(ns),
            })
            .collect(),
    }
}

/// Gives up a start whose threads could not all be spawned: lets the time run
/// and lets the threads already spawned end on their own (they call `on_stop`
/// after `on_start`). Does not wait for them, since the caller may hold a lock
/// they need.
fn abandon<N: NodeHooks>(s: &Shared<N>, which: &str, e: &std::io::Error) -> RuntimeError {
    let mut c = s.lock();
    s.set_command(&mut c, Command::Stop);
    c.stop_allowed = true;
    s.mark_finished(c);
    RuntimeError::Internal(format!("failed to spawn {which} thread: {e}"))
}

impl<N: NodeHooks> Drop for Runtime<N> {
    fn drop(&mut self) {
        // Do not wait: the caller may hold a lock the nodes need (the GIL).
        self.request_stop();
    }
}
