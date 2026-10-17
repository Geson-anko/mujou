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
//! The timer is frozen through [`PauseHold`]s, so that neither this runtime's
//! freezes nor those of the host or another runtime undo each other.
//!
//! The runtime knows nothing about what the hooks compute or what the nodes
//! exchange: [`NodeHooks`] is the whole contract, so any update rule fits.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::time::{PauseHold, Timer, TimerState};

/// An error raised by a hook, shared because it is reported to several places.
pub type SharedError = Arc<dyn Error + Send + Sync>;

/// A hook failed. `hook` names the hook as the node's author knows it.
#[derive(Debug, Clone)]
pub struct HookError {
    pub hook: &'static str,
    pub source: SharedError,
}

/// The lifecycle hooks of a node, called by the runtime.
///
/// Every hook except `state_dict` runs on the node's own thread. `state_dict`
/// runs on the control thread while the node is paused (or has stopped after an
/// exception), so it never overlaps with the other hooks.
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

/// The first hook error, which ended the system.
#[derive(Debug, Clone)]
pub struct NodeFailure {
    pub node: usize,
    pub hook: &'static str,
    pub error: SharedError,
}

pub struct RuntimeConfig {
    /// Run the nodes only when [`Runtime::step`] allows them, one at a time.
    pub manual: bool,
    /// How long a capture waits for every node to pause.
    pub pause_timeout: Duration,
    /// Where each node's first Δt starts. Defaults to the time at the start.
    pub last_update_ns: Option<Vec<u64>>,
    /// Each node's period on the timer, in ns (> 0): a free-running node
    /// updates at targets a period apart (see `node_loop`), so once every
    /// period on average while its update and its wake-up fit in the period. `None` (for all nodes or one) updates as soon as any time
    /// has passed. The manual mode ignores it.
    pub period_ns: Option<Vec<Option<u64>>>,
    /// The timer state to continue from.
    pub timer_state: Option<TimerState>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            manual: false,
            pause_timeout: Duration::from_secs(10),
            last_update_ns: None,
            period_ns: None,
            timer_state: None,
        }
    }
}

/// Takes the state when the system stops because of an exception. The map
/// lacks the nodes whose `state_dict` failed.
pub type ExceptionCapture<S> = Box<dyn FnOnce(BTreeMap<usize, S>, &NodeFailure) + Send>;

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
struct Panicked(String);

impl fmt::Display for Panicked {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "panicked: {}", self.0)
    }
}

impl Error for Panicked {}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map_or_else(|| "unknown payload".to_owned(), |s| (*s).to_owned()),
    }
}

thread_local! {
    /// The runtime (the address of its shared state) whose thread this is, or 0.
    static RUNTIME_OF_THREAD: Cell<usize> = const { Cell::new(0) };
}

/// The instruction from the control thread to the nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Run = 0,
    Pause = 1,
    /// Final: nothing changes the command once it is `Stop`.
    Stop = 2,
}

/// The command and its epoch in one word, so that a node can read both with
/// one atomic load: `epoch << 2 | command`.
fn control_word(epoch: u64, command: Command) -> u64 {
    epoch << 2 | command as u64
}

fn is_run(word: u64) -> bool {
    word & 3 == Command::Run as u64
}

/// Why the runtime holds the timer frozen. The runtime keeps one
/// [`PauseHold`] while any reason is left, so ending one (a capture) cannot
/// undo another (an exception).
#[derive(Debug, Clone, Copy)]
enum FreezeReason {
    /// From before the nodes start until all of them are ready to update.
    Startup = 1,
    /// A capture: from the pause request until the state is built.
    Capture = 2,
    /// An exception, when the state is captured on exceptions.
    Failure = 4,
}

#[derive(Default)]
struct Freeze {
    reasons: u8,
    hold: Option<PauseHold>,
}

impl Freeze {
    fn hold(&mut self, reason: FreezeReason, timer: &Arc<Timer>) {
        if self.hold.is_none() {
            self.hold = Some(timer.hold());
        }
        self.reasons |= reason as u8;
    }

    fn release(&mut self, reason: FreezeReason) {
        self.reasons &= !(reason as u8);
        if self.reasons == 0 {
            self.hold = None;
        }
    }

    fn release_all(&mut self) {
        self.reasons = 0;
        self.hold = None;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ExceptionHook {
    #[default]
    NotCalled,
    Calling,
    Done,
}

/// What a node reports to the control thread.
#[derive(Debug, Default)]
struct Report {
    /// `start` returned (successfully or not).
    started: bool,
    /// The node called `pause` for the pause request of this epoch and waits.
    paused_at: Option<u64>,
    /// The node left its loop and will call no hook but `exception` and `stop`.
    exited: bool,
    exception: ExceptionHook,
}

impl Report {
    /// Ready for `stop`: out of the loop, and through `exception` if the system
    /// stops because of one.
    fn ready_to_stop(&self, failed: bool) -> bool {
        self.exited && (!failed || self.exception == ExceptionHook::Done)
    }
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
    /// Every node may call `stop` (set after the exception capture, if any).
    stop_allowed: bool,
    freeze: Freeze,
    /// Counts this runtime as live on its timer until it finishes.
    live: Option<Live>,
    finished: bool,
}

/// A request from the host, run on the control thread.
type Job<N> = Box<dyn FnOnce(&mut Controller<N>) + Send>;

enum Message<N: NodeHooks> {
    Job(Job<N>),
    /// A node recorded the first failure; the control thread ends the system.
    Failed,
}

struct Shared<N: NodeHooks> {
    timer: Arc<Timer>,
    names: Vec<String>,
    manual: bool,
    capture_on_failure: bool,
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
        // Freeze right after raising the flag, as a pause does, so that the
        // state captured on the exception is consistent at one time.
        if self.capture_on_failure {
            c.freeze.hold(FreezeReason::Failure, &self.timer);
        }
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
                "cannot restore the time while another runtime is running on the same timer"
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

    /// The thread of node `i`.
    fn run_node(&self, i: usize) {
        RUNTIME_OF_THREAD.set(self.id());
        let started = self.call(i, "on_start", |n| n.start()).is_ok();
        let mut c = self.lock();
        c.nodes[i].started = true;
        self.changed.notify_all();
        while started && !c.go && c.command != Command::Stop {
            c = self.wait(c);
        }
        let go = started && c.command != Command::Stop;
        drop(c);
        if go {
            self.node_loop(i);
        }
        self.node_finish(i);
    }

    /// Updates node `i` until the stop request or one of its hooks fails.
    ///
    /// Running freely, the node updates once the time reaches its target. The
    /// first target is one period after the last update (the start, or the
    /// restored one); each next target is one period after the previous target,
    /// so that waking up late does not delay the following updates. Only an
    /// update a period or more late (already at or past the target after its
    /// own) starts counting again from its own time, so the node never updates
    /// several times in a row to catch up. Without a period, the target is 1 ns
    /// after the last update. The manual mode updates on every step that has
    /// Δt > 0.
    fn node_loop(&self, i: usize) {
        let mut last_ns = self.last_update_ns[i].load(Ordering::SeqCst);
        let period_ns = self.period_ns[i];
        let mut target_ns = last_ns.saturating_add(period_ns);
        loop {
            let (now_ns, word) = match self.boundary(i) {
                Boundary::Exit => return,
                Boundary::Again => continue,
                Boundary::Update { now_ns, word } => (now_ns, word),
            };
            let next_ns = if self.manual {
                last_ns.saturating_add(1)
            } else {
                target_ns
            };
            if now_ns < next_ns {
                if self.manual {
                    // Nothing to do for a step without time passing.
                    self.end_step();
                } else {
                    // Wait for the time to reach the next update instead of
                    // spinning, but wake up on any control request.
                    self.timer.wait_until_ns_or(next_ns, Duration::MAX, || {
                        self.word.load(Ordering::SeqCst) != word
                    });
                }
                continue;
            }
            // From the nanosecond difference, not a difference of seconds,
            // which would round differently at every absolute time.
            let dt = (now_ns - last_ns) as f64 / 1e9;
            if self.call(i, "on_update", |n| n.update(dt)).is_err() {
                return;
            }
            last_ns = now_ns;
            self.last_update_ns[i].store(now_ns, Ordering::SeqCst);
            target_ns = target_ns.saturating_add(period_ns);
            if now_ns >= target_ns {
                target_ns = now_ns.saturating_add(period_ns);
            }
            if self.manual {
                self.end_step();
            }
        }
    }

    /// The loop boundary: the only place where the node pauses or stops, so an
    /// update always runs to its end. Returns the time for the next update.
    ///
    /// The time read here is never later than the frozen time T of a pause (or
    /// of a stop on an exception) whose request the node did not see. The
    /// controller stores the new control word and only then freezes the timer,
    /// both under the control lock. The node reads the word, then the time, then
    /// the word again, and uses the time only if the word did not change. If the
    /// freeze came before the time was read, the timer's lock orders the
    /// controller's store before the node's second load, which then sees the new
    /// word, and the node discards the time. So a time that is used was read
    /// before the freeze, and is at most T.
    fn boundary(&self, i: usize) -> Boundary {
        // Fast path for free running: no lock shared with the other nodes.
        let word = self.word.load(Ordering::SeqCst);
        if !self.manual && is_run(word) {
            let now_ns = self.timer.time_ns();
            if self.word.load(Ordering::SeqCst) != word {
                return Boundary::Again;
            }
            return Boundary::Update { now_ns, word };
        }
        let c = self.lock();
        match c.command {
            Command::Stop => Boundary::Exit,
            Command::Pause => {
                let epoch = c.epoch;
                drop(c);
                if self.pause_node(i, epoch) {
                    Boundary::Again
                } else {
                    Boundary::Exit
                }
            }
            Command::Run if self.manual && c.grant != Some(i) => {
                drop(self.wait(c));
                Boundary::Again
            }
            Command::Run => {
                // Under the control lock, the freeze cannot come in between.
                let now_ns = self.timer.time_ns();
                let word = self.word.load(Ordering::SeqCst);
                Boundary::Update { now_ns, word }
            }
        }
    }

    /// Pauses node `i` for the pause request of `epoch` and waits until the
    /// request changes. Returns whether the node goes on with its loop.
    fn pause_node(&self, i: usize, epoch: u64) -> bool {
        if self.call(i, "on_pause", |n| n.pause()).is_err() {
            return false;
        }
        let mut c = self.lock();
        c.nodes[i].paused_at = Some(epoch);
        self.changed.notify_all();
        while c.epoch == epoch {
            c = self.wait(c);
        }
        c.nodes[i].paused_at = None;
        // A stop while paused ends the node without `resume`. But if the pause
        // was ended first and the stop came before this node woke up, the node
        // still resumes (and stops at the next boundary), so that every
        // completed capture is followed by `resume` on every node.
        let resumed = c.run_epoch > epoch;
        drop(c);
        resumed && self.call(i, "on_resume", |n| n.resume()).is_ok()
    }

    fn end_step(&self) {
        let mut c = self.lock();
        c.grant = None;
        self.changed.notify_all();
    }

    /// Node `i` left its loop: report it, call `exception` if the system stops
    /// because of one, and call `stop` once the control thread allows it.
    fn node_finish(&self, i: usize) {
        let mut c = self.lock();
        c.nodes[i].exited = true;
        self.changed.notify_all();
        while !c.stop_allowed {
            // An exception can come after this node already left the loop for
            // a normal stop; it still gets `exception` before `stop`.
            if c.failed && c.nodes[i].exception == ExceptionHook::NotCalled {
                c.nodes[i].exception = ExceptionHook::Calling;
                drop(c);
                // Called without waiting for the others, for an early safe stop.
                let _ = self.invoke(i, "on_exception", |n| n.exception());
                c = self.lock();
                c.nodes[i].exception = ExceptionHook::Done;
                self.changed.notify_all();
            } else {
                c = self.wait(c);
            }
        }
        drop(c);
        let _ = self.invoke(i, "on_stop", |n| n.stop());
    }
}

/// The number of runtimes started and not yet finished, per timer (by the
/// address of its `Arc`). A timer has an entry only while the number is
/// positive, and each counted runtime holds the timer through its [`Live`], so
/// the address cannot be reused by another timer while it has an entry.
static LIVE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

fn lock_live() -> MutexGuard<'static, BTreeMap<usize, usize>> {
    LIVE.lock().unwrap_or_else(|e| e.into_inner())
}

/// One live runtime on this timer; dropping it uncounts it.
struct Live(Arc<Timer>);

impl Drop for Live {
    fn drop(&mut self) {
        let timer_id = Arc::as_ptr(&self.0) as usize;
        let mut live = lock_live();
        if let Some(k) = live.get_mut(&timer_id) {
            *k -= 1;
            if *k == 0 {
                live.remove(&timer_id);
            }
        }
    }
}

enum Boundary {
    Exit,
    Again,
    Update { now_ns: u64, word: u64 },
}

fn finished_error(failure: &OnceLock<NodeFailure>) -> RuntimeError {
    match failure.get() {
        Some(failure) => RuntimeError::Failed(failure.clone()),
        None => RuntimeError::Finished,
    }
}

/// What a request returns when the system has already ended.
type Ended<T> = fn(&OnceLock<NodeFailure>) -> Result<T, RuntimeError>;

fn unavailable<T>(failure: &OnceLock<NodeFailure>) -> Result<T, RuntimeError> {
    Err(finished_error(failure))
}

/// A stop request succeeds on a system that has stopped, unless by a failure.
fn stopped(failure: &OnceLock<NodeFailure>) -> Result<(), RuntimeError> {
    match failure.get() {
        Some(failure) => Err(RuntimeError::Failed(failure.clone())),
        None => Ok(()),
    }
}

/// What is left of a wait until `deadline` (no end if `None`).
fn remaining(deadline: Option<Instant>) -> Duration {
    deadline.map_or(Duration::MAX, |d| {
        d.saturating_duration_since(Instant::now())
    })
}

/// The control thread: serves the host's requests one at a time and ends the
/// system on a failure.
struct Controller<N: NodeHooks> {
    shared: Arc<Shared<N>>,
    threads: Vec<JoinHandle<()>>,
    exception_capture: Option<ExceptionCapture<N::State>>,
    pause_timeout: Duration,
    done: bool,
}

impl<N: NodeHooks> Controller<N> {
    fn run(mut self, messages: Receiver<Message<N>>, started: Sender<Result<(), RuntimeError>>) {
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

    fn step(&mut self, order: Vec<usize>) -> Result<(), RuntimeError> {
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

    fn capture<R>(
        &mut self,
        timeout: Duration,
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
        let deadline = Instant::now().checked_add(timeout);
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
    fn finish(&mut self) {
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
        let failed = c.failed;
        drop(c);
        if failed {
            if let Some(capture) = self.exception_capture.take() {
                // `exception` does not change what `state_dict` returns, so the
                // state is still the one at the failure. A node whose
                // `state_dict` fails is left out; the hook logged the error.
                let states = (0..s.hooks.len())
                    .filter_map(|i| {
                        let state = s.invoke(i, "state_dict", N::state_dict);
                        state.ok().map(|state| (i, state))
                    })
                    .collect();
                let failure = s.failure.get().expect("a failure was recorded");
                // A panic must not keep the system from ending (and the time
                // frozen); the panic hook has already printed it.
                let _ = panic::catch_unwind(AssertUnwindSafe(|| capture(states, failure)));
            }
        }
        let mut c = s.lock();
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

    fn shutdown(&mut self) -> Result<(), RuntimeError> {
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
        exception_capture: Option<ExceptionCapture<N::State>>,
    ) -> Result<(Self, Reply<()>), RuntimeError> {
        let n = nodes.len();
        if names.len() != n {
            return Err(RuntimeError::InvalidArgument(format!(
                "got {n} nodes but {} names",
                names.len()
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
        let period_ns = match config.period_ns {
            None => vec![1; n],
            Some(periods) if periods.len() != n => {
                return Err(RuntimeError::InvalidArgument(format!(
                    "got {n} nodes but {} periods",
                    periods.len()
                )));
            }
            Some(periods) => periods
                .into_iter()
                .map(|period| match period {
                    None => Ok(1),
                    Some(0) => Err(RuntimeError::InvalidArgument(
                        "a period must be > 0 ns".to_owned(),
                    )),
                    Some(ns) => Ok(ns),
                })
                .collect::<Result<_, _>>()?,
        };
        let (to_controller, messages) = mpsc::channel();
        let shared = Arc::new(Shared {
            timer,
            names,
            manual: config.manual,
            capture_on_failure: exception_capture.is_some(),
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
        // is right even if the system ends while starting (the state captured
        // on that exception holds it), and stays right since the time is frozen.
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
            exception_capture,
            pause_timeout: config.pause_timeout,
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
    /// (the configured pause timeout if `None`) if some node does not pause.
    pub fn capture<R: Send + 'static>(
        &self,
        timeout: Option<Duration>,
        build: impl FnOnce(Vec<N::State>) -> R + Send + 'static,
    ) -> Reply<R> {
        self.submit(unavailable, move |controller| {
            let timeout = timeout.unwrap_or(controller.pause_timeout);
            controller.capture(timeout, build)
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

    /// Started and not yet ended.
    pub fn is_running(&self) -> bool {
        let c = self.shared.lock();
        c.go && !c.finished
    }

    pub fn is_finished(&self) -> bool {
        self.shared.lock().finished
    }

    pub fn names(&self) -> &[String] {
        &self.shared.names
    }

    /// The time of each node's last update on the runtime's timer (where its
    /// next Δt starts).
    pub fn last_update_ns(&self) -> Vec<u64> {
        let last = &self.shared.last_update_ns;
        last.iter().map(|ns| ns.load(Ordering::SeqCst)).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freeze_holds_the_timer_until_every_reason_is_released() {
        let timer = Arc::new(Timer::new());
        let mut freeze = Freeze::default();
        freeze.hold(FreezeReason::Failure, &timer);
        freeze.hold(FreezeReason::Capture, &timer);
        // Ending the capture must not undo the freeze for the exception.
        freeze.release(FreezeReason::Capture);
        assert!(timer.is_paused());
        // Releasing a reason that is not held changes nothing.
        freeze.release(FreezeReason::Capture);
        assert!(timer.is_paused());
        freeze.release(FreezeReason::Failure);
        assert!(!timer.is_paused());
    }

    #[test]
    fn releasing_every_reason_keeps_a_pause_by_someone_else() {
        let timer = Arc::new(Timer::new());
        timer.pause();
        let mut freeze = Freeze::default();
        freeze.hold(FreezeReason::Startup, &timer);
        freeze.release_all();
        assert!(timer.is_paused());
    }
}
