//! The state shared by every thread of a runtime, under one lock, and the
//! operations on it: scheduling a node after a change, recording what a hook
//! did, calling a hook, failing and finishing.

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use super::error::{Hook, HookError, NodeFailure, Panicked, RuntimeError, panic_message};
use super::hooks::{NodeHooks, NodeSpec};
use super::lifecycle::{self, Command, Lifecycle, Next, Work};
use super::sched::{Lane, Sched};
use crate::time::{PauseHold, RunClock};

thread_local! {
    /// The runtime (the address of its shared state) whose thread this is, or 0.
    static RUNTIME_OF_THREAD: Cell<usize> = const { Cell::new(0) };
}

/// The state shared by the control thread and the nodes, under one lock.
pub(super) struct Control {
    pub(super) life: Lifecycle,
    pub(super) sched: Sched,
    /// Freezes the time from before the nodes start until all of them are
    /// ready to update.
    pub(super) startup: Option<PauseHold>,
    /// Freezes the time of a capture, from the pause request until the state
    /// is built.
    pub(super) capture: Option<PauseHold>,
    /// The system has ended: every node thread has been joined.
    pub(super) finished: bool,
}

/// A node as the threads of its runtime hold it.
pub(super) struct Member<N> {
    pub(super) name: String,
    pub(super) period_ns: u64,
    /// The start of the node's next Δt. Written only by the thread that runs
    /// the node, after an update.
    pub(super) last_update_ns: AtomicU64,
    hooks: Mutex<N>,
}

pub(super) struct Shared<N: NodeHooks> {
    pub(super) clock: Arc<RunClock>,
    pub(super) nodes: Vec<Member<N>>,
    /// The signals of each lane of [`Control::sched`].
    pub(super) lanes: Vec<Lane>,
    /// The number of workers of the pool.
    pub(super) workers: usize,
    control: Mutex<Control>,
    /// Notified on a change the control thread waits for
    /// ([`Lifecycle::take_changed`], a failure) and at the end.
    changed: Condvar,
    /// [`Lifecycle::word`], stored under the control lock. A node running
    /// freely reads it without taking the lock, and while it waits on the
    /// timer (which checks the interrupt condition under the timer's lock, and
    /// the control lock is taken before the timer's elsewhere).
    pub(super) word: AtomicU64,
    pub(super) failure: Arc<OnceLock<NodeFailure>>,
}

impl<N: NodeHooks> Shared<N> {
    /// The state of a runtime of `nodes` in `lanes` (the lane of each node,
    /// the pool being lane 0), with `workers` workers in the pool.
    pub(super) fn new(
        clock: Arc<RunClock>,
        nodes: Vec<NodeSpec<N>>,
        lanes: &[usize],
        workers: usize,
        manual: bool,
    ) -> Self {
        let n_lanes = lanes.iter().max().map_or(1, |&last| last + 1);
        let life = Lifecycle::new(nodes.len(), manual);
        let members = (nodes.into_iter())
            .map(|spec| Member {
                name: spec.name,
                period_ns: spec.period_ns.get(),
                // Set once the time is frozen for the start.
                last_update_ns: AtomicU64::new(0),
                hooks: Mutex::new(spec.hooks),
            })
            .collect();
        Self {
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
        }
    }

    /// Marks the calling thread as one of this runtime's.
    pub(super) fn own_thread(&self) {
        RUNTIME_OF_THREAD.set(self.id());
    }

    /// The calling thread is one of this runtime's.
    pub(super) fn on_own_thread(&self) -> bool {
        RUNTIME_OF_THREAD.get() == self.id()
    }

    /// Identifies this runtime for [`RUNTIME_OF_THREAD`].
    fn id(&self) -> usize {
        self as *const Self as usize
    }

    pub(super) fn lock(&self) -> MutexGuard<'_, Control> {
        // Hooks run outside this lock, so a poisoned state is still consistent.
        self.control.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn wait<'a>(&self, guard: MutexGuard<'a, Control>) -> MutexGuard<'a, Control> {
        self.changed.wait(guard).unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn wait_timeout<'a>(
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
    pub(super) fn set_command(&self, c: &mut Control, command: Command) -> bool {
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
    pub(super) fn wake_all(&self, c: &mut Control) {
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
    pub(super) fn wake(&self, c: &mut Control, i: usize) {
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
    /// when it leaves work behind (`worker`): waking one for the node it is
    /// about to take again would make the workers chase each other. Waking
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
    pub(super) fn advance(&self, i: usize, c: &mut Control) -> Next {
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
    pub(super) fn record(&self, i: usize, work: Work, ok: bool, c: &mut Control) {
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
    pub(super) fn interrupt_timekeeper(&self, lane: usize, c: &Control) {
        if c.sched.timekeeper(lane).is_some() {
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
    pub(super) fn dt(&self, i: usize, now_ns: u64) -> f64 {
        let last_ns = self.nodes[i].last_update_ns.load(Ordering::SeqCst);
        lifecycle::dt(now_ns, last_ns)
    }

    /// Calls a hook on node `i`, with the node locked, turning a panic into an
    /// error. The scheduling never lets two threads run a node at once; only
    /// the captures after the end may wait here for each other.
    pub(super) fn invoke<T>(
        &self,
        i: usize,
        hook: Hook,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, HookError> {
        // Hooks run outside the runtime's locks, so a poisoned node is consistent.
        let mut node = self.nodes[i]
            .hooks
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        guarded(&mut *node, hook, f)
    }

    /// Calls a hook whose error ends the system ([`Self::fail`]).
    pub(super) fn call<T>(
        &self,
        i: usize,
        hook: Hook,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, ()> {
        self.invoke(i, hook, f).map_err(|e| self.fail(i, e))
    }

    /// Takes every node's state after a normal end, on the calling thread,
    /// and builds the result. A failure of `capture` is reported to the
    /// caller only: the system has already ended.
    pub(super) fn capture_ended<R>(
        &self,
        build: impl FnOnce(Vec<N::State>) -> R,
    ) -> Result<R, RuntimeError> {
        self.check_failure()?;
        let states = (0..self.nodes.len()).map(|i| {
            self.invoke(i, Hook::Capture, N::capture)
                .map_err(|e| RuntimeError::CaptureFailed(self.failure_of(i, e)))
        });
        build_states(states.collect::<Result<_, _>>()?, build)
    }

    /// Fails with the first failure, if one was recorded.
    pub(super) fn check_failure(&self) -> Result<(), RuntimeError> {
        match self.failure.get() {
            Some(failure) => Err(RuntimeError::Failed(failure.clone())),
            None => Ok(()),
        }
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
    /// their next loop boundary. The caller tells the control thread, which
    /// ends the system.
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
    }

    /// Lets the time run and marks the system finished.
    pub(super) fn mark_finished(&self, c: &mut Control) {
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
pub(super) fn build_states<S, R>(
    states: Vec<S>,
    build: impl FnOnce(Vec<S>) -> R,
) -> Result<R, RuntimeError> {
    panic::catch_unwind(AssertUnwindSafe(|| build(states))).map_err(|payload| {
        let message = panic_message(payload);
        RuntimeError::Internal(format!("the build of a capture panicked: {message}"))
    })
}
