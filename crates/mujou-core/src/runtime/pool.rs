//! The worker pool: the nodes without a thread of their own, M of them on N
//! worker threads.
//!
//! A pool node is a small state machine ([`Task`]) kept under the control lock.
//! A worker takes a piece of work for one node under that lock, runs the hook
//! outside every lock, and puts the node back. The lifecycle work (start,
//! pause, resume, exception, stop) goes first, in the order it became due
//! except that a node whose hook failed goes ahead with its `exception`; then
//! the updates, earliest target first and by index on a tie. A node is taken
//! only under the lock, from a queue, and only the worker that took it puts it
//! back, so no node runs on two workers at once.
//!
//! The transitions follow the loop of a dedicated thread (`node`). What such a
//! thread blocks on is here a state that the change it waits for re-evaluates
//! ([`Shared::wake_pool`]): a worker never waits for one node, which would
//! deadlock with fewer workers than nodes. The only wait on the time is that of
//! the one worker that keeps the time for the earliest target (the
//! timekeeper); the others wait for work.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::MutexGuard;
use std::sync::atomic::Ordering;
use std::time::Duration;

use super::node::{Command, ExceptionHook, next_target_ns};
use super::{Control, NodeHooks, Shared};

/// Where a pool node is in its life, as the position of a dedicated thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// `start` is due.
    Start,
    /// Started; waits until every node has started (or the stop).
    WaitGo,
    /// In the update loop.
    Loop,
    /// Called `pause` for the pause request of this epoch and waits.
    Paused(u64),
    /// Left the loop: `exception` (on a failure) and `stop` are left.
    Finish,
    /// `stop` returned.
    Done,
}

/// Where a pool node waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum At {
    /// Waits for a change of the control state.
    Parked,
    /// In [`Sched::ready`]; what it does is decided when a worker takes it.
    Ready,
    /// In [`Sched::heap`] under this generation; older entries are stale.
    Heap(u64),
    /// A worker runs one of its hooks.
    Running,
}

struct Task {
    phase: Phase,
    at: At,
    generation: u64,
    /// The time of the next free-running update (see [`next_target_ns`]).
    target_ns: u64,
}

#[derive(Debug, Clone, Copy)]
enum Work {
    Start,
    Update { now_ns: u64, dt: f64 },
    Pause(u64),
    Resume,
    Exception,
    Stop,
}

enum Next {
    Hook(Work),
    /// An update at this target.
    Heap(u64),
    Park,
}

/// The state of the pool, under the control lock. A dedicated node has a
/// [`Task`] too, which stays untouched.
pub(super) struct Sched {
    tasks: Vec<Task>,
    ready: VecDeque<usize>,
    /// The free-running updates: `(target_ns, node, generation)`, earliest first.
    heap: BinaryHeap<Reverse<(u64, usize, u64)>>,
    /// The target the timekeeper waits for, if a worker waits on the timer.
    timekeeper: Option<u64>,
    /// The pool nodes that have not finished `stop`.
    left: usize,
}

impl Sched {
    /// Every pool node starts with its `start` due, in index order.
    pub(super) fn new(dedicated: &[bool]) -> Self {
        let mut sched = Self {
            tasks: Vec::with_capacity(dedicated.len()),
            ready: VecDeque::new(),
            heap: BinaryHeap::new(),
            timekeeper: None,
            left: 0,
        };
        for (i, &own_thread) in dedicated.iter().enumerate() {
            let at = if own_thread {
                At::Parked
            } else {
                sched.ready.push_back(i);
                sched.left += 1;
                At::Ready
            };
            sched.tasks.push(Task {
                phase: Phase::Start,
                at,
                generation: 0,
                target_ns: 0,
            });
        }
        sched
    }
}

impl<N: NodeHooks> Shared<N> {
    /// The body of a worker thread: runs the pool nodes until all of them have
    /// stopped.
    pub(super) fn run_worker(&self) {
        let mut c = self.lock();
        while c.sched.left > 0 {
            // Work in the ready queue means the timekeeper, if any, is already
            // woken: the control side wakes it when it queues work, a worker
            // takes the hook it queues itself under the same lock, and a worker
            // keeps the time only with the queue empty. So a target this worker
            // just put back is not missed while it runs a hook first.
            if let Some(i) = c.sched.ready.pop_front() {
                if let Some(work) = self.take(i, &mut c) {
                    c = self.run_work(i, work, c);
                }
                continue;
            }
            if let Some((target_ns, i)) = self.earliest(&mut c) {
                // Read under the control lock, so no freeze for a pause comes
                // between reading the command and the time.
                let now_ns = self.timer.time_ns();
                if now_ns >= target_ns {
                    c.sched.heap.pop();
                    c.sched.tasks[i].at = At::Running;
                    // From the nanosecond difference, as on a dedicated thread.
                    let last_ns = self.last_update_ns[i].load(Ordering::SeqCst);
                    let dt = (now_ns - last_ns) as f64 / 1e9;
                    c = self.run_work(i, Work::Update { now_ns, dt }, c);
                    continue;
                }
                match c.sched.timekeeper {
                    None => {
                        c = self.keep_time(target_ns, c);
                        continue;
                    }
                    // A worker put back an update due before the timekeeper's
                    // target: let the timekeeper wait for this one instead.
                    Some(waiting_ns) if target_ns < waiting_ns => self.interrupt_timekeeper(&c),
                    Some(_) => {}
                }
            }
            c = self
                .sched_changed
                .wait(c)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Re-evaluates every pool node after a change of the control state, and
    /// wakes the timekeeper once if any of them needs it.
    pub(super) fn wake_pool(&self, c: &mut Control) {
        let mut look = false;
        for i in 0..self.dedicated.len() {
            look |= !self.dedicated[i] && self.schedule(i, c);
        }
        if look {
            self.interrupt_timekeeper(c);
        }
    }

    /// Re-evaluates node `i` (if it is in the pool) after a change for it.
    pub(super) fn wake_node(&self, c: &mut Control, i: usize) {
        if !self.dedicated[i] && self.schedule(i, c) {
            self.interrupt_timekeeper(c);
        }
    }

    /// The earliest free-running update, if the nodes run freely. Drops the
    /// stale entries on the way.
    fn earliest(&self, c: &mut Control) -> Option<(u64, usize)> {
        if self.manual || c.command != Command::Run {
            return None;
        }
        while let Some(&Reverse((target_ns, i, generation))) = c.sched.heap.peek() {
            if c.sched.tasks[i].at == At::Heap(generation) {
                return Some((target_ns, i));
            }
            c.sched.heap.pop();
        }
        None
    }

    /// Waits on the timer for `target_ns`, as the only worker that does, until
    /// the time reaches it, an earlier target or other work comes, or the
    /// command changes.
    fn keep_time<'a>(
        &'a self,
        target_ns: u64,
        mut c: MutexGuard<'a, Control>,
    ) -> MutexGuard<'a, Control> {
        c.sched.timekeeper = Some(target_ns);
        // Read under the control lock: a change after this bumps one of them.
        let epoch = self.sched_epoch.load(Ordering::SeqCst);
        let word = self.word.load(Ordering::SeqCst);
        drop(c);
        self.timer.wait_until_ns_or(target_ns, Duration::MAX, || {
            self.sched_epoch.load(Ordering::SeqCst) != epoch
                || self.word.load(Ordering::SeqCst) != word
        });
        let mut c = self.lock();
        c.sched.timekeeper = None;
        c
    }

    /// Wakes the timekeeper, so that it looks at the queues again.
    fn interrupt_timekeeper(&self, c: &Control) {
        if c.sched.timekeeper.is_some() {
            self.sched_epoch.fetch_add(1, Ordering::SeqCst);
            self.timer.interrupt_waiters();
        }
    }

    /// Puts node `i` where it waits, unless a worker has it or will decide
    /// when it takes it. Returns whether the timekeeper must look again.
    fn schedule(&self, i: usize, c: &mut Control) -> bool {
        if matches!(c.sched.tasks[i].at, At::Ready | At::Running) {
            return false;
        }
        let next = self.advance(i, c);
        self.place(i, next, false, c)
    }

    /// Takes node `i` out of the ready queue: the hook to run now, if any.
    fn take(&self, i: usize, c: &mut Control) -> Option<Work> {
        c.sched.tasks[i].at = At::Parked;
        match self.advance(i, c) {
            Next::Hook(work) => {
                if let Work::Exception = work {
                    c.nodes[i].exception = ExceptionHook::Calling;
                }
                c.sched.tasks[i].at = At::Running;
                Some(work)
            }
            next => {
                // This worker looks at the queues next.
                self.place(i, next, false, c);
                None
            }
        }
    }

    /// Puts node `i` where `next` says: a hook goes to the back of the ready
    /// queue, or to its front if `first`.
    ///
    /// Returns whether the timekeeper must look again: the node has a hook to
    /// run (with every other worker busy, the timekeeper takes it) or an update
    /// due before the timekeeper's target. A worker that puts a node back looks
    /// at the queues itself right after, so it ignores this, and only the
    /// control side wakes the timekeeper (the worker may run a hook from the
    /// ready queue first; see [`Self::run_worker`] for why that misses no
    /// target). Waking it wakes every waiter on the timer (the other runs'
    /// timekeepers, dedicated threads, the host), so a node of period 1 ns
    /// putting itself back after each update must not.
    fn place(&self, i: usize, next: Next, first: bool, c: &mut Control) -> bool {
        match next {
            Next::Hook(_) => {
                c.sched.tasks[i].at = At::Ready;
                if first {
                    c.sched.ready.push_front(i);
                } else {
                    c.sched.ready.push_back(i);
                }
                self.sched_changed.notify_one();
                true
            }
            Next::Heap(target_ns) => {
                let task = &mut c.sched.tasks[i];
                if let At::Heap(_) = task.at {
                    // Already there: the target changes only after an update.
                    return false;
                }
                task.generation += 1;
                task.at = At::Heap(task.generation);
                let entry = (target_ns, i, task.generation);
                c.sched.heap.push(Reverse(entry));
                match c.sched.timekeeper {
                    None => {
                        self.sched_changed.notify_one();
                        false
                    }
                    Some(waiting_ns) => target_ns < waiting_ns,
                }
            }
            Next::Park => {
                c.sched.tasks[i].at = At::Parked;
                false
            }
        }
    }

    /// What node `i` does next under the control state, as a dedicated thread
    /// would decide at the same point. Makes the transitions that call no hook.
    fn advance(&self, i: usize, c: &mut Control) -> Next {
        loop {
            match c.sched.tasks[i].phase {
                Phase::Start => return Next::Hook(Work::Start),
                Phase::WaitGo => {
                    if c.command == Command::Stop {
                        self.leave_loop(i, c);
                        continue;
                    }
                    if !c.go {
                        return Next::Park;
                    }
                    let last_ns = self.last_update_ns[i].load(Ordering::SeqCst);
                    let task = &mut c.sched.tasks[i];
                    task.target_ns = last_ns.saturating_add(self.period_ns[i]);
                    task.phase = Phase::Loop;
                }
                Phase::Loop => match c.command {
                    Command::Stop => self.leave_loop(i, c),
                    Command::Pause => return Next::Hook(Work::Pause(c.epoch)),
                    Command::Run if self.manual => {
                        if c.grant != Some(i) {
                            return Next::Park;
                        }
                        // Under the control lock, the freeze cannot come in
                        // between.
                        let now_ns = self.timer.time_ns();
                        let last_ns = self.last_update_ns[i].load(Ordering::SeqCst);
                        if now_ns <= last_ns {
                            // Nothing to do for a step without time passing.
                            self.end_step(c);
                            return Next::Park;
                        }
                        let dt = (now_ns - last_ns) as f64 / 1e9;
                        return Next::Hook(Work::Update { now_ns, dt });
                    }
                    Command::Run => return Next::Heap(c.sched.tasks[i].target_ns),
                },
                Phase::Paused(epoch) => {
                    if c.epoch == epoch {
                        return Next::Park;
                    }
                    c.nodes[i].paused_at = None;
                    // As on a dedicated thread: a stop while paused ends the
                    // node without `resume`, but a pause ended before the stop
                    // is still followed by `resume`.
                    if c.run_epoch > epoch {
                        return Next::Hook(Work::Resume);
                    }
                    self.leave_loop(i, c);
                }
                Phase::Finish => {
                    if c.stop_allowed {
                        return Next::Hook(Work::Stop);
                    }
                    // Also after the node left the loop for a normal stop.
                    if c.failed && c.nodes[i].exception == ExceptionHook::NotCalled {
                        return Next::Hook(Work::Exception);
                    }
                    return Next::Park;
                }
                Phase::Done => return Next::Park,
            }
        }
    }

    /// Runs `work` for node `i` outside the control lock, records how it went
    /// and puts the node back.
    fn run_work<'a>(
        &'a self,
        i: usize,
        work: Work,
        c: MutexGuard<'a, Control>,
    ) -> MutexGuard<'a, Control> {
        let sched = &c.sched;
        if sched.timekeeper.is_none() && !(sched.ready.is_empty() && sched.heap.is_empty()) {
            // Leave what is left (and the timekeeping) to another worker.
            self.sched_changed.notify_one();
        }
        drop(c);
        let ok = match work {
            Work::Start => self.call(i, "on_start", |n| n.start()).is_ok(),
            Work::Update { dt, .. } => self.call(i, "on_update", |n| n.update(dt)).is_ok(),
            Work::Pause(_) => self.call(i, "on_pause", |n| n.pause()).is_ok(),
            Work::Resume => self.call(i, "on_resume", |n| n.resume()).is_ok(),
            // Their errors are ignored.
            Work::Exception => {
                self.clean_up(i, "on_exception", |n| n.exception());
                true
            }
            Work::Stop => {
                self.clean_up(i, "on_stop", |n| n.stop());
                true
            }
        };
        let mut c = self.lock();
        self.record(i, work, ok, &mut c);
        c.sched.tasks[i].at = At::Parked;
        let next = self.advance(i, &mut c);
        // A failed hook is followed by `exception` ahead of the nodes the
        // failure woke, as a dedicated thread calls it right after catching.
        // This worker looks at the queues next.
        self.place(i, next, !ok, &mut c);
        c
    }

    /// Records that `work` for node `i` returned (`ok` if without an error).
    /// A failed hook takes the node out of its loop.
    fn record(&self, i: usize, work: Work, ok: bool, c: &mut Control) {
        if let Work::Start = work {
            c.nodes[i].started = true;
            self.changed.notify_all();
        }
        if !ok {
            self.leave_loop(i, c);
            return;
        }
        match work {
            Work::Start => c.sched.tasks[i].phase = Phase::WaitGo,
            Work::Update { now_ns, .. } => {
                self.last_update_ns[i].store(now_ns, Ordering::SeqCst);
                let task = &mut c.sched.tasks[i];
                task.target_ns = next_target_ns(task.target_ns, now_ns, self.period_ns[i]);
                if self.manual {
                    self.end_step(c);
                }
            }
            Work::Pause(epoch) => {
                c.nodes[i].paused_at = Some(epoch);
                self.changed.notify_all();
                c.sched.tasks[i].phase = Phase::Paused(epoch);
            }
            Work::Resume => c.sched.tasks[i].phase = Phase::Loop,
            Work::Exception => {
                c.nodes[i].exception = ExceptionHook::Done;
                self.changed.notify_all();
            }
            Work::Stop => {
                c.sched.tasks[i].phase = Phase::Done;
                c.sched.left -= 1;
                if c.sched.left == 0 {
                    // Let the idle workers end too.
                    self.sched_changed.notify_all();
                }
            }
        }
    }

    /// Node `i` left its loop and will call no hook but `exception` and `stop`.
    fn leave_loop(&self, i: usize, c: &mut Control) {
        c.sched.tasks[i].phase = Phase::Finish;
        c.nodes[i].exited = true;
        self.changed.notify_all();
    }
}
