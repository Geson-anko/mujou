//! The threads that run the nodes, and the queues they take the work from.
//!
//! The nodes are split into lanes: the pool, whose M nodes share N worker
//! threads, and one lane per node given a dedicated thread, which is a worker
//! of its own. Every worker runs the same loop ([`Worker::run`]): it takes a
//! piece of work for one node of its lane under the control lock, runs the
//! hook outside every lock, and puts the node back. What the node does next is
//! the [`Lifecycle`](super::lifecycle::Lifecycle)'s decision. The lifecycle
//! work (start, pause, resume, exception, stop) goes first, in the order it
//! became due except that a node whose hook failed goes ahead with its
//! `exception`; then the updates, earliest target first and by index on a
//! tie. A node is taken only under the lock, from a queue, and only the worker
//! that took it puts it back, so no node runs on two threads at once.
//!
//! A worker never waits for one node, which would deadlock with fewer workers
//! than nodes: it waits for work, and the one worker of a lane that keeps the
//! time for the earliest target (the timekeeper) waits on the timer. A lane of
//! one node needs no queue while the nodes run freely, so its worker updates it
//! without the control lock until the command changes ([`Worker::run_alone`]).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, MutexGuard};
use std::time::Duration;

use super::error::Hook;
use super::lifecycle::{Next, Work, next_target_ns};
use super::{Control, NodeHooks, Shared};

/// Where a node waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum At {
    /// Waits for a change of the control state.
    Parked,
    /// In its lane's ready queue; what it does is decided when a worker takes it.
    Ready,
    /// In its lane's heap under this generation; older entries are stale.
    Heap(u64),
    /// A worker runs one of its hooks.
    Running,
}

struct Seat {
    lane: usize,
    at: At,
    generation: u64,
}

/// The work of one lane, under the control lock.
struct Queue {
    ready: VecDeque<usize>,
    /// The free-running updates: `(target_ns, node, generation)`, earliest first.
    heap: BinaryHeap<Reverse<(u64, usize, u64)>>,
    /// The target the timekeeper waits for, if a worker waits on the timer.
    timekeeper: Option<u64>,
    nodes: usize,
    /// The nodes that have not finished `stop`.
    left: usize,
}

/// How [`Sched::place`] put a node.
pub(super) struct Placed {
    pub(super) lane: usize,
    /// A worker of the lane may have work now.
    pub(super) notify: bool,
    /// The timekeeper must look again: the node has a hook to run (with every
    /// other worker busy, the timekeeper takes it) or an update due before its
    /// target.
    pub(super) look: bool,
}

/// The queues of every lane, under the control lock.
pub(super) struct Sched {
    seats: Vec<Seat>,
    queues: Vec<Queue>,
}

impl Sched {
    /// The nodes in the lanes `lanes[i]`, of `n` lanes, each with its `start`
    /// due, in index order.
    pub(super) fn new(lanes: &[usize], n: usize) -> Self {
        let mut queues: Vec<_> = (0..n)
            .map(|_| Queue {
                ready: VecDeque::new(),
                heap: BinaryHeap::new(),
                timekeeper: None,
                nodes: 0,
                left: 0,
            })
            .collect();
        for (i, &lane) in lanes.iter().enumerate() {
            let queue = &mut queues[lane];
            queue.ready.push_back(i);
            queue.nodes += 1;
            queue.left += 1;
        }
        let seats = (lanes.iter())
            .map(|&lane| Seat {
                lane,
                at: At::Ready,
                generation: 0,
            })
            .collect();
        Self { seats, queues }
    }

    pub(super) fn lane(&self, i: usize) -> usize {
        self.seats[i].lane
    }

    /// A worker has node `i` or will decide what it does when it takes it.
    pub(super) fn busy(&self, i: usize) -> bool {
        matches!(self.seats[i].at, At::Ready | At::Running)
    }

    /// Parks node `i` once its worker has it back.
    pub(super) fn park(&mut self, i: usize) {
        self.seats[i].at = At::Parked;
    }

    /// Puts node `i` where `next` says: a hook goes to the back of its lane's
    /// ready queue, or to its front if `first`.
    pub(super) fn place(&mut self, i: usize, next: Next, first: bool) -> Placed {
        let seat = &mut self.seats[i];
        let queue = &mut self.queues[seat.lane];
        let (notify, look) = match next {
            Next::Hook(_) => {
                seat.at = At::Ready;
                if first {
                    queue.ready.push_front(i);
                } else {
                    queue.ready.push_back(i);
                }
                (true, true)
            }
            // Already there: the target changes only after an update.
            Next::Wait(_) if matches!(seat.at, At::Heap(_)) => (false, false),
            Next::Wait(target_ns) => {
                seat.generation += 1;
                seat.at = At::Heap(seat.generation);
                queue.heap.push(Reverse((target_ns, i, seat.generation)));
                match queue.timekeeper {
                    None => (true, false),
                    Some(waiting_ns) => (false, target_ns < waiting_ns),
                }
            }
            Next::Park => {
                seat.at = At::Parked;
                (false, false)
            }
        };
        Placed {
            lane: seat.lane,
            notify,
            look,
        }
    }

    /// Node `i` finished `stop`. Returns whether its lane has no node left.
    pub(super) fn finish(&mut self, i: usize) -> bool {
        let queue = &mut self.queues[self.seats[i].lane];
        queue.left -= 1;
        queue.left == 0
    }

    pub(super) fn has_timekeeper(&self, lane: usize) -> bool {
        self.queues[lane].timekeeper.is_some()
    }

    /// The earliest free-running update of `lane`. Drops the stale entries on
    /// the way.
    fn earliest(&mut self, lane: usize) -> Option<(u64, usize)> {
        let queue = &mut self.queues[lane];
        while let Some(&Reverse((target_ns, i, generation))) = queue.heap.peek() {
            if self.seats[i].at == At::Heap(generation) {
                return Some((target_ns, i));
            }
            queue.heap.pop();
        }
        None
    }

    /// Takes node `i`, the earliest of its lane's heap, to run it.
    fn take_earliest(&mut self, i: usize) {
        let seat = &mut self.seats[i];
        self.queues[seat.lane].heap.pop();
        seat.at = At::Running;
    }
}

/// The signals of one lane, outside the control lock.
#[derive(Default)]
pub(super) struct Lane {
    /// Notified when the lane has new work, on the control lock.
    pub(super) changed: Condvar,
    /// Bumped to wake the lane's timekeeper: read with the control word in
    /// its interrupt condition.
    pub(super) epoch: AtomicU64,
}

/// A thread that runs the nodes of one lane.
pub(super) struct Worker<'a, N: NodeHooks> {
    pub(super) shared: &'a Shared<N>,
    pub(super) lane: usize,
}

type Guard<'a> = MutexGuard<'a, Control>;

impl<'a, N: NodeHooks> Worker<'a, N> {
    /// Runs the nodes of the lane until all of them have stopped.
    pub(super) fn run(&self) {
        let s = self.shared;
        let mut c = s.lock();
        while c.sched.queues[self.lane].left > 0 {
            // Work in the ready queue means the timekeeper, if any, is already
            // woken: the control side wakes it when it queues work, a worker
            // takes the hook it queues itself under the same lock, and a worker
            // keeps the time only with the queue empty. So a target this worker
            // just put back is not missed while it runs a hook first.
            if let Some(i) = c.sched.queues[self.lane].ready.pop_front() {
                if let Some(work) = self.take(i, &mut c) {
                    c = self.run_work(i, work, c);
                }
                continue;
            }
            if let Some((target_ns, i)) = self.earliest(&mut c) {
                if c.sched.queues[self.lane].nodes == 1 {
                    c = self.run_alone(i, target_ns, c);
                    continue;
                }
                // Read under the control lock, so no freeze for a pause comes
                // between reading the command and the time.
                let now_ns = s.clock.now_ns();
                if now_ns >= target_ns {
                    c.sched.take_earliest(i);
                    c = self.run_work(i, Work::Update { now_ns }, c);
                    continue;
                }
                match c.sched.queues[self.lane].timekeeper {
                    None => {
                        c = self.keep_time(target_ns, c);
                        continue;
                    }
                    // A worker put back an update due before the timekeeper's
                    // target: let the timekeeper wait for this one instead.
                    Some(waiting_ns) if target_ns < waiting_ns => {
                        s.interrupt_timekeeper(self.lane, &c);
                    }
                    Some(_) => {}
                }
            }
            c = (s.lanes[self.lane].changed)
                .wait(c)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The earliest free-running update of the lane, if the nodes run freely.
    fn earliest(&self, c: &mut Control) -> Option<(u64, usize)> {
        if !c.life.running_freely() {
            return None;
        }
        c.sched.earliest(self.lane)
    }

    /// Takes node `i` out of the ready queue: the hook to run now, if any.
    fn take(&self, i: usize, c: &mut Control) -> Option<Work> {
        let s = self.shared;
        c.sched.park(i);
        match s.advance(i, c) {
            Next::Hook(work) => {
                c.sched.seats[i].at = At::Running;
                Some(work)
            }
            next => {
                // This worker looks at the queues next, so it wakes nobody
                // (see `Shared::schedule`).
                c.sched.place(i, next, false);
                None
            }
        }
    }

    /// Runs `work` for node `i` outside the control lock, records how it went
    /// and puts the node back.
    fn run_work(&self, i: usize, work: Work, c: Guard<'a>) -> Guard<'a> {
        let s = self.shared;
        let queue = &c.sched.queues[self.lane];
        if queue.timekeeper.is_none() && !(queue.ready.is_empty() && queue.heap.is_empty()) {
            // Leave what is left (and the timekeeping) to another worker.
            s.lanes[self.lane].changed.notify_one();
        }
        drop(c);
        let ok = self.call(i, work);
        let mut c = s.lock();
        s.record(i, work, ok, &mut c);
        c.sched.park(i);
        let next = s.advance(i, &mut c);
        // A failed hook is followed by `exception` ahead of the nodes the
        // failure woke, as right after catching it. This worker looks at the
        // queues next, so it wakes nobody.
        c.sched.place(i, next, !ok);
        c
    }

    /// Calls the hook of `work` on node `i`. Returns false if it failed.
    fn call(&self, i: usize, work: Work) -> bool {
        let s = self.shared;
        match work {
            Work::Start => s
                .call(i, Hook::Start, N::start)
                .and_then(|()| s.call(i, Hook::Restore, N::restore))
                .is_ok(),
            Work::Update { now_ns } => {
                let dt = s.dt(i, now_ns);
                s.call(i, Hook::Update, |n| n.update(dt)).is_ok()
            }
            Work::Pause(_) => s.call(i, Hook::Pause, N::pause).is_ok(),
            Work::Resume => s.call(i, Hook::Resume, N::resume).is_ok(),
            // Their errors are ignored.
            Work::Exception => {
                let _ = s.invoke(i, Hook::Exception, N::exception);
                true
            }
            Work::Stop => {
                let _ = s.invoke(i, Hook::Stop, N::stop);
                true
            }
        }
    }

    /// Updates node `i`, the only one of the lane, at its targets from
    /// `target_ns` on, without the control lock, until the command changes or
    /// an update fails; then records the updates and puts the node back.
    ///
    /// The time used for an update is never later than the frozen time T of a
    /// pause whose request the node did not see. The controller stores the
    /// new control word and only then freezes the timer, both under the
    /// control lock. The node reads the time, then the word, and uses the time
    /// only if the word did not change. If the freeze came before the time was
    /// read, the timer's lock orders the controller's store before the node's
    /// load, which then sees the new word, and the node discards the time. So
    /// a time that is used was read before the freeze, and is at most T. A
    /// stop needs no such argument: the clock never reads later than its end.
    fn run_alone(&self, i: usize, mut target_ns: u64, mut c: Guard<'a>) -> Guard<'a> {
        let s = self.shared;
        c.sched.take_earliest(i);
        // Under the control lock: the word of the command `Run`.
        let word = s.word.load(Ordering::SeqCst);
        drop(c);
        let changed = || s.word.load(Ordering::SeqCst) != word;
        let period_ns = s.nodes[i].period_ns;
        let ok = loop {
            let now_ns = s.clock.now_ns();
            if changed() {
                break true;
            }
            if now_ns < target_ns {
                // Wake up on any control request too.
                s.clock.wait_until_ns_or(target_ns, Duration::MAX, changed);
                continue;
            }
            let dt = s.dt(i, now_ns);
            if s.call(i, Hook::Update, |n| n.update(dt)).is_err() {
                break false;
            }
            s.nodes[i].last_update_ns.store(now_ns, Ordering::SeqCst);
            target_ns = next_target_ns(target_ns, now_ns, period_ns);
        };
        let mut c = s.lock();
        c.life.record_free_run(i, target_ns, ok);
        c.sched.park(i);
        let next = s.advance(i, &mut c);
        // This worker looks at the queues next, so it wakes nobody.
        c.sched.place(i, next, !ok);
        c
    }

    /// Waits on the timer for `target_ns`, as the only worker of the lane that
    /// does, until the time reaches it, an earlier target or other work comes,
    /// or the command changes.
    fn keep_time(&self, target_ns: u64, mut c: Guard<'a>) -> Guard<'a> {
        let s = self.shared;
        let lane = &s.lanes[self.lane];
        c.sched.queues[self.lane].timekeeper = Some(target_ns);
        // Read under the control lock: a change after this bumps one of them.
        let epoch = lane.epoch.load(Ordering::SeqCst);
        let word = s.word.load(Ordering::SeqCst);
        drop(c);
        s.clock.wait_until_ns_or(target_ns, Duration::MAX, || {
            lane.epoch.load(Ordering::SeqCst) != epoch || s.word.load(Ordering::SeqCst) != word
        });
        let mut c = s.lock();
        c.sched.queues[self.lane].timekeeper = None;
        c
    }
}
