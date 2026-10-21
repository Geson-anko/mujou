//! The queues the threads of a runtime take the work from, under the control
//! lock: which node waits where, lane by lane.
//!
//! The nodes are split into lanes: the pool, whose M nodes share N worker
//! threads, and one lane per node given a dedicated thread, which is a worker
//! of its own (`worker`). Each lane has a ready queue of the nodes with a
//! hook to run, in the order they became due except that a node whose hook
//! failed goes ahead with its `exception`, and a heap of the free-running
//! updates, earliest target first and by index on a tie. A node is taken only
//! from a queue, under the lock, so no node runs on two threads at once.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::sync::Condvar;
use std::sync::atomic::AtomicU64;

use super::lifecycle::Next;

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

    /// The nodes of `lane` that have not finished `stop`.
    pub(super) fn left(&self, lane: usize) -> usize {
        self.queues[lane].left
    }

    /// `lane` has one node, which its worker may update without the lock.
    pub(super) fn alone(&self, lane: usize) -> bool {
        self.queues[lane].nodes == 1
    }

    /// The target the timekeeper of `lane` waits for, if a worker waits on
    /// the timer.
    pub(super) fn timekeeper(&self, lane: usize) -> Option<u64> {
        self.queues[lane].timekeeper
    }

    pub(super) fn set_timekeeper(&mut self, lane: usize, target_ns: Option<u64>) {
        self.queues[lane].timekeeper = target_ns;
    }

    /// `lane` has queued work and nobody keeps its time: another worker could
    /// take that work.
    pub(super) fn left_behind(&self, lane: usize) -> bool {
        let queue = &self.queues[lane];
        queue.timekeeper.is_none() && !(queue.ready.is_empty() && queue.heap.is_empty())
    }

    /// Takes the first node of the ready queue of `lane`, parked until its
    /// worker decides what it does ([`Self::run`] or [`Self::place`]).
    pub(super) fn pop_ready(&mut self, lane: usize) -> Option<usize> {
        let i = self.queues[lane].ready.pop_front()?;
        self.park(i);
        Some(i)
    }

    /// A worker runs a hook of node `i`, which it took.
    pub(super) fn run(&mut self, i: usize) {
        self.seats[i].at = At::Running;
    }

    /// The earliest free-running update of `lane`. Drops the stale entries on
    /// the way.
    pub(super) fn earliest(&mut self, lane: usize) -> Option<(u64, usize)> {
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
    pub(super) fn take_earliest(&mut self, i: usize) {
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
