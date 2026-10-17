//! The thread of a node: start, the update loop, pauses, and the end.

use std::sync::atomic::Ordering;
use std::time::Duration;

use super::{NodeHooks, RUNTIME_OF_THREAD, Shared};

/// The instruction from the control thread to the nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Command {
    Run = 0,
    Pause = 1,
    /// Final: nothing changes the command once it is `Stop`.
    Stop = 2,
}

/// The command and its epoch in one word, so that a node can read both with
/// one atomic load: `epoch << 2 | command`.
pub(super) fn control_word(epoch: u64, command: Command) -> u64 {
    epoch << 2 | command as u64
}

fn is_run(word: u64) -> bool {
    word & 3 == Command::Run as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ExceptionHook {
    #[default]
    NotCalled,
    Calling,
    Done,
}

/// What a node reports to the control thread.
#[derive(Debug, Default)]
pub(super) struct Report {
    /// `start` returned (successfully or not).
    pub(super) started: bool,
    /// The node called `pause` for the pause request of this epoch and waits.
    pub(super) paused_at: Option<u64>,
    /// The node left its loop and will call no hook but `exception` and `stop`.
    pub(super) exited: bool,
    pub(super) exception: ExceptionHook,
}

impl Report {
    /// Ready for `stop`: out of the loop, and through `exception` if the system
    /// stops because of one.
    pub(super) fn ready_to_stop(&self, failed: bool) -> bool {
        self.exited && (!failed || self.exception == ExceptionHook::Done)
    }
}

enum Boundary {
    Exit,
    Again,
    Update { now_ns: u64, word: u64 },
}

impl<N: NodeHooks> Shared<N> {
    /// The thread of node `i`.
    pub(super) fn run_node(&self, i: usize) {
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
    /// The time read here is never later than the frozen time T of a pause
    /// whose request the node did not see. The controller stores the new
    /// control word and only then freezes the timer, both under the control
    /// lock. The node reads the word, then the time, then the word again, and
    /// uses the time only if the word did not change. If the freeze came before
    /// the time was read, the timer's lock orders the controller's store before
    /// the node's second load, which then sees the new word, and the node
    /// discards the time. So a time that is used was read before the freeze,
    /// and is at most T.
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
