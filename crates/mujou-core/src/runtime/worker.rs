//! The threads that run the nodes: one loop ([`Worker::run`]) for a worker of
//! the pool and for the thread of a node given one of its own.
//!
//! A worker takes a piece of work for one node of its lane under the control
//! lock (`sched`), runs the hook outside every lock, and puts the node back.
//! What the node does next is the [`Lifecycle`](super::lifecycle::Lifecycle)'s
//! decision. Only the worker that took a node puts it back.
//!
//! A worker never waits for one node, which would deadlock with fewer workers
//! than nodes: it waits for work, and the one worker of a lane that keeps the
//! time for the earliest target (the timekeeper) waits on the timer. A lane of
//! one node needs no queue while the nodes run freely, so its worker updates it
//! without the control lock until the command changes ([`Worker::run_alone`]).
//! A worker whose hook failed tells the control thread, which ends the system.

use std::sync::MutexGuard;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::time::Duration;

use super::control::Message;
use super::error::{Hook, HookError};
use super::hooks::NodeHooks;
use super::lifecycle::{Next, Work, next_target_ns};
use super::shared::{Control, Shared};

/// A thread that runs the nodes of one lane.
pub(super) struct Worker<'a, N: NodeHooks> {
    pub(super) shared: &'a Shared<N>,
    pub(super) lane: usize,
    pub(super) to_controller: Sender<Message<N>>,
}

type Guard<'a> = MutexGuard<'a, Control>;

impl<'a, N: NodeHooks> Worker<'a, N> {
    /// Runs the nodes of the lane until all of them have stopped.
    pub(super) fn run(&self) {
        let s = self.shared;
        let mut c = s.lock();
        while c.sched.left(self.lane) > 0 {
            // Work in the ready queue means the timekeeper, if any, is already
            // woken: the control side wakes it when it queues work, a worker
            // takes the hook it queues itself under the same lock, and a worker
            // keeps the time only with the queue empty. So a target this worker
            // just put back is not missed while it runs a hook first.
            if let Some(i) = c.sched.pop_ready(self.lane) {
                if let Some(work) = self.take(i, &mut c) {
                    c = self.run_work(i, work, c);
                }
                continue;
            }
            if let Some((target_ns, i)) = self.earliest(&mut c) {
                if c.sched.alone(self.lane) {
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
                match c.sched.timekeeper(self.lane) {
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

    /// Decides what node `i`, just taken out of the ready queue, does: the
    /// hook to run now, if any.
    fn take(&self, i: usize, c: &mut Control) -> Option<Work> {
        match self.shared.advance(i, c) {
            Next::Hook(work) => {
                c.sched.run(i);
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
        if c.sched.left_behind(self.lane) {
            // Leave what is left (and the timekeeping) to another worker.
            s.lanes[self.lane].changed.notify_one();
        }
        drop(c);
        let ok = self.run_hook(i, work);
        let mut c = s.lock();
        s.record(i, work, ok, &mut c);
        self.put_back(i, ok, &mut c);
        c
    }

    /// Puts node `i` back where its lifecycle says, after its hook returned.
    /// A failed hook is followed by `exception` ahead of the nodes the failure
    /// woke, as right after catching it. This worker looks at the queues next,
    /// so it wakes nobody.
    fn put_back(&self, i: usize, ok: bool, c: &mut Control) {
        c.sched.park(i);
        let next = self.shared.advance(i, c);
        c.sched.place(i, next, !ok);
    }

    /// Runs the hook of `work` on node `i`. Returns false if it failed.
    fn run_hook(&self, i: usize, work: Work) -> bool {
        let s = self.shared;
        match work {
            Work::Start => self
                .call(i, Hook::Start, N::start)
                .and_then(|()| self.call(i, Hook::Restore, N::restore))
                .is_ok(),
            Work::Update { now_ns } => self.update(i, now_ns),
            Work::Pause(_) => self.call(i, Hook::Pause, N::pause).is_ok(),
            Work::Resume => self.call(i, Hook::Resume, N::resume).is_ok(),
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

    /// Updates node `i` at `now_ns`. Returns false if it failed.
    fn update(&self, i: usize, now_ns: u64) -> bool {
        let dt = self.shared.dt(i, now_ns);
        self.call(i, Hook::Update, |n| n.update(dt)).is_ok()
    }

    /// Calls a hook whose error ends the system, and tells the control thread
    /// if it failed.
    fn call<T>(
        &self,
        i: usize,
        hook: Hook,
        f: impl FnOnce(&mut N) -> Result<T, HookError>,
    ) -> Result<T, ()> {
        let result = self.shared.call(i, hook, f);
        if result.is_err() {
            // The control thread is gone only after the system finished.
            let _ = self.to_controller.send(Message::Failed);
        }
        result
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
        let node = &s.nodes[i];
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
            if !self.update(i, now_ns) {
                break false;
            }
            node.last_update_ns.store(now_ns, Ordering::SeqCst);
            target_ns = next_target_ns(target_ns, now_ns, node.period_ns);
        };
        let mut c = s.lock();
        c.life.record_free_run(i, target_ns, ok);
        self.put_back(i, ok, &mut c);
        c
    }

    /// Waits on the timer for `target_ns`, as the only worker of the lane that
    /// does, until the time reaches it, an earlier target or other work comes,
    /// or the command changes.
    fn keep_time(&self, target_ns: u64, mut c: Guard<'a>) -> Guard<'a> {
        let s = self.shared;
        let lane = &s.lanes[self.lane];
        c.sched.set_timekeeper(self.lane, Some(target_ns));
        // Read under the control lock: a change after this bumps one of them.
        let epoch = lane.epoch.load(Ordering::SeqCst);
        let word = s.word.load(Ordering::SeqCst);
        drop(c);
        s.clock.wait_until_ns_or(target_ns, Duration::MAX, || {
            lane.epoch.load(Ordering::SeqCst) != epoch || s.word.load(Ordering::SeqCst) != word
        });
        let mut c = s.lock();
        c.sched.set_timekeeper(self.lane, None);
        c
    }
}
