//! The lifecycle of a node: the one state machine that every node goes
//! through, whichever thread runs it.
//!
//! [`Lifecycle`] lives under the control lock. It decides which hook a node
//! runs next ([`Lifecycle::advance`]) and records how each one returned
//! ([`Lifecycle::record`]); it calls no hook and waits for nothing. A thread
//! that runs the nodes (`sched`) asks it what to do and does it, so the rules
//! of the lifecycle are written once: a pause ends with `resume` unless the
//! system stops meanwhile, a node leaves its loop only at its end, every node
//! calls `exception` before `stop` when the system stops because of one, and
//! the manual mode updates only on a step that moved the time.

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
fn control_word(epoch: u64, command: Command) -> u64 {
    epoch << 2 | command as u64
}

/// The target of a free-running node's next update, after an update at
/// `now_ns` for the target `target_ns`.
///
/// Each target is one period after the previous one, so that waking up late
/// does not delay the following updates. Only an update a period or more late
/// (already at or past the target after its own) starts counting again from its
/// own time, so the node never updates several times in a row to catch up.
/// With a period of 1 ns (the shortest), the target is 1 ns after the update.
pub(super) fn next_target_ns(target_ns: u64, now_ns: u64, period_ns: u64) -> u64 {
    let next_ns = target_ns.saturating_add(period_ns);
    if now_ns >= next_ns {
        now_ns.saturating_add(period_ns)
    } else {
        next_ns
    }
}

/// The Δt in seconds of an update at `now_ns` after one at `last_ns`: from the
/// nanosecond difference, not a difference of seconds, which would round
/// differently at every absolute time. `last_ns` is never later than
/// `now_ns`: both are times of the run's clock, which never goes back.
pub(super) fn dt(now_ns: u64, last_ns: u64) -> f64 {
    (now_ns - last_ns) as f64 / 1e9
}

/// Where a node is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// `start` is due.
    Start,
    /// Started; waits until every node has started (or the stop).
    WaitGo,
    /// In the update loop.
    Loop,
    /// Called `pause` for the pause request of this epoch.
    Paused(u64),
    /// Left the loop: `exception` (on a failure) and `stop` are left.
    Finish,
    /// `stop` returned.
    Done,
}

struct Life {
    phase: Phase,
    /// `exception` returned.
    excepted: bool,
    /// The time of the next free-running update (see [`next_target_ns`]).
    target_ns: u64,
}

/// A hook call that [`Lifecycle::advance`] decided on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Work {
    /// `start`, then `restore`.
    Start,
    Update {
        now_ns: u64,
    },
    Pause(u64),
    Resume,
    Exception,
    Stop,
}

/// What a node does next.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Next {
    Hook(Work),
    /// An update once the time reaches this target.
    Wait(u64),
    /// Nothing until the control state changes.
    Park,
}

/// The control state and the life of every node, under the control lock.
pub(super) struct Lifecycle {
    command: Command,
    /// Bumped on every change of `command`, so that a paused node can tell the
    /// end of its pause from a new pause request that came before it woke up.
    epoch: u64,
    /// The epoch of the last change to `Run`.
    run_epoch: u64,
    failed: bool,
    /// The nodes may enter their loops (set after every node started).
    go: bool,
    /// Manual mode: the node allowed to run one step. Cleared by its step.
    grant: Option<usize>,
    /// Every node may call `stop` (set once every node is ready to stop).
    stop_allowed: bool,
    manual: bool,
    nodes: Vec<Life>,
    /// Set by a change that the control thread waits for (a node started,
    /// paused, left its loop, called `exception`, or ended its step). Whoever
    /// makes the change notifies the control thread and clears it.
    changed: bool,
}

impl Lifecycle {
    /// `n` nodes, each with its `start` due.
    pub(super) fn new(n: usize, manual: bool) -> Self {
        Self {
            command: Command::Run,
            epoch: 0,
            run_epoch: 0,
            failed: false,
            go: false,
            grant: None,
            stop_allowed: false,
            manual,
            nodes: (0..n)
                .map(|_| Life {
                    phase: Phase::Start,
                    excepted: false,
                    target_ns: 0,
                })
                .collect(),
            changed: false,
        }
    }

    /// The command and its epoch, as [`control_word`] packs them.
    pub(super) fn word(&self) -> u64 {
        control_word(self.epoch, self.command)
    }

    /// Whether a change the control thread waits for came since the last
    /// call.
    pub(super) fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    pub(super) fn manual(&self) -> bool {
        self.manual
    }

    /// The nodes run freely: they update at their targets.
    pub(super) fn running_freely(&self) -> bool {
        !self.manual && self.command == Command::Run
    }

    pub(super) fn epoch(&self) -> u64 {
        self.epoch
    }

    pub(super) fn failed(&self) -> bool {
        self.failed
    }

    /// Changes the command. Returns whether it changed: nothing changes it
    /// once it is `Stop`.
    pub(super) fn set_command(&mut self, command: Command) -> bool {
        if self.command == Command::Stop || self.command == command {
            return false;
        }
        self.command = command;
        self.epoch += 1;
        if command == Command::Run {
            self.run_epoch = self.epoch;
        }
        true
    }

    pub(super) fn fail(&mut self) {
        self.failed = true;
    }

    /// Lets the started nodes enter their loops.
    pub(super) fn go(&mut self) {
        self.go = true;
    }

    pub(super) fn allow_stop(&mut self) {
        self.stop_allowed = true;
    }

    /// Manual mode: allows node `i` one step.
    pub(super) fn grant(&mut self, i: usize) {
        self.grant = Some(i);
    }

    /// The granted step has not ended.
    pub(super) fn granted(&self) -> bool {
        self.grant.is_some()
    }

    /// `start` returned on every node (successfully or not).
    pub(super) fn all_started(&self) -> bool {
        self.nodes.iter().all(|n| n.phase != Phase::Start)
    }

    /// Node `i` called `pause` for the pause request of `epoch`.
    pub(super) fn paused(&self, i: usize, epoch: u64) -> bool {
        self.nodes[i].phase == Phase::Paused(epoch)
    }

    /// Every node is out of its loop, and through `exception` if the system
    /// stops because of one.
    pub(super) fn ready_to_stop(&self) -> bool {
        (self.nodes.iter())
            .all(|n| matches!(n.phase, Phase::Finish | Phase::Done) && (!self.failed || n.excepted))
    }

    /// What node `i` does next under the control state. Makes the transitions
    /// that call no hook. `last_ns` is the node's last update, `period_ns` its
    /// period, and `now` reads the time in ns (only for a step of the manual
    /// mode).
    pub(super) fn advance(
        &mut self,
        i: usize,
        last_ns: u64,
        period_ns: u64,
        now: impl FnOnce() -> u64,
    ) -> Next {
        loop {
            let life = &mut self.nodes[i];
            match life.phase {
                Phase::Start => return Next::Hook(Work::Start),
                Phase::WaitGo if self.command == Command::Stop => self.leave_loop(i),
                Phase::WaitGo if !self.go => return Next::Park,
                Phase::WaitGo => {
                    life.target_ns = last_ns.saturating_add(period_ns);
                    life.phase = Phase::Loop;
                }
                Phase::Loop => match self.command {
                    Command::Stop => self.leave_loop(i),
                    Command::Pause => return Next::Hook(Work::Pause(self.epoch)),
                    Command::Run if !self.manual => return Next::Wait(life.target_ns),
                    Command::Run if self.grant != Some(i) => return Next::Park,
                    Command::Run => {
                        let now_ns = now();
                        if now_ns > last_ns {
                            return Next::Hook(Work::Update { now_ns });
                        }
                        // Nothing to do for a step without time passing.
                        self.end_step();
                        return Next::Park;
                    }
                },
                Phase::Paused(epoch) if self.epoch == epoch => return Next::Park,
                // A stop while paused ends the node without `resume`. But if the
                // pause was ended first and the stop came before this node woke
                // up, the node still resumes (and stops after), so that every
                // completed capture is followed by `resume` on every node.
                Phase::Paused(epoch) if self.run_epoch > epoch => {
                    return Next::Hook(Work::Resume);
                }
                Phase::Paused(_) => self.leave_loop(i),
                Phase::Finish if self.stop_allowed => return Next::Hook(Work::Stop),
                // Also after the node left the loop for a normal stop.
                Phase::Finish if self.failed && !life.excepted => {
                    return Next::Hook(Work::Exception);
                }
                Phase::Finish | Phase::Done => return Next::Park,
            }
        }
    }

    /// Records that `work` for node `i` returned (`ok` if without an error).
    /// A failed hook takes the node out of its loop.
    pub(super) fn record(&mut self, i: usize, work: Work, ok: bool, period_ns: u64) {
        if !ok {
            self.leave_loop(i);
            return;
        }
        let life = &mut self.nodes[i];
        match work {
            Work::Start => {
                life.phase = Phase::WaitGo;
                self.changed = true;
            }
            Work::Update { now_ns } => {
                life.target_ns = next_target_ns(life.target_ns, now_ns, period_ns);
                if self.manual {
                    self.end_step();
                }
            }
            Work::Pause(epoch) => {
                life.phase = Phase::Paused(epoch);
                self.changed = true;
            }
            Work::Resume => life.phase = Phase::Loop,
            Work::Exception => {
                life.excepted = true;
                self.changed = true;
            }
            Work::Stop => life.phase = Phase::Done,
        }
    }

    /// Records the free-running updates that node `i` ran on its own (see
    /// `sched`), as [`Self::record`] would have one by one: `target_ns` is the
    /// target after the last of them, and `ok` is false if the last one failed.
    pub(super) fn record_free_run(&mut self, i: usize, target_ns: u64, ok: bool) {
        self.nodes[i].target_ns = target_ns;
        if !ok {
            self.leave_loop(i);
        }
    }

    /// Node `i` left its loop and will call no hook but `exception` and `stop`.
    fn leave_loop(&mut self, i: usize) {
        self.nodes[i].phase = Phase::Finish;
        self.changed = true;
    }

    /// Manual mode: the step of the granted node is over.
    fn end_step(&mut self) {
        self.grant = None;
        self.changed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 10;

    #[test]
    fn late_by_less_than_a_period_keeps_the_grid() {
        // Target P, updated at 1.5 P: the next target is 2 P, not 2.5 P.
        assert_eq!(next_target_ns(P, 3 * P / 2, P), 2 * P);
    }

    #[test]
    fn late_by_more_than_a_period_counts_again_from_the_update() {
        // Target P, updated at 2.5 P: 3.5 P, not 3 P on the grid.
        assert_eq!(next_target_ns(P, 5 * P / 2, P), 7 * P / 2);
    }

    #[test]
    fn exactly_a_period_late_counts_again_from_the_update() {
        // Target 2 P, updated at 3 P: 4 P, so that the next Δt is not 0.
        assert_eq!(next_target_ns(2 * P, 3 * P, P), 4 * P);
    }

    #[test]
    fn without_a_period_the_target_is_1_ns_after_the_update() {
        assert_eq!(next_target_ns(1, 1_000, 1), 1_001);
    }

    /// Node 0 of `n`, whose last update was at `LAST`, asked at time `NOW`.
    const LAST: u64 = 100;
    const NOW: u64 = 105;

    fn advance(life: &mut Lifecycle) -> Next {
        life.advance(0, LAST, P, || NOW)
    }

    /// Runs what `advance` decides for node 0, successfully, until it waits.
    fn settle(life: &mut Lifecycle) -> Next {
        loop {
            match advance(life) {
                Next::Hook(work) => life.record(0, work, true, P),
                next => return next,
            }
        }
    }

    /// One node, started and let into its loop.
    fn in_loop(manual: bool) -> Lifecycle {
        let mut life = Lifecycle::new(1, manual);
        assert_eq!(advance(&mut life), Next::Hook(Work::Start));
        life.record(0, Work::Start, true, P);
        assert_eq!(advance(&mut life), Next::Park);
        assert!(life.all_started());
        life.go();
        life
    }

    #[test]
    fn the_first_target_is_a_period_after_the_last_update() {
        let mut life = in_loop(false);
        assert_eq!(advance(&mut life), Next::Wait(LAST + P));
        life.record(0, Work::Update { now_ns: LAST + P }, true, P);
        assert_eq!(advance(&mut life), Next::Wait(LAST + 2 * P));
    }

    #[test]
    fn a_stop_before_the_go_leaves_the_node_waiting_for_the_stop() {
        let mut life = Lifecycle::new(1, false);
        life.record(0, Work::Start, true, P);
        life.set_command(Command::Stop);
        assert_eq!(advance(&mut life), Next::Park);
        assert!(life.ready_to_stop());
        life.allow_stop();
        assert_eq!(advance(&mut life), Next::Hook(Work::Stop));
    }

    #[test]
    fn a_failure_calls_exception_before_stop() {
        let mut life = in_loop(false);
        life.record(0, Work::Update { now_ns: NOW }, false, P);
        life.fail();
        life.set_command(Command::Stop);
        assert!(!life.ready_to_stop());
        assert_eq!(advance(&mut life), Next::Hook(Work::Exception));
        life.record(0, Work::Exception, true, P);
        assert_eq!(advance(&mut life), Next::Park);
        assert!(life.ready_to_stop());
        life.allow_stop();
        assert_eq!(advance(&mut life), Next::Hook(Work::Stop));
        life.record(0, Work::Stop, true, P);
        assert_eq!(advance(&mut life), Next::Park);
    }

    #[test]
    fn a_pause_ends_with_resume() {
        let mut life = in_loop(false);
        life.set_command(Command::Pause);
        let epoch = life.epoch();
        assert_eq!(advance(&mut life), Next::Hook(Work::Pause(epoch)));
        life.record(0, Work::Pause(epoch), true, P);
        assert!(life.paused(0, epoch));
        assert_eq!(advance(&mut life), Next::Park);
        life.set_command(Command::Run);
        assert_eq!(advance(&mut life), Next::Hook(Work::Resume));
        life.record(0, Work::Resume, true, P);
        assert_eq!(advance(&mut life), Next::Wait(LAST + P));
    }

    #[test]
    fn a_stop_while_paused_ends_without_resume() {
        let mut life = in_loop(false);
        life.set_command(Command::Pause);
        assert_eq!(settle(&mut life), Next::Park);
        life.set_command(Command::Stop);
        assert_eq!(advance(&mut life), Next::Park);
        assert!(life.ready_to_stop());
    }

    #[test]
    fn a_stop_after_the_pause_ended_still_resumes_first() {
        let mut life = in_loop(false);
        life.set_command(Command::Pause);
        assert_eq!(settle(&mut life), Next::Park);
        // The node did not wake up between the two changes.
        life.set_command(Command::Run);
        life.set_command(Command::Stop);
        assert_eq!(advance(&mut life), Next::Hook(Work::Resume));
        life.record(0, Work::Resume, true, P);
        assert_eq!(advance(&mut life), Next::Park);
        assert!(life.ready_to_stop());
    }

    #[test]
    fn a_manual_step_updates_once_if_the_time_moved() {
        let mut life = in_loop(true);
        assert_eq!(advance(&mut life), Next::Park);
        life.grant(0);
        assert_eq!(advance(&mut life), Next::Hook(Work::Update { now_ns: NOW }));
        life.record(0, Work::Update { now_ns: NOW }, true, P);
        assert!(!life.granted());
        assert_eq!(advance(&mut life), Next::Park);
    }

    #[test]
    fn a_manual_step_without_time_passing_ends_without_an_update() {
        let mut life = in_loop(true);
        assert_eq!(advance(&mut life), Next::Park);
        life.grant(0);
        assert_eq!(life.advance(0, NOW, P, || NOW), Next::Park);
        assert!(!life.granted());
    }

    #[test]
    fn a_failed_free_run_leaves_the_loop() {
        let mut life = in_loop(false);
        assert_eq!(advance(&mut life), Next::Wait(LAST + P));
        life.record_free_run(0, LAST + 3 * P, true);
        assert_eq!(advance(&mut life), Next::Wait(LAST + 3 * P));
        life.record_free_run(0, LAST + 4 * P, false);
        life.set_command(Command::Stop);
        assert!(life.ready_to_stop());
    }
}
