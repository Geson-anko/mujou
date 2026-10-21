use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Instant;

use super::*;

const P_NS: u64 = 10 * H_NS;

/// Sends each Δt (in ns) as it updates, and blocks in the update (or in
/// `resume`, for `hold_resume`) until its receiver gets a message.
struct Ticker {
    dts: Sender<u64>,
    hold: Option<Receiver<()>>,
    hold_resume: Option<Receiver<()>>,
}

impl Ticker {
    fn new(dts: Sender<u64>) -> Self {
        Self {
            dts,
            hold: None,
            hold_resume: None,
        }
    }
}

impl NodeHooks for Ticker {
    type State = ();

    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        let _ = self.dts.send((dt * 1e9).round() as u64);
        if let Some(hold) = &self.hold {
            let _ = hold.recv();
        }
        Ok(())
    }

    fn resume(&mut self) -> Result<(), HookError> {
        if let Some(hold) = &self.hold_resume {
            let _ = hold.recv();
        }
        Ok(())
    }

    fn capture(&mut self) -> Result<(), HookError> {
        Ok(())
    }
}

/// Tickers of these periods (ns), the first holding on `hold`, and their Δts.
fn tickers(
    periods_ns: &[u64],
    mut hold: Option<Receiver<()>>,
) -> (RunConfig<Ticker>, Vec<Receiver<u64>>) {
    let (tickers, dts): (Vec<_>, Vec<_>) = (periods_ns.iter())
        .map(|_| {
            let (dts, rx) = mpsc::channel();
            let hold = hold.take();
            (
                Ticker {
                    hold,
                    ..Ticker::new(dts)
                },
                rx,
            )
        })
        .unzip();
    (with_periods(config(tickers), periods_ns), dts)
}

/// One ticker of this period (ns), holding on `hold`, and its Δts.
fn ticker(period_ns: u64, hold: Option<Receiver<()>>) -> (RunConfig<Ticker>, Receiver<u64>) {
    let (config, mut dts) = tickers(&[period_ns], hold);
    (config, dts.remove(0))
}

fn start(mode: Mode, timer: &Arc<Timer>, config: RunConfig<Ticker>) -> Runtime<Ticker> {
    let (rt, started) = mode.start(config, Arc::clone(timer)).unwrap();
    started.wait().unwrap();
    rt
}

fn assert_quiet(dts: &Receiver<u64>) {
    assert_eq!(dts.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
}

#[test]
fn waits_for_the_period_and_then_updates_once_with_dt_equal_to_it() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = ticker(P_NS, None);
        let rt = start(mode, &timer, config);
        timer.advance_ns(P_NS - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        assert_quiet(&dts);
        // In steps: what counts is the time since the last update.
        timer.advance_ns(P_NS / 2).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(P_NS / 2).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        shut_down(&rt).unwrap();
    });
}

#[test]
fn late_by_less_than_a_period_the_next_target_stays_one_period_after_the_last() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = ticker(P_NS, None);
        let rt = start(mode, &timer, config);
        // Wakes up half a period late: one update.
        timer.advance_ns(3 * P_NS / 2).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(3 * P_NS / 2));
        assert_quiet(&dts);
        // The next target is 2 P, not 2.5 P: the lateness is not carried on.
        timer.advance_ns(P_NS / 2 - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS / 2));
        assert_quiet(&dts);
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        shut_down(&rt).unwrap();
    });
}

#[test]
fn late_by_a_period_or_more_it_updates_once_and_counts_again_from_there() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = ticker(P_NS, None);
        let rt = start(mode, &timer, config);
        // 1.5 P late: one update, not one per missed target.
        timer.advance_ns(5 * P_NS / 2).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(5 * P_NS / 2));
        assert_quiet(&dts);
        // The next target is 3.5 P, not 3 P on the grid of multiples of P.
        timer.advance_ns(P_NS - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        shut_down(&rt).unwrap();
    });
}

#[test]
fn exactly_a_period_late_counts_again_from_the_late_update() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (release, hold) = mpsc::channel();
        let (config, dts) = ticker(P_NS, Some(hold));
        let rt = start(mode, &timer, config);
        timer.advance_ns(P_NS / 2).unwrap();
        timer.advance_ns(P_NS / 2).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        // The update at P takes 2 P, so the next one comes at 3 P, exactly a
        // period after its target 2 P. It must count again (target 4 P):
        // otherwise the target would stay at 3 P and the node would update
        // again at once with Δt = 0.
        timer.advance_ns(2 * P_NS).unwrap();
        release.send(()).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(2 * P_NS));
        release.send(()).unwrap();
        timer.advance_ns(P_NS - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        release.send(()).unwrap();
        shut_down(&rt).unwrap();
    });
}

#[test]
fn an_update_longer_than_the_period_is_followed_at_once_without_catching_up() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (release, hold) = mpsc::channel();
        let (config, dts) = ticker(P_NS, Some(hold));
        let rt = start(mode, &timer, config);
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        // The update takes 2.5 P.
        timer.advance_ns(5 * P_NS / 2).unwrap();
        release.send(()).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(5 * P_NS / 2));
        release.send(()).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(P_NS - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        release.send(()).unwrap();
        shut_down(&rt).unwrap();
    });
}

#[test]
fn a_capture_does_not_move_the_target() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = ticker(P_NS, None);
        let rt = start(mode, &timer, config);
        timer.advance_ns(P_NS / 2).unwrap();
        rt.capture(|states: Vec<()>| states.len(), PAUSE)
            .wait()
            .unwrap();
        timer.advance_ns(P_NS / 2 - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        assert_quiet(&dts);
        shut_down(&rt).unwrap();
    });
}

#[test]
fn a_restored_node_counts_its_period_from_its_last_update() {
    each_mode(|mode| {
        let at_ns = 100 * P_NS;
        let timer = Arc::new(Timer::new(Source::Virtual, Timestamp::from_ns(at_ns)));
        let (mut config, dts) = ticker(P_NS, None);
        config.nodes[0].last_update_ns = Some(at_ns - P_NS / 2);
        let rt = start(mode, &timer, config);
        timer.advance_ns(P_NS / 2 - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        shut_down(&rt).unwrap();
    });
}

#[test]
fn each_node_keeps_its_own_period() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = tickers(&[P_NS, 3 * P_NS], None);
        let rt = start(mode, &timer, config);
        for _ in 0..2 {
            timer.advance_ns(P_NS).unwrap();
            assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
            assert_quiet(&dts[1]);
        }
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
        assert_eq!(dts[1].recv_timeout(LONG), Ok(3 * P_NS));
        shut_down(&rt).unwrap();
    });
}

/// A worker that puts a node back with a target earlier than the one the
/// timekeeper waits for must hand that target to it: no one else wakes the
/// timekeeper for it.
#[test]
fn a_target_a_worker_puts_back_before_the_timekeepers_is_kept() {
    // Needs a second worker to keep the time while the first one updates.
    in_modes(&[Mode::TwoWorkers], |mode| {
        let timer = virtual_timer();
        let (release, hold) = mpsc::channel();
        let (config, dts) = tickers(&[P_NS, 1_000 * P_NS], Some(hold));
        let rt = start(mode, &timer, config);
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
        // Meanwhile the other worker waits for the far target of node 1.
        thread::sleep(QUIET);
        release.send(()).unwrap();
        // Node 0 is back, with its target at 2 P, before the time gets there.
        thread::sleep(QUIET);
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
        release.send(()).unwrap();
        assert_quiet(&dts[1]);
        shut_down(&rt).unwrap();
    });
}

/// A worker that puts a node back and then runs a hook from the ready
/// queue must not leave the timekeeper waiting for a later target: the
/// node's update is due while the hook blocks.
#[test]
fn a_blocking_resume_does_not_delay_another_nodes_update() {
    in_modes(&[Mode::TwoWorkers], |mode| {
        let timer = virtual_timer();
        let (release, hold) = mpsc::channel();
        let (mut config, dts) = tickers(&[P_NS, 1_000 * P_NS, P_NS], None);
        config.nodes[2].hooks.hold_resume = Some(hold);
        let rt = start(mode, &timer, config);
        // Every node gets `resume`; that of node 2 blocks a worker.
        rt.capture(|states: Vec<()>| states.len(), PAUSE)
            .wait()
            .unwrap();
        thread::sleep(QUIET);
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
        release.send(()).unwrap();
        shut_down(&rt).unwrap();
    });
}

#[test]
fn with_a_period_of_1_ns_any_time_passing_is_enough() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = ticker(1, None);
        let rt = start(mode, &timer, config);
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(1));
        shut_down(&rt).unwrap();
    });
}

#[test]
fn the_manual_mode_ignores_the_period() {
    each_mode(|mode| {
        let timer = virtual_timer();
        let (config, dts) = ticker(P_NS, None);
        let rt = start(
            mode,
            &timer,
            RunConfig {
                manual: true,
                ..config
            },
        );
        timer.advance_ns(1).unwrap();
        rt.step(None).wait().unwrap();
        assert_eq!(dts.try_recv(), Ok(1));
        shut_down(&rt).unwrap();
    });
}

#[test]
fn a_stop_or_a_capture_wakes_the_wait_for_the_period() {
    each_mode(|mode| {
        let timer = Arc::new(Timer::new(Source::Real, Timestamp::from_ns(0)));
        let period = 1_000 * 1_000_000_000;
        let (config, dts) = ticker(period, None);
        let rt = start(mode, &timer, config);
        let begin = Instant::now();
        rt.capture(|states: Vec<()>| states.len(), PAUSE)
            .wait()
            .unwrap();
        rt.stop();
        assert!(rt.join(Some(LONG)).unwrap());
        assert!(begin.elapsed() < Duration::from_secs(2));
        assert!(dts.try_recv().is_err());
    });
}

/// How long the real-clock tests run, in real time.
const RUN: Duration = Duration::from_secs(1);

/// Runs a ticker of period `period_ns` (on the timer) on the real clock at
/// `scale` for `RUN` and returns its Δts and the real time from before its
/// start to after its end.
///
/// The real-clock tests keep the period at 200 ms of real time: a loaded
/// runner (e.g. GitHub's macOS) may wake a thread about 100 ms late every
/// time, and only a wake-up less than a period late keeps the targets from
/// slipping (a later one counts again from itself).
fn tick_for(mode: Mode, period_ns: u64, scale: f64) -> (Vec<u64>, Duration) {
    let timer = Arc::new(Timer::new(Source::Real, Timestamp::from_ns(0)));
    timer.set_time_scale(scale).unwrap();
    let begin = Instant::now();
    let (config, dts) = ticker(period_ns, None);
    let rt = start(mode, &timer, config);
    thread::sleep(RUN);
    shut_down(&rt).unwrap();
    let elapsed = begin.elapsed();
    (dts.try_iter().collect(), elapsed)
}

/// The node waits for its period on the real clock: neither faster (free
/// running would update far more often) nor drifting behind.
#[test]
fn on_the_real_clock_it_updates_about_once_per_period() {
    // One node: the default pool has a single worker too.
    in_modes(&[Mode::Dedicated, Mode::OneWorker], |mode| {
        let period_ns = 200 * H_NS;
        let (dts, elapsed) = tick_for(mode, period_ns, 1.0);
        // The targets are P apart, so the updates cannot outnumber them.
        let at_most = elapsed.as_nanos() as f64 / period_ns as f64;
        assert!((dts.len() as f64) <= at_most, "{} > {at_most}", dts.len());
        // The targets at 200, 400, 600 and 800 ms come before the end at 1 s
        // as long as each wake-up is less than 200 ms late.
        assert!(dts.len() >= 4, "{} updates", dts.len());
    });
}

/// The period is on the timer, so the time scale changes the updates per
/// real second: a period of 800 ms at scale 4 is 200 ms of real time.
#[test]
fn the_time_scale_scales_the_updates_per_real_second() {
    // One node: the default pool has a single worker too.
    in_modes(&[Mode::Dedicated, Mode::OneWorker], |mode| {
        let (period_ns, scale) = (800 * H_NS, 4.0);
        let (dts, elapsed) = tick_for(mode, period_ns, scale);
        let at_most = elapsed.as_nanos() as f64 * scale / period_ns as f64;
        assert!((dts.len() as f64) <= at_most, "{} > {at_most}", dts.len());
        // 1 s of real time is 4 s on the timer: targets at 0.8, 1.6, 2.4, 3.2
        // and 4 s. The first four come before the end as long as each wake-up
        // is less than 200 ms (a period) of real time late. At scale 1 the same
        // second would hold one target, so 4 shows that the scale applies.
        assert!(dts.len() >= 4, "{} updates", dts.len());
    });
}
