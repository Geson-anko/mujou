use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mujou_core::runtime::{HookError, NodeHooks, Runtime, RuntimeConfig, RuntimeError};
use mujou_core::time::Timer;

const H_NS: u64 = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Start,
    Update(f64),
    Pause,
    Resume,
    Exception,
    Stop,
    StateDict,
}

#[derive(Debug)]
struct Boom;

impl fmt::Display for Boom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("boom")
    }
}

impl std::error::Error for Boom {}

/// Records its hook calls; fails (or panics) on the `n`-th update if asked.
struct Recorder {
    id: usize,
    log: Arc<Mutex<Vec<(usize, Event)>>>,
    updates: usize,
    fail_on_update: Option<usize>,
    panic_on_update: Option<usize>,
}

impl Recorder {
    fn record(&self, event: Event) {
        self.log.lock().unwrap().push((self.id, event));
    }
}

impl NodeHooks for Recorder {
    type State = usize;

    fn start(&mut self) -> Result<(), HookError> {
        self.record(Event::Start);
        Ok(())
    }

    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.record(Event::Update(dt));
        self.updates += 1;
        if self.panic_on_update == Some(self.updates) {
            panic!("panic in node {}", self.id);
        }
        if self.fail_on_update == Some(self.updates) {
            return Err(HookError {
                hook: "on_update",
                source: Arc::new(Boom),
            });
        }
        Ok(())
    }

    fn pause(&mut self) -> Result<(), HookError> {
        self.record(Event::Pause);
        Ok(())
    }

    fn resume(&mut self) -> Result<(), HookError> {
        self.record(Event::Resume);
        Ok(())
    }

    fn exception(&mut self) -> Result<(), HookError> {
        self.record(Event::Exception);
        Ok(())
    }

    fn stop(&mut self) -> Result<(), HookError> {
        self.record(Event::Stop);
        Ok(())
    }

    fn state_dict(&mut self) -> Result<usize, HookError> {
        self.record(Event::StateDict);
        Ok(self.updates)
    }
}

/// A virtual timer of its own: `cargo test` runs the tests in parallel threads.
fn virtual_timer() -> Arc<Timer> {
    let timer = Arc::new(Timer::new());
    timer.use_virtual_clock(true);
    timer
}

struct Setup {
    log: Arc<Mutex<Vec<(usize, Event)>>>,
    nodes: Vec<Recorder>,
    names: Vec<String>,
}

fn recorders(n: usize) -> Setup {
    let log = Arc::new(Mutex::new(Vec::new()));
    let nodes = (0..n)
        .map(|id| Recorder {
            id,
            log: Arc::clone(&log),
            updates: 0,
            fail_on_update: None,
            panic_on_update: None,
        })
        .collect();
    let names = (0..n).map(|i| format!("n{i}")).collect();
    Setup { log, nodes, names }
}

fn manual() -> RuntimeConfig {
    RuntimeConfig {
        manual: true,
        ..RuntimeConfig::default()
    }
}

fn events_of(log: &Mutex<Vec<(usize, Event)>>, id: usize) -> Vec<Event> {
    let log = log.lock().unwrap();
    log.iter()
        .filter(|(i, _)| *i == id)
        .map(|(_, e)| e.clone())
        .collect()
}

mod manual_mode {
    use super::*;

    #[test]
    fn steps_in_the_given_order_with_the_advanced_dt() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(3);
        let t0 = timer.time_ns();
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        for _ in 0..2 {
            timer.advance_ns(H_NS).unwrap();
            rt.step(Some(&[2, 0, 1])).wait().unwrap();
        }
        rt.shutdown().wait().unwrap();

        let order: Vec<usize> = log.lock().unwrap().iter().map(|(i, _)| *i).collect();
        // 3 starts in any order, then the updates in the given order, then 3 stops.
        assert_eq!(&order[3..9], &[2, 0, 1, 2, 0, 1]);
        for id in 0..3 {
            let dt = H_NS as f64 / 1e9;
            let expected = [
                Event::Start,
                Event::Update(dt),
                Event::Update(dt),
                Event::Stop,
            ];
            assert_eq!(events_of(&log, id), expected);
        }
        assert_eq!(rt.last_update_ns(), vec![t0 + 2 * H_NS; 3]);
    }

    #[test]
    fn a_step_without_time_passing_does_not_update() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(1);
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        rt.step(None).wait().unwrap();
        rt.shutdown().wait().unwrap();
        assert_eq!(events_of(&log, 0), [Event::Start, Event::Stop]);
    }

    #[test]
    fn step_needs_the_manual_mode() {
        let timer = virtual_timer();
        let Setup { nodes, names, .. } = recorders(1);
        let config = RuntimeConfig::default();
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        assert!(matches!(rt.step(None).wait(), Err(RuntimeError::NotManual)));
        rt.shutdown().wait().unwrap();
    }
}

mod capture {
    use super::*;

    #[test]
    fn builds_from_every_state_while_the_time_is_frozen() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(2);
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        let frozen = Arc::clone(&timer);
        let (states, paused) = rt
            .capture(move |states| (states, frozen.is_paused()))
            .wait()
            .unwrap();
        assert_eq!(states, vec![1, 1]);
        assert!(paused);
        assert!(!timer.is_paused());
        rt.shutdown().wait().unwrap();
        for id in 0..2 {
            let events = events_of(&log, id);
            assert_eq!(
                &events[2..5],
                &[Event::Pause, Event::StateDict, Event::Resume]
            );
        }
    }
}

mod failure {
    use super::*;

    fn assert_exception_then_stop_once(events: &[Event]) {
        let exception = events.iter().position(|e| *e == Event::Exception);
        let stop = events.iter().position(|e| *e == Event::Stop);
        assert_eq!(events.iter().filter(|e| **e == Event::Exception).count(), 1);
        assert_eq!(events.iter().filter(|e| **e == Event::Stop).count(), 1);
        let (exception, stop) = (exception.unwrap(), stop.unwrap());
        assert!(exception < stop);
        // Nothing but `stop` after `exception`.
        assert_eq!(events.len(), stop + 1);
    }

    #[test]
    fn an_update_error_ends_every_node_with_exception_then_stop() {
        let timer = virtual_timer();
        let Setup {
            log,
            mut nodes,
            names,
        } = recorders(3);
        nodes[1].fail_on_update = Some(3);
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        let mut result = Ok(());
        for _ in 0..3 {
            timer.advance_ns(H_NS).unwrap();
            result = rt.step(None).wait();
            if result.is_err() {
                break;
            }
        }
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.node, failure.hook), (1, "on_update"));
        assert!(rt.is_finished());
        for id in 0..3 {
            assert_exception_then_stop_once(&events_of(&log, id));
        }
        // Later requests report the same failure.
        assert!(matches!(rt.shutdown().wait(), Err(RuntimeError::Failed(_))));
        assert!(matches!(rt.join(None), Err(RuntimeError::Failed(_))));
    }

    #[test]
    fn a_panic_is_a_failure() {
        let timer = virtual_timer();
        let Setup {
            log,
            mut nodes,
            names,
        } = recorders(2);
        nodes[0].panic_on_update = Some(1);
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        let result = rt.step(None).wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.node, failure.hook), (0, "on_update"));
        assert!(failure.error.to_string().contains("panic in node 0"));
        for id in 0..2 {
            assert_exception_then_stop_once(&events_of(&log, id));
        }
    }
}

mod restore {
    use super::*;
    use mujou_core::time::TimerState;

    fn restoring() -> RuntimeConfig {
        RuntimeConfig {
            timer_state: Some(TimerState {
                time_ns: 7 * H_NS,
                scale: 1.0,
            }),
            ..manual()
        }
    }

    #[test]
    fn the_time_is_not_restored_under_a_running_runtime() {
        let timer = virtual_timer();
        let Setup { nodes, names, .. } = recorders(1);
        let (running, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        let before = timer.time_ns();

        let Setup { log, nodes, names } = recorders(1);
        let result = Runtime::start(nodes, names, Arc::clone(&timer), restoring());
        assert!(matches!(result, Err(RuntimeError::InvalidArgument(_))));
        // Nothing started, and the running runtime's time did not move.
        assert!(log.lock().unwrap().is_empty());
        assert_eq!(timer.time_ns(), before);
        assert!(!timer.is_paused());

        // Once it has finished, the time can be restored.
        running.shutdown().wait().unwrap();
        let Setup { nodes, names, .. } = recorders(1);
        let (restored, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), restoring()).unwrap();
        started.wait().unwrap();
        assert_eq!(timer.time_ns(), 7 * H_NS);
        restored.shutdown().wait().unwrap();
    }
}

mod separate_timers {
    use super::*;
    use mujou_core::time::TimerState;

    #[test]
    fn each_runtime_measures_dt_on_its_own_timer() {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Setup {
            log: la,
            nodes,
            names,
        } = recorders(1);
        let (a, started) = Runtime::start(nodes, names, Arc::clone(&ta), manual()).unwrap();
        started.wait().unwrap();
        let Setup {
            log: lb,
            nodes,
            names,
        } = recorders(1);
        let (b, started) = Runtime::start(nodes, names, Arc::clone(&tb), manual()).unwrap();
        started.wait().unwrap();

        ta.advance_ns(2 * H_NS).unwrap();
        tb.advance_ns(5 * H_NS).unwrap();
        a.step(None).wait().unwrap();
        b.step(None).wait().unwrap();
        a.shutdown().wait().unwrap();
        b.shutdown().wait().unwrap();
        assert_eq!(events_of(&la, 0)[1], Event::Update(2.0 * H_NS as f64 / 1e9));
        assert_eq!(events_of(&lb, 0)[1], Event::Update(5.0 * H_NS as f64 / 1e9));
    }

    #[test]
    fn a_capture_freezes_only_its_own_timer() {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Setup { nodes, names, .. } = recorders(1);
        let (a, started) = Runtime::start(nodes, names, Arc::clone(&ta), manual()).unwrap();
        started.wait().unwrap();
        let Setup {
            log: lb,
            nodes,
            names,
        } = recorders(1);
        let (b, started) = Runtime::start(nodes, names, Arc::clone(&tb), manual()).unwrap();
        started.wait().unwrap();
        let b = Arc::new(b);

        // While `a` is paused, the runtime on the other timer keeps updating.
        let (own, other, other_rt) = (Arc::clone(&ta), Arc::clone(&tb), Arc::clone(&b));
        let (own_paused, other_paused) = a
            .capture(move |_| {
                other.advance_ns(H_NS).unwrap();
                other_rt.step(None).wait().unwrap();
                (own.is_paused(), other.is_paused())
            })
            .wait()
            .unwrap();
        assert!(own_paused);
        assert!(!other_paused);
        assert_eq!(events_of(&lb, 0)[1], Event::Update(H_NS as f64 / 1e9));
        a.shutdown().wait().unwrap();
        b.shutdown().wait().unwrap();
    }

    #[test]
    fn a_runtime_on_another_timer_does_not_block_a_restore() {
        let (ta, tb) = (virtual_timer(), virtual_timer());
        let Setup { nodes, names, .. } = recorders(1);
        let (running, started) = Runtime::start(nodes, names, Arc::clone(&ta), manual()).unwrap();
        started.wait().unwrap();
        let before = ta.time_ns();

        let config = RuntimeConfig {
            timer_state: Some(TimerState {
                time_ns: 7 * H_NS,
                scale: 1.0,
            }),
            ..manual()
        };
        let Setup { nodes, names, .. } = recorders(1);
        let (restored, started) = Runtime::start(nodes, names, Arc::clone(&tb), config).unwrap();
        started.wait().unwrap();
        assert_eq!(tb.time_ns(), 7 * H_NS);
        assert_eq!(ta.time_ns(), before);
        running.shutdown().wait().unwrap();
        restored.shutdown().wait().unwrap();
    }
}

mod stop_request {
    use std::sync::mpsc::{self, Receiver, Sender};

    use super::*;

    /// Tells when it enters an update, then blocks there until released.
    struct Gate {
        entered: Sender<()>,
        release: Receiver<()>,
        stopped: Arc<Mutex<bool>>,
    }

    impl NodeHooks for Gate {
        type State = ();

        fn start(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn update(&mut self, _dt: f64) -> Result<(), HookError> {
            let _ = self.entered.send(());
            let _ = self.release.recv();
            Ok(())
        }

        fn pause(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn resume(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn exception(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn stop(&mut self) -> Result<(), HookError> {
            *self.stopped.lock().unwrap() = true;
            Ok(())
        }

        fn state_dict(&mut self) -> Result<(), HookError> {
            Ok(())
        }
    }

    #[test]
    fn returns_while_a_node_is_still_updating_and_join_sees_the_end() {
        let timer = virtual_timer();
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let stopped = Arc::new(Mutex::new(false));
        let gate = Gate {
            entered: entered_tx,
            release: release_rx,
            stopped: Arc::clone(&stopped),
        };
        let config = RuntimeConfig::default();
        let (rt, started) = Runtime::start(
            vec![gate],
            vec!["gate".to_owned()],
            Arc::clone(&timer),
            config,
        )
        .unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        entered.recv_timeout(Duration::from_secs(5)).unwrap();

        // The node is blocked in its update, so a waiting stop would not return.
        rt.request_stop();
        assert!(!rt.join(Some(Duration::from_millis(50))).unwrap());
        assert!(!*stopped.lock().unwrap());

        release.send(()).unwrap();
        assert!(rt.join(Some(Duration::from_secs(5))).unwrap());
        assert!(*stopped.lock().unwrap());
        assert!(rt.is_finished());
    }

    #[test]
    fn does_nothing_once_the_system_has_ended() {
        let timer = virtual_timer();
        let Setup { log, nodes, names } = recorders(2);
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), manual()).unwrap();
        started.wait().unwrap();
        rt.request_stop();
        assert!(rt.join(Some(Duration::from_secs(5))).unwrap());
        rt.request_stop();
        rt.shutdown().wait().unwrap();
        for id in 0..2 {
            assert_eq!(events_of(&log, id), [Event::Start, Event::Stop]);
        }
    }
}

mod period {
    use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
    use std::thread;
    use std::time::Instant;

    use mujou_core::time::TimerState;

    use super::*;

    const P_NS: u64 = 10 * H_NS;
    /// How long to watch for an update that must not come.
    const QUIET: Duration = Duration::from_millis(50);
    const LONG: Duration = Duration::from_secs(5);

    /// Sends each Δt (in ns) as it updates, and blocks in the update while
    /// `hold` has a message for it to wait for.
    struct Ticker {
        dts: Sender<u64>,
        hold: Option<Receiver<()>>,
    }

    impl NodeHooks for Ticker {
        type State = ();

        fn start(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn update(&mut self, dt: f64) -> Result<(), HookError> {
            let _ = self.dts.send((dt * 1e9).round() as u64);
            if let Some(hold) = &self.hold {
                let _ = hold.recv();
            }
            Ok(())
        }

        fn pause(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn resume(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn exception(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn stop(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn state_dict(&mut self) -> Result<(), HookError> {
            Ok(())
        }
    }

    fn with_period(period_ns: Option<u64>, manual: bool) -> RuntimeConfig {
        RuntimeConfig {
            manual,
            period_ns: Some(vec![period_ns]),
            ..RuntimeConfig::default()
        }
    }

    /// Starts one ticker per entry of `config.period_ns` (one if it is `None`).
    fn start_tickers(
        timer: &Arc<Timer>,
        config: RuntimeConfig,
        mut hold: Option<Receiver<()>>,
    ) -> (Runtime<Ticker>, Vec<Receiver<u64>>) {
        let n = config.period_ns.as_ref().map_or(1, Vec::len);
        let (tickers, dts): (Vec<_>, Vec<_>) = (0..n)
            .map(|_| {
                let (dts, rx) = mpsc::channel();
                let hold = hold.take();
                (Ticker { dts, hold }, rx)
            })
            .unzip();
        let names = (0..n).map(|i| format!("ticker{i}")).collect();
        let (rt, started) = Runtime::start(tickers, names, Arc::clone(timer), config).unwrap();
        started.wait().unwrap();
        (rt, dts)
    }

    fn start_ticker(
        timer: &Arc<Timer>,
        config: RuntimeConfig,
        hold: Option<Receiver<()>>,
    ) -> (Runtime<Ticker>, Receiver<u64>) {
        let (rt, mut dts) = start_tickers(timer, config, hold);
        (rt, dts.remove(0))
    }

    fn assert_quiet(dts: &Receiver<u64>) {
        assert_eq!(dts.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
    }

    #[test]
    fn waits_for_the_period_and_then_updates_once_with_dt_equal_to_it() {
        let timer = virtual_timer();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), false), None);
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
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn late_by_less_than_a_period_the_next_target_stays_one_period_after_the_last() {
        let timer = virtual_timer();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), false), None);
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
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn late_by_a_period_or_more_it_updates_once_and_counts_again_from_there() {
        let timer = virtual_timer();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), false), None);
        // 1.5 P late: one update, not one per missed target.
        timer.advance_ns(5 * P_NS / 2).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(5 * P_NS / 2));
        assert_quiet(&dts);
        // The next target is 3.5 P, not 3 P on the grid of multiples of P.
        timer.advance_ns(P_NS - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn exactly_a_period_late_counts_again_from_the_late_update() {
        let timer = virtual_timer();
        let (release, hold) = mpsc::channel();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), false), Some(hold));
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
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn an_update_longer_than_the_period_is_followed_at_once_without_catching_up() {
        let timer = virtual_timer();
        let (release, hold) = mpsc::channel();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), false), Some(hold));
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
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn a_capture_does_not_move_the_target() {
        let timer = virtual_timer();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), false), None);
        timer.advance_ns(P_NS / 2).unwrap();
        rt.capture(|states: Vec<()>| states.len()).wait().unwrap();
        timer.advance_ns(P_NS / 2 - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        assert_quiet(&dts);
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn a_restored_node_counts_its_period_from_its_last_update() {
        let timer = virtual_timer();
        let at_ns = 100 * P_NS;
        let config = RuntimeConfig {
            last_update_ns: Some(vec![at_ns - P_NS / 2]),
            timer_state: Some(TimerState {
                time_ns: at_ns,
                scale: 1.0,
            }),
            ..with_period(Some(P_NS), false)
        };
        let (rt, dts) = start_ticker(&timer, config, None);
        timer.advance_ns(P_NS / 2 - 1).unwrap();
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn each_node_keeps_its_own_period() {
        let timer = virtual_timer();
        let config = RuntimeConfig {
            period_ns: Some(vec![Some(P_NS), Some(3 * P_NS)]),
            ..RuntimeConfig::default()
        };
        let (rt, dts) = start_tickers(&timer, config, None);
        for _ in 0..2 {
            timer.advance_ns(P_NS).unwrap();
            assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
            assert_quiet(&dts[1]);
        }
        timer.advance_ns(P_NS).unwrap();
        assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
        assert_eq!(dts[1].recv_timeout(LONG), Ok(3 * P_NS));
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn without_a_period_any_time_passing_is_enough() {
        let timer = virtual_timer();
        let (rt, dts) = start_ticker(&timer, with_period(None, false), None);
        assert_quiet(&dts);
        timer.advance_ns(1).unwrap();
        assert_eq!(dts.recv_timeout(LONG), Ok(1));
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn the_manual_mode_ignores_the_period() {
        let timer = virtual_timer();
        let (rt, dts) = start_ticker(&timer, with_period(Some(P_NS), true), None);
        timer.advance_ns(1).unwrap();
        rt.step(None).wait().unwrap();
        assert_eq!(dts.try_recv(), Ok(1));
        rt.shutdown().wait().unwrap();
    }

    #[test]
    fn a_stop_or_a_capture_wakes_the_wait_for_the_period() {
        let timer = Arc::new(Timer::new());
        let period = Some(1_000 * 1_000_000_000);
        let (rt, dts) = start_ticker(&timer, with_period(period, false), None);
        let begin = Instant::now();
        rt.capture(|states: Vec<()>| states.len()).wait().unwrap();
        rt.request_stop();
        assert!(rt.join(Some(LONG)).unwrap());
        assert!(begin.elapsed() < Duration::from_secs(2));
        assert!(dts.try_recv().is_err());
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
    fn tick_for(period_ns: u64, scale: f64) -> (Vec<u64>, Duration) {
        let timer = Arc::new(Timer::new());
        timer.set_scale(scale).unwrap();
        let begin = Instant::now();
        let (rt, dts) = start_ticker(&timer, with_period(Some(period_ns), false), None);
        thread::sleep(RUN);
        rt.shutdown().wait().unwrap();
        let elapsed = begin.elapsed();
        (dts.try_iter().collect(), elapsed)
    }

    /// The node waits for its period on the real clock: neither faster (free
    /// running would update far more often) nor drifting behind.
    #[test]
    fn on_the_real_clock_it_updates_about_once_per_period() {
        let period_ns = 200 * H_NS;
        let (dts, elapsed) = tick_for(period_ns, 1.0);
        // The targets are P apart, so the updates cannot outnumber them.
        let at_most = elapsed.as_nanos() as f64 / period_ns as f64;
        assert!((dts.len() as f64) <= at_most, "{} > {at_most}", dts.len());
        // The targets at 200, 400, 600 and 800 ms come before the end at 1 s
        // as long as each wake-up is less than 200 ms late.
        assert!(dts.len() >= 4, "{} updates", dts.len());
    }

    /// The period is on the timer, so the time scale changes the updates per
    /// real second: a period of 800 ms at scale 4 is 200 ms of real time.
    #[test]
    fn the_time_scale_scales_the_updates_per_real_second() {
        let (period_ns, scale) = (800 * H_NS, 4.0);
        let (dts, elapsed) = tick_for(period_ns, scale);
        let at_most = elapsed.as_nanos() as f64 * scale / period_ns as f64;
        assert!((dts.len() as f64) <= at_most, "{} > {at_most}", dts.len());
        // 1 s of real time is 4 s on the timer: targets at 0.8, 1.6, 2.4, 3.2
        // and 4 s. The first four come before the end as long as each wake-up
        // is less than 200 ms (a period) of real time late. At scale 1 the same
        // second would hold one target, so 4 shows that the scale applies.
        assert!(dts.len() >= 4, "{} updates", dts.len());
    }

    #[test]
    fn rejects_a_zero_period_and_a_wrong_number_of_periods() {
        let timer = virtual_timer();
        for period_ns in [vec![Some(0)], vec![None, None]] {
            let config = RuntimeConfig {
                period_ns: Some(period_ns),
                ..RuntimeConfig::default()
            };
            let (dts, _) = mpsc::channel();
            let started = Runtime::start(
                vec![Ticker { dts, hold: None }],
                vec!["ticker".to_owned()],
                Arc::clone(&timer),
                config,
            );
            assert!(matches!(started, Err(RuntimeError::InvalidArgument(_))));
        }
    }
}
