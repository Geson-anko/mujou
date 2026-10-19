use std::fmt;
use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use mujou_core::runtime::{HookError, NodeHooks, Reply, Runtime, RuntimeConfig, RuntimeError};
use mujou_core::time::Timer;

const H_NS: u64 = 1_000_000;
/// How long a capture waits for the nodes to pause.
const PAUSE: Duration = Duration::from_secs(30);
/// How long to wait for something that must come.
const LONG: Duration = Duration::from_secs(5);
/// How long to watch for an update that must not come.
const QUIET: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Start,
    Update(f64),
    Pause,
    Resume,
    Exception,
    Stop,
    StateDict,
    /// [`NodeHooks::enter`] before this hook.
    Enter(&'static str),
}

#[derive(Debug)]
struct Boom;

impl fmt::Display for Boom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("boom")
    }
}

impl std::error::Error for Boom {}

fn boom(hook: &'static str) -> HookError {
    HookError {
        hook,
        source: Arc::new(Boom),
    }
}

/// How a hook of a [`Probe`] fails.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Failure {
    Error,
    Panic,
}

type Log = Arc<Mutex<Vec<(usize, Event, String)>>>;

/// Records its hook calls with the name of the thread that ran them.
struct Probe {
    id: usize,
    log: Log,
    updates: usize,
    enters: usize,
    /// Fails every `enter` from the `k`-th on (counting from 1).
    fail_enter: Option<(usize, Failure)>,
    fail_on_start: bool,
    /// Fails the `k`-th update (counting from 1).
    fail_on_update: Option<(usize, Failure)>,
    fail_on_state_dict: bool,
    /// Sends `(id, Δt in ns)` on every update.
    ticks: Option<Sender<(usize, u64)>>,
    /// Moves the virtual timer by `H_NS` in each of its first `k` updates.
    advance: Option<(Arc<Timer>, usize)>,
    /// Tells when it enters an update, then waits there until released.
    gate: Option<(Sender<()>, Receiver<()>)>,
}

impl Probe {
    fn hook(&mut self, event: Event) -> Result<(), HookError> {
        let thread = thread::current().name().unwrap_or_default().to_owned();
        let entry = (self.id, event.clone(), thread);
        self.log.lock().unwrap().push(entry);
        self.react(&event)
    }

    fn react(&mut self, event: &Event) -> Result<(), HookError> {
        match event {
            Event::Enter(_) => {
                self.enters += 1;
                match self.fail_enter {
                    // A name of its own: the core reports the next hook.
                    Some((k, failure)) if self.enters >= k => self.fail("enter", failure),
                    _ => Ok(()),
                }
            }
            Event::Start if self.fail_on_start => Err(boom("on_start")),
            Event::StateDict if self.fail_on_state_dict => Err(boom("state_dict")),
            Event::Update(dt) => {
                self.updates += 1;
                if let Some(ticks) = &self.ticks {
                    let _ = ticks.send((self.id, (dt * 1e9).round() as u64));
                }
                if let Some((timer, k)) = &self.advance {
                    if self.updates <= *k {
                        timer.advance_ns(H_NS).unwrap();
                    }
                }
                if let Some((entered, release)) = &self.gate {
                    let _ = entered.send(());
                    let _ = release.recv();
                }
                match self.fail_on_update {
                    Some((k, failure)) if self.updates == k => self.fail("on_update", failure),
                    _ => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    fn fail(&self, hook: &'static str, failure: Failure) -> Result<(), HookError> {
        match failure {
            Failure::Error => Err(boom(hook)),
            Failure::Panic => panic!("panic in {hook} of node {}", self.id),
        }
    }
}

impl NodeHooks for Probe {
    type State = usize;

    fn enter(&mut self, next_hook: &'static str) -> Result<(), HookError> {
        self.hook(Event::Enter(next_hook))
    }

    fn start(&mut self) -> Result<(), HookError> {
        self.hook(Event::Start)
    }

    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.hook(Event::Update(dt))
    }

    fn pause(&mut self) -> Result<(), HookError> {
        self.hook(Event::Pause)
    }

    fn resume(&mut self) -> Result<(), HookError> {
        self.hook(Event::Resume)
    }

    fn exception(&mut self) -> Result<(), HookError> {
        self.hook(Event::Exception)
    }

    fn stop(&mut self) -> Result<(), HookError> {
        self.hook(Event::Stop)
    }

    fn state_dict(&mut self) -> Result<usize, HookError> {
        self.hook(Event::StateDict).map(|()| self.updates)
    }
}

struct Probes {
    log: Log,
    nodes: Vec<Probe>,
    names: Vec<String>,
}

fn probes(n: usize) -> Probes {
    let log = Arc::new(Mutex::new(Vec::new()));
    let nodes = (0..n)
        .map(|id| Probe {
            id,
            log: Arc::clone(&log),
            updates: 0,
            enters: 0,
            fail_enter: None,
            fail_on_start: false,
            fail_on_update: None,
            fail_on_state_dict: false,
            ticks: None,
            advance: None,
            gate: None,
        })
        .collect();
    let names = (0..n).map(|i| format!("p{i}")).collect();
    Probes { log, nodes, names }
}

/// The hooks of node `id`, without the calls of `enter`.
fn events(log: &Log, id: usize) -> Vec<Event> {
    let log = log.lock().unwrap();
    log.iter()
        .filter(|(i, e, _)| *i == id && !matches!(e, Event::Enter(_)))
        .map(|(_, e, _)| e.clone())
        .collect()
}

/// Where the nodes of a test run: [`each_mode`] runs a test in the first three.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    /// Each node on a dedicated thread.
    Dedicated,
    /// The pool with the default number of workers.
    Pool,
    /// The pool with a single worker, fewer than the nodes.
    OneWorker,
    /// The pool with two workers, for a test that needs a second one whatever
    /// the number of CPUs.
    TwoWorkers,
}

impl Mode {
    /// Starts the nodes in this mode, with the rest of `config` as given.
    fn start<N: NodeHooks>(
        self,
        nodes: Vec<N>,
        names: Vec<String>,
        timer: Arc<Timer>,
        config: RuntimeConfig,
    ) -> Result<(Runtime<N>, Reply<()>), RuntimeError> {
        let config = RuntimeConfig {
            dedicated: Some(vec![self == Mode::Dedicated; nodes.len()]),
            workers: match self {
                Mode::OneWorker => Some(NonZeroUsize::MIN),
                Mode::TwoWorkers => NonZeroUsize::new(2),
                Mode::Dedicated | Mode::Pool => None,
            },
            ..config
        };
        Runtime::start(nodes, names, timer, config)
    }
}

/// Runs `test` in every [`Mode`]; a failure prints the mode it failed in.
fn each_mode(test: impl FnMut(Mode)) {
    in_modes(&[Mode::Dedicated, Mode::Pool, Mode::OneWorker], test);
}

/// Runs `test` in each of `modes`; a failure prints the mode it failed in.
fn in_modes(modes: &[Mode], mut test: impl FnMut(Mode)) {
    for &mode in modes {
        eprintln!("mode: {mode:?}");
        test(mode);
    }
}

/// A virtual timer of its own: `cargo test` runs the tests in parallel threads.
fn virtual_timer() -> Arc<Timer> {
    let timer = Arc::new(Timer::new());
    timer.use_virtual_clock(true);
    timer
}

fn manual() -> RuntimeConfig {
    RuntimeConfig {
        manual: true,
        ..RuntimeConfig::default()
    }
}

/// `exception` and `stop` once each, and nothing after them.
fn assert_exception_then_stop_once(events: &[Event]) {
    let count = |event: Event| events.iter().filter(|e| **e == event).count();
    assert_eq!((count(Event::Exception), count(Event::Stop)), (1, 1));
    let n = events.len();
    assert_eq!(&events[n - 2..], &[Event::Exception, Event::Stop]);
}

mod manual_mode {
    use super::*;

    #[test]
    fn steps_in_the_given_order_with_the_advanced_dt() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes { log, nodes, names } = probes(3);
            let t0 = timer.time_ns();
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
            started.wait().unwrap();
            for _ in 0..2 {
                timer.advance_ns(H_NS).unwrap();
                rt.step(Some(&[2, 0, 1])).wait().unwrap();
            }
            rt.shutdown().wait().unwrap();

            let order: Vec<usize> = (log.lock().unwrap().iter())
                .filter(|(_, e, _)| !matches!(e, Event::Enter(_)))
                .map(|(i, _, _)| *i)
                .collect();
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
                assert_eq!(events(&log, id), expected);
            }
            assert_eq!(rt.last_update_ns(), vec![t0 + 2 * H_NS; 3]);
        });
    }

    #[test]
    fn a_step_without_time_passing_does_not_update() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes { log, nodes, names } = probes(1);
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
            started.wait().unwrap();
            rt.step(None).wait().unwrap();
            rt.shutdown().wait().unwrap();
            assert_eq!(events(&log, 0), [Event::Start, Event::Stop]);
        });
    }

    #[test]
    fn step_needs_the_manual_mode() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes { nodes, names, .. } = probes(1);
            let config = RuntimeConfig::default();
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), config)
                .unwrap();
            started.wait().unwrap();
            assert!(matches!(rt.step(None).wait(), Err(RuntimeError::NotManual)));
            rt.shutdown().wait().unwrap();
        });
    }
}

mod startup {
    use super::*;

    /// Tries to move the time in `start`, as a slow start would on the real
    /// clock, and keeps whether the timer was held and whether it moved.
    struct Starter {
        timer: Arc<Timer>,
        seen: Arc<Mutex<Option<(bool, bool)>>>,
        dts: Arc<Mutex<Vec<f64>>>,
    }

    impl NodeHooks for Starter {
        type State = ();

        fn start(&mut self) -> Result<(), HookError> {
            let moved = self.timer.advance_ns(H_NS).is_ok();
            *self.seen.lock().unwrap() = Some((self.timer.is_paused(), moved));
            Ok(())
        }

        fn update(&mut self, dt: f64) -> Result<(), HookError> {
            self.dts.lock().unwrap().push(dt);
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

    #[test]
    fn the_time_is_held_while_the_nodes_start() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let seen = Arc::new(Mutex::new(None));
            let dts = Arc::new(Mutex::new(Vec::new()));
            let node = Starter {
                timer: Arc::clone(&timer),
                seen: Arc::clone(&seen),
                dts: Arc::clone(&dts),
            };
            let t0 = timer.time_ns();
            let (rt, started) = mode
                .start(
                    vec![node],
                    vec!["n".to_owned()],
                    Arc::clone(&timer),
                    manual(),
                )
                .unwrap();
            started.wait().unwrap();
            // Held while starting, so the start took no time; running afterwards.
            assert_eq!(*seen.lock().unwrap(), Some((true, false)));
            assert!(!timer.is_paused());
            assert_eq!(timer.time_ns(), t0);
            timer.advance_ns(H_NS).unwrap();
            rt.step(None).wait().unwrap();
            rt.shutdown().wait().unwrap();
            // The first Δt is only the time after the start.
            assert_eq!(*dts.lock().unwrap(), [H_NS as f64 / 1e9]);
        });
    }
}

mod capture {
    use super::*;

    #[test]
    fn builds_from_every_state_while_the_time_is_frozen() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes { log, nodes, names } = probes(2);
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
            started.wait().unwrap();
            timer.advance_ns(H_NS).unwrap();
            rt.step(None).wait().unwrap();
            let frozen = Arc::clone(&timer);
            let (states, paused) = rt
                .capture(move |states| (states, frozen.is_paused()), PAUSE)
                .wait()
                .unwrap();
            assert_eq!(states, vec![1, 1]);
            assert!(paused);
            assert!(!timer.is_paused());
            rt.shutdown().wait().unwrap();
            for id in 0..2 {
                let events = events(&log, id);
                assert_eq!(
                    &events[2..5],
                    &[Event::Pause, Event::StateDict, Event::Resume]
                );
            }
        });
    }
}

mod failure {
    use super::*;

    #[test]
    fn an_update_error_ends_every_node_with_exception_then_stop() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes {
                log,
                mut nodes,
                names,
            } = probes(3);
            nodes[1].fail_on_update = Some((3, Failure::Error));
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
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
                assert_exception_then_stop_once(&events(&log, id));
            }
            // Later requests report the same failure.
            assert!(matches!(rt.shutdown().wait(), Err(RuntimeError::Failed(_))));
            assert!(matches!(rt.join(None), Err(RuntimeError::Failed(_))));
        });
    }

    #[test]
    fn a_panic_is_a_failure() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes {
                log,
                mut nodes,
                names,
            } = probes(2);
            nodes[0].fail_on_update = Some((1, Failure::Panic));
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
            started.wait().unwrap();
            timer.advance_ns(H_NS).unwrap();
            let result = rt.step(None).wait();
            let Err(RuntimeError::Failed(failure)) = result else {
                panic!("expected a failure, got {result:?}");
            };
            assert_eq!((failure.node, failure.hook), (0, "on_update"));
            assert!(
                failure
                    .error
                    .to_string()
                    .contains("panic in on_update of node 0")
            );
            for id in 0..2 {
                assert_exception_then_stop_once(&events(&log, id));
            }
        });
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
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes { nodes, names, .. } = probes(1);
            let (running, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
            started.wait().unwrap();
            let before = timer.time_ns();

            let Probes { log, nodes, names } = probes(1);
            let result = mode.start(nodes, names, Arc::clone(&timer), restoring());
            assert!(matches!(result, Err(RuntimeError::InvalidArgument(_))));
            // Nothing started, and the running runtime's time did not move.
            assert!(log.lock().unwrap().is_empty());
            assert_eq!(timer.time_ns(), before);
            assert!(!timer.is_paused());

            // Once it has finished, the time can be restored.
            running.shutdown().wait().unwrap();
            let Probes { nodes, names, .. } = probes(1);
            let (restored, started) = mode
                .start(nodes, names, Arc::clone(&timer), restoring())
                .unwrap();
            started.wait().unwrap();
            assert_eq!(timer.time_ns(), 7 * H_NS);
            restored.shutdown().wait().unwrap();
        });
    }
}

mod separate_timers {
    use super::*;
    use mujou_core::time::TimerState;

    #[test]
    fn each_runtime_measures_dt_on_its_own_timer() {
        each_mode(|mode| {
            let (ta, tb) = (virtual_timer(), virtual_timer());
            let Probes {
                log: la,
                nodes,
                names,
            } = probes(1);
            let (a, started) = mode.start(nodes, names, Arc::clone(&ta), manual()).unwrap();
            started.wait().unwrap();
            let Probes {
                log: lb,
                nodes,
                names,
            } = probes(1);
            let (b, started) = mode.start(nodes, names, Arc::clone(&tb), manual()).unwrap();
            started.wait().unwrap();

            ta.advance_ns(2 * H_NS).unwrap();
            tb.advance_ns(5 * H_NS).unwrap();
            a.step(None).wait().unwrap();
            b.step(None).wait().unwrap();
            a.shutdown().wait().unwrap();
            b.shutdown().wait().unwrap();
            assert_eq!(events(&la, 0)[1], Event::Update(2.0 * H_NS as f64 / 1e9));
            assert_eq!(events(&lb, 0)[1], Event::Update(5.0 * H_NS as f64 / 1e9));
        });
    }

    #[test]
    fn a_capture_freezes_only_its_own_timer() {
        each_mode(|mode| {
            let (ta, tb) = (virtual_timer(), virtual_timer());
            let Probes { nodes, names, .. } = probes(1);
            let (a, started) = mode.start(nodes, names, Arc::clone(&ta), manual()).unwrap();
            started.wait().unwrap();
            let Probes {
                log: lb,
                nodes,
                names,
            } = probes(1);
            let (b, started) = mode.start(nodes, names, Arc::clone(&tb), manual()).unwrap();
            started.wait().unwrap();
            let b = Arc::new(b);

            // While `a` is paused, the runtime on the other timer keeps updating.
            let (own, other, other_rt) = (Arc::clone(&ta), Arc::clone(&tb), Arc::clone(&b));
            let (own_paused, other_paused) = a
                .capture(
                    move |_| {
                        other.advance_ns(H_NS).unwrap();
                        other_rt.step(None).wait().unwrap();
                        (own.is_paused(), other.is_paused())
                    },
                    PAUSE,
                )
                .wait()
                .unwrap();
            assert!(own_paused);
            assert!(!other_paused);
            assert_eq!(events(&lb, 0)[1], Event::Update(H_NS as f64 / 1e9));
            a.shutdown().wait().unwrap();
            b.shutdown().wait().unwrap();
        });
    }

    #[test]
    fn a_runtime_on_another_timer_does_not_block_a_restore() {
        each_mode(|mode| {
            let (ta, tb) = (virtual_timer(), virtual_timer());
            let Probes { nodes, names, .. } = probes(1);
            let (running, started) = mode.start(nodes, names, Arc::clone(&ta), manual()).unwrap();
            started.wait().unwrap();
            let before = ta.time_ns();

            let config = RuntimeConfig {
                timer_state: Some(TimerState {
                    time_ns: 7 * H_NS,
                    scale: 1.0,
                }),
                ..manual()
            };
            let Probes { nodes, names, .. } = probes(1);
            let (restored, started) = mode.start(nodes, names, Arc::clone(&tb), config).unwrap();
            started.wait().unwrap();
            assert_eq!(tb.time_ns(), 7 * H_NS);
            assert_eq!(ta.time_ns(), before);
            running.shutdown().wait().unwrap();
            restored.shutdown().wait().unwrap();
        });
    }
}

mod stop_request {
    use std::sync::mpsc;

    use super::*;

    #[test]
    fn returns_while_a_node_is_still_updating_and_join_sees_the_end() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes {
                log,
                mut nodes,
                names,
            } = probes(1);
            let (entered_tx, entered) = mpsc::channel();
            let (release, release_rx) = mpsc::channel();
            nodes[0].gate = Some((entered_tx, release_rx));
            let config = RuntimeConfig::default();
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), config)
                .unwrap();
            started.wait().unwrap();
            timer.advance_ns(H_NS).unwrap();
            entered.recv_timeout(LONG).unwrap();

            // The node is blocked in its update, so a waiting stop would not return.
            rt.request_stop();
            assert!(!rt.join(Some(QUIET)).unwrap());
            assert!(!events(&log, 0).contains(&Event::Stop));

            release.send(()).unwrap();
            assert!(rt.join(Some(LONG)).unwrap());
            assert!(events(&log, 0).contains(&Event::Stop));
            assert!(rt.is_finished());
        });
    }

    #[test]
    fn does_nothing_once_the_system_has_ended() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let Probes { log, nodes, names } = probes(2);
            let (rt, started) = mode
                .start(nodes, names, Arc::clone(&timer), manual())
                .unwrap();
            started.wait().unwrap();
            rt.request_stop();
            assert!(rt.join(Some(Duration::from_secs(5))).unwrap());
            rt.request_stop();
            rt.shutdown().wait().unwrap();
            for id in 0..2 {
                assert_eq!(events(&log, id), [Event::Start, Event::Stop]);
            }
        });
    }
}

mod period {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::Instant;

    use mujou_core::time::TimerState;

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
            if let Some(hold) = &self.hold_resume {
                let _ = hold.recv();
            }
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

    fn with_period(period_ns: u64, manual: bool) -> RuntimeConfig {
        RuntimeConfig {
            manual,
            period_ns: Some(vec![period_ns]),
            ..RuntimeConfig::default()
        }
    }

    /// Starts one ticker per entry of `config.period_ns` (one if it is `None`).
    fn start_tickers(
        mode: Mode,
        timer: &Arc<Timer>,
        config: RuntimeConfig,
        mut hold: Option<Receiver<()>>,
    ) -> (Runtime<Ticker>, Vec<Receiver<u64>>) {
        let n = config.period_ns.as_ref().map_or(1, Vec::len);
        let (tickers, dts): (Vec<_>, Vec<_>) = (0..n)
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
        let names = (0..n).map(|i| format!("ticker{i}")).collect();
        let (rt, started) = mode
            .start(tickers, names, Arc::clone(timer), config)
            .unwrap();
        started.wait().unwrap();
        (rt, dts)
    }

    fn start_ticker(
        mode: Mode,
        timer: &Arc<Timer>,
        config: RuntimeConfig,
        hold: Option<Receiver<()>>,
    ) -> (Runtime<Ticker>, Receiver<u64>) {
        let (rt, mut dts) = start_tickers(mode, timer, config, hold);
        (rt, dts.remove(0))
    }

    fn assert_quiet(dts: &Receiver<u64>) {
        assert_eq!(dts.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
    }

    #[test]
    fn waits_for_the_period_and_then_updates_once_with_dt_equal_to_it() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, false), None);
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
        });
    }

    #[test]
    fn late_by_less_than_a_period_the_next_target_stays_one_period_after_the_last() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, false), None);
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
        });
    }

    #[test]
    fn late_by_a_period_or_more_it_updates_once_and_counts_again_from_there() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, false), None);
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
        });
    }

    #[test]
    fn exactly_a_period_late_counts_again_from_the_late_update() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (release, hold) = mpsc::channel();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, false), Some(hold));
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
        });
    }

    #[test]
    fn an_update_longer_than_the_period_is_followed_at_once_without_catching_up() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (release, hold) = mpsc::channel();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, false), Some(hold));
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
        });
    }

    #[test]
    fn a_capture_does_not_move_the_target() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, false), None);
            timer.advance_ns(P_NS / 2).unwrap();
            rt.capture(|states: Vec<()>| states.len(), PAUSE)
                .wait()
                .unwrap();
            timer.advance_ns(P_NS / 2 - 1).unwrap();
            assert_quiet(&dts);
            timer.advance_ns(1).unwrap();
            assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
            assert_quiet(&dts);
            rt.shutdown().wait().unwrap();
        });
    }

    #[test]
    fn a_restored_node_counts_its_period_from_its_last_update() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let at_ns = 100 * P_NS;
            let config = RuntimeConfig {
                last_update_ns: Some(vec![at_ns - P_NS / 2]),
                timer_state: Some(TimerState {
                    time_ns: at_ns,
                    scale: 1.0,
                }),
                ..with_period(P_NS, false)
            };
            let (rt, dts) = start_ticker(mode, &timer, config, None);
            timer.advance_ns(P_NS / 2 - 1).unwrap();
            assert_quiet(&dts);
            timer.advance_ns(1).unwrap();
            assert_eq!(dts.recv_timeout(LONG), Ok(P_NS));
            rt.shutdown().wait().unwrap();
        });
    }

    #[test]
    fn each_node_keeps_its_own_period() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let config = RuntimeConfig {
                period_ns: Some(vec![P_NS, 3 * P_NS]),
                ..RuntimeConfig::default()
            };
            let (rt, dts) = start_tickers(mode, &timer, config, None);
            for _ in 0..2 {
                timer.advance_ns(P_NS).unwrap();
                assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
                assert_quiet(&dts[1]);
            }
            timer.advance_ns(P_NS).unwrap();
            assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
            assert_eq!(dts[1].recv_timeout(LONG), Ok(3 * P_NS));
            rt.shutdown().wait().unwrap();
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
            let config = RuntimeConfig {
                period_ns: Some(vec![P_NS, 1_000 * P_NS]),
                ..RuntimeConfig::default()
            };
            let (rt, dts) = start_tickers(mode, &timer, config, Some(hold));
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
            rt.shutdown().wait().unwrap();
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
            let (mut tickers, dts): (Vec<_>, Vec<_>) = (0..3)
                .map(|_| {
                    let (dts, rx) = mpsc::channel();
                    (Ticker::new(dts), rx)
                })
                .unzip();
            tickers[2].hold_resume = Some(hold);
            let config = RuntimeConfig {
                period_ns: Some(vec![P_NS, 1_000 * P_NS, P_NS]),
                ..RuntimeConfig::default()
            };
            let names = (0..3).map(|i| format!("ticker{i}")).collect();
            let (rt, started) = mode
                .start(tickers, names, Arc::clone(&timer), config)
                .unwrap();
            started.wait().unwrap();
            // Every node gets `resume`; that of node 2 blocks a worker.
            rt.capture(|states: Vec<()>| states.len(), PAUSE)
                .wait()
                .unwrap();
            thread::sleep(QUIET);
            timer.advance_ns(P_NS).unwrap();
            assert_eq!(dts[0].recv_timeout(LONG), Ok(P_NS));
            release.send(()).unwrap();
            rt.shutdown().wait().unwrap();
        });
    }

    #[test]
    fn with_a_period_of_1_ns_any_time_passing_is_enough() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (rt, dts) = start_ticker(mode, &timer, with_period(1, false), None);
            assert_quiet(&dts);
            timer.advance_ns(1).unwrap();
            assert_eq!(dts.recv_timeout(LONG), Ok(1));
            rt.shutdown().wait().unwrap();
        });
    }

    #[test]
    fn the_manual_mode_ignores_the_period() {
        each_mode(|mode| {
            let timer = virtual_timer();
            let (rt, dts) = start_ticker(mode, &timer, with_period(P_NS, true), None);
            timer.advance_ns(1).unwrap();
            rt.step(None).wait().unwrap();
            assert_eq!(dts.try_recv(), Ok(1));
            rt.shutdown().wait().unwrap();
        });
    }

    #[test]
    fn a_stop_or_a_capture_wakes_the_wait_for_the_period() {
        each_mode(|mode| {
            let timer = Arc::new(Timer::new());
            let period = 1_000 * 1_000_000_000;
            let (rt, dts) = start_ticker(mode, &timer, with_period(period, false), None);
            let begin = Instant::now();
            rt.capture(|states: Vec<()>| states.len(), PAUSE)
                .wait()
                .unwrap();
            rt.request_stop();
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
        let timer = Arc::new(Timer::new());
        timer.set_scale(scale).unwrap();
        let begin = Instant::now();
        let (rt, dts) = start_ticker(mode, &timer, with_period(period_ns, false), None);
        thread::sleep(RUN);
        rt.shutdown().wait().unwrap();
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

    #[test]
    fn rejects_a_zero_period_and_a_wrong_number_of_periods() {
        each_mode(|mode| {
            let timer = virtual_timer();
            for period_ns in [vec![0], vec![1, 1]] {
                let config = RuntimeConfig {
                    period_ns: Some(period_ns),
                    ..RuntimeConfig::default()
                };
                let (dts, _) = mpsc::channel();
                let started = mode.start(
                    vec![Ticker::new(dts)],
                    vec!["ticker".to_owned()],
                    Arc::clone(&timer),
                    config,
                );
                assert!(matches!(started, Err(RuntimeError::InvalidArgument(_))));
            }
        });
    }
}

mod pool {
    use std::collections::BTreeSet;
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::time::Instant;

    use super::*;

    fn pool(workers: usize, config: RuntimeConfig) -> RuntimeConfig {
        RuntimeConfig {
            workers: NonZeroUsize::new(workers),
            ..config
        }
    }

    fn threads(log: &Log) -> BTreeSet<String> {
        let log = log.lock().unwrap();
        log.iter().map(|(_, _, thread)| thread.clone()).collect()
    }

    fn numbered(prefix: &str, k: usize) -> BTreeSet<String> {
        (0..k).map(|i| format!("{prefix}{i}")).collect()
    }

    /// The updates of one step of the virtual time: receives exactly `k`.
    fn receive(ticks: &Receiver<(usize, u64)>, k: usize) -> Vec<(usize, u64)> {
        (0..k).map(|_| ticks.recv_timeout(LONG).unwrap()).collect()
    }

    #[test]
    fn updates_the_earliest_target_first_and_by_index_on_a_tie() {
        let timer = virtual_timer();
        let Probes {
            mut nodes, names, ..
        } = probes(3);
        let (tx, ticks) = mpsc::channel();
        for node in &mut nodes {
            node.ticks = Some(tx.clone());
        }
        let config = RuntimeConfig {
            period_ns: Some(vec![3 * H_NS, 2 * H_NS, H_NS]),
            ..pool(1, RuntimeConfig::default())
        };
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        let h = H_NS;
        // The targets are 3, 2 and 1 ms: at 5 ms all are due, earliest first.
        // Then `next_target_ns` gives 6, 7 and 6 ms.
        timer.advance_ns(5 * h).unwrap();
        assert_eq!(receive(&ticks, 3), [(2, 5 * h), (1, 5 * h), (0, 5 * h)]);
        // 6 ms: nodes 0 and 2 tie, by index. Next 9 and 7 ms.
        timer.advance_ns(h).unwrap();
        assert_eq!(receive(&ticks, 2), [(0, h), (2, h)]);
        // 7 ms: nodes 1 and 2. Next 9 and 8 ms.
        timer.advance_ns(h).unwrap();
        assert_eq!(receive(&ticks, 2), [(1, 2 * h), (2, h)]);
        timer.advance_ns(h).unwrap();
        assert_eq!(receive(&ticks, 1), [(2, h)]);
        timer.advance_ns(h).unwrap();
        assert_eq!(receive(&ticks, 3), [(0, 3 * h), (1, 2 * h), (2, h)]);
        assert_eq!(ticks.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
        rt.shutdown().wait().unwrap();
    }

    /// Updates that take longer than the time between the targets: the node
    /// that just updated waits behind the others, so they take turns, and only
    /// their Δt grows (the real time since each one's last update).
    #[test]
    fn an_overloaded_worker_serves_the_nodes_in_turn() {
        let timer = virtual_timer();
        let Probes {
            mut nodes, names, ..
        } = probes(3);
        let (tx, ticks) = mpsc::channel();
        for node in &mut nodes {
            node.ticks = Some(tx.clone());
            // Each of the first two updates takes 1 ms.
            node.advance = Some((Arc::clone(&timer), 2));
        }
        let config = pool(1, RuntimeConfig::default());
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        timer.advance_ns(1).unwrap();
        let h = H_NS;
        // The updates start at 1 ns, 1 ms + 1 ns, 2 ms + 1 ns, and so on, each
        // first two moving the time by 1 ms; the last three all at 6 ms + 1 ns.
        let expected = [
            (0, 1),
            (1, h + 1),
            (2, 2 * h + 1),
            (0, 3 * h),
            (1, 3 * h),
            (2, 3 * h),
            (0, 3 * h),
            (1, 2 * h),
            (2, h),
        ];
        assert_eq!(receive(&ticks, 9), expected);
        assert_eq!(ticks.recv_timeout(QUIET), Err(RecvTimeoutError::Timeout));
        rt.shutdown().wait().unwrap();
    }

    /// Steps every node of a manual runtime a few times.
    fn step_all(timer: &Timer, rt: &Runtime<Probe>) {
        for _ in 0..3 {
            timer.advance_ns(H_NS).unwrap();
            rt.step(None).wait().unwrap();
        }
    }

    #[test]
    fn runs_the_hooks_on_the_given_number_of_workers() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(64);
        let (rt, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), pool(2, manual())).unwrap();
        started.wait().unwrap();
        step_all(&timer, &rt);
        rt.shutdown().wait().unwrap();
        assert!(threads(&log).is_subset(&numbered("mujou-worker-", 2)));
    }

    /// Sees only the threads that ran a hook: an extra idle worker would go
    /// unnoticed unless it happened to take some work.
    #[test]
    fn spawns_no_more_workers_than_nodes_in_the_pool() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(2);
        let (rt, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), pool(8, manual())).unwrap();
        started.wait().unwrap();
        step_all(&timer, &rt);
        rt.shutdown().wait().unwrap();
        assert!(threads(&log).is_subset(&numbered("mujou-worker-", 2)));
    }

    /// Sees only the threads that ran a hook: a worker would run none here, so
    /// this checks that the dedicated threads run every hook.
    #[test]
    fn the_dedicated_threads_run_every_hook_without_nodes_in_the_pool() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(2);
        let config = RuntimeConfig {
            dedicated: Some(vec![true; 2]),
            ..manual()
        };
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        step_all(&timer, &rt);
        rt.shutdown().wait().unwrap();
        assert_eq!(threads(&log), numbered("mujou-node-p", 2));
    }

    #[test]
    fn enters_a_node_only_when_the_thread_switches_to_it() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(2);
        let (rt, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), pool(1, manual())).unwrap();
        started.wait().unwrap();
        for i in [0, 1, 1, 0] {
            timer.advance_ns(H_NS).unwrap();
            rt.step(Some(&[i])).wait().unwrap();
        }
        rt.shutdown().wait().unwrap();
        let dt = |k: u64| Event::Update((k * H_NS) as f64 / 1e9);
        let expected = [
            (0, Event::Enter("on_start")),
            (0, Event::Start),
            (1, Event::Enter("on_start")),
            (1, Event::Start),
            (0, Event::Enter("on_update")),
            (0, dt(1)),
            (1, Event::Enter("on_update")),
            (1, dt(2)),
            // The worker ran node 1 last: no `enter`.
            (1, dt(1)),
            (0, Event::Enter("on_update")),
            (0, dt(3)),
            (0, Event::Stop),
            (1, Event::Enter("on_stop")),
            (1, Event::Stop),
        ];
        let log: Vec<_> = log
            .lock()
            .unwrap()
            .iter()
            .map(|(i, e, _)| (*i, e.clone()))
            .collect();
        assert_eq!(log, expected);
    }

    #[test]
    fn a_dedicated_thread_enters_its_node_once_before_start() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(1);
        let config = RuntimeConfig {
            dedicated: Some(vec![true]),
            ..manual()
        };
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        step_all(&timer, &rt);
        rt.shutdown().wait().unwrap();
        let log = log.lock().unwrap();
        let enters: Vec<_> = (log.iter())
            .filter(|(_, e, _)| matches!(e, Event::Enter(_)))
            .collect();
        assert_eq!(enters.len(), 1);
        assert_eq!(log[0].1, Event::Enter("on_start"));
    }

    #[test]
    fn dedicated_and_pool_nodes_go_through_the_same_lifecycle() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(2);
        let config = RuntimeConfig {
            dedicated: Some(vec![true, false]),
            ..manual()
        };
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        let states = rt.capture(|states| states, PAUSE).wait().unwrap();
        assert_eq!(states, [1, 1]);
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        rt.shutdown().wait().unwrap();
        let dt = Event::Update(H_NS as f64 / 1e9);
        for id in 0..2 {
            let expected = [
                Event::Start,
                dt.clone(),
                Event::Pause,
                Event::StateDict,
                Event::Resume,
                dt.clone(),
                Event::Stop,
            ];
            assert_eq!(events(&log, id), expected);
        }
    }

    #[test]
    fn a_failure_with_fewer_workers_than_nodes_ends_every_node() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
            ..
        } = probes(8);
        nodes[3].fail_on_update = Some((2, Failure::Error));
        let config = pool(2, RuntimeConfig::default());
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        let mut ended = Ok(false);
        for _ in 0..100 {
            timer.advance_ns(H_NS).unwrap();
            ended = rt.join(Some(Duration::from_millis(20)));
            if !matches!(ended, Ok(false)) {
                break;
            }
        }
        let Err(RuntimeError::Failed(failure)) = ended else {
            panic!("expected a failure, got {ended:?}");
        };
        assert_eq!((failure.node, failure.hook), (3, "on_update"));
        for id in 0..8 {
            assert_exception_then_stop_once(&events(&log, id));
        }
    }

    #[test]
    fn a_failed_start_in_the_pool_lets_no_node_update() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
            ..
        } = probes(4);
        nodes[2].fail_on_start = true;
        let config = pool(2, RuntimeConfig::default());
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        let result = started.wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.node, failure.hook), (2, "on_start"));
        assert!(rt.is_finished());
        for id in 0..4 {
            let events = events(&log, id);
            assert_eq!(events, [Event::Start, Event::Exception, Event::Stop]);
        }
    }

    #[test]
    fn an_exception_after_a_normal_stop_still_comes_before_stop() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
            ..
        } = probes(2);
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        nodes[0].gate = Some((entered_tx, release_rx));
        nodes[0].fail_on_update = Some((1, Failure::Error));
        let config = pool(2, RuntimeConfig::default());
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        entered.recv_timeout(LONG).unwrap();
        // Node 1 leaves its loop for the stop while node 0 is still updating.
        // The sleep only makes that order likely: the outcome is the same if
        // the exception comes first.
        rt.request_stop();
        thread::sleep(QUIET);
        release.send(()).unwrap();
        assert!(matches!(rt.join(Some(LONG)), Err(RuntimeError::Failed(_))));
        for id in 0..2 {
            assert_exception_then_stop_once(&events(&log, id));
        }
    }

    #[test]
    fn a_stop_while_paused_ends_the_nodes_without_resume() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
            ..
        } = probes(3);
        nodes[0].fail_on_state_dict = true;
        let (rt, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), pool(1, manual())).unwrap();
        started.wait().unwrap();
        let result = rt.capture(|states| states, PAUSE).wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.node, failure.hook), (0, "state_dict"));
        for id in 0..3 {
            let events = events(&log, id);
            assert!(!events.contains(&Event::Resume), "{events:?}");
            assert_exception_then_stop_once(&events);
        }
    }

    #[test]
    fn rejects_dedicated_flags_of_the_wrong_length() {
        let timer = virtual_timer();
        let Probes {
            log, nodes, names, ..
        } = probes(2);
        let config = RuntimeConfig {
            dedicated: Some(vec![true]),
            ..RuntimeConfig::default()
        };
        let result = Runtime::start(nodes, names, Arc::clone(&timer), config);
        assert!(matches!(result, Err(RuntimeError::InvalidArgument(_))));
        assert!(log.lock().unwrap().is_empty());
    }

    /// Asks its own runtime to shut down from an update.
    struct Caller {
        runtime: Arc<Mutex<Option<Arc<Runtime<Caller>>>>>,
        result: Arc<Mutex<Option<Result<(), RuntimeError>>>>,
    }

    impl NodeHooks for Caller {
        type State = ();

        fn start(&mut self) -> Result<(), HookError> {
            Ok(())
        }

        fn update(&mut self, _dt: f64) -> Result<(), HookError> {
            let rt = self.runtime.lock().unwrap().clone().unwrap();
            *self.result.lock().unwrap() = Some(rt.shutdown().wait());
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

    #[test]
    fn a_hook_on_a_worker_cannot_wait_for_its_own_runtime() {
        let timer = virtual_timer();
        let runtime = Arc::new(Mutex::new(None));
        let result = Arc::new(Mutex::new(None));
        let caller = Caller {
            runtime: Arc::clone(&runtime),
            result: Arc::clone(&result),
        };
        let (rt, started) = Runtime::start(
            vec![caller],
            vec!["caller".to_owned()],
            Arc::clone(&timer),
            pool(1, manual()),
        )
        .unwrap();
        started.wait().unwrap();
        let rt = Arc::new(rt);
        *runtime.lock().unwrap() = Some(Arc::clone(&rt));
        timer.advance_ns(H_NS).unwrap();
        rt.step(None).wait().unwrap();
        assert!(matches!(
            *result.lock().unwrap(),
            Some(Err(RuntimeError::Reentrant))
        ));
        // Break the cycle between the node and its runtime.
        *runtime.lock().unwrap() = None;
        rt.shutdown().wait().unwrap();
    }

    /// A smoke test on the real clock, after the deterministic ones: free
    /// running nodes on several workers update and stop. The hooks of a node
    /// cannot overlap here even if the scheduling broke, since a node is
    /// called under its lock; a node taken by two workers would trip the
    /// `debug_assert` in the core instead and leave the runtime hanging.
    #[test]
    fn free_running_nodes_on_several_workers_update_and_stop() {
        let timer = Arc::new(Timer::new());
        let Probes {
            log, nodes, names, ..
        } = probes(16);
        let config = pool(4, RuntimeConfig::default());
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        thread::sleep(Duration::from_millis(200));
        rt.shutdown().wait().unwrap();
        assert!(threads(&log).is_subset(&numbered("mujou-worker-", 4)));
        for id in 0..16 {
            assert!(events(&log, id).len() > 2);
        }
    }

    #[test]
    fn a_failed_enter_fails_the_next_hook_and_still_lets_the_node_clean_up() {
        for failure in [Failure::Error, Failure::Panic] {
            in_modes(&[Mode::Dedicated, Mode::OneWorker], |mode| {
                let timer = virtual_timer();
                let Probes {
                    log,
                    mut nodes,
                    names,
                } = probes(2);
                nodes[0].fail_enter = Some((1, failure));
                let started = mode.start(nodes, names, Arc::clone(&timer), manual());
                let (rt, started) = started.unwrap();
                let result = started.wait();
                let Err(RuntimeError::Failed(failure)) = result else {
                    panic!("expected a failure, got {result:?}");
                };
                // Under the name of the hook it stands for, not "enter".
                assert_eq!((failure.node, failure.hook), (0, "on_start"));
                assert!(rt.is_finished());
                // `start` is not called; `exception` and `stop` still are.
                assert_eq!(events(&log, 0), [Event::Exception, Event::Stop]);
                assert_exception_then_stop_once(&events(&log, 1));
            });
        }
    }

    #[test]
    fn a_failed_enter_on_a_switch_fails_the_update() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
        } = probes(2);
        // The first `enter` (before `start`) succeeds, the next ones fail.
        nodes[0].fail_enter = Some((2, Failure::Error));
        let (rt, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), pool(1, manual())).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        // The worker ran node 1 last (its start), so it enters node 0 again.
        let result = rt.step(Some(&[0])).wait();
        let Err(RuntimeError::Failed(failure)) = result else {
            panic!("expected a failure, got {result:?}");
        };
        assert_eq!((failure.node, failure.hook), (0, "on_update"));
        // Every hook after the failed `enter` is preceded by another `enter`.
        let node_0: Vec<_> = (log.lock().unwrap().iter())
            .filter(|(i, _, _)| *i == 0)
            .map(|(_, e, _)| e.clone())
            .collect();
        assert_eq!(
            node_0,
            [
                Event::Enter("on_start"),
                Event::Start,
                Event::Enter("on_update"),
                Event::Enter("on_exception"),
                Event::Exception,
                Event::Enter("on_stop"),
                Event::Stop,
            ]
        );
        assert_exception_then_stop_once(&events(&log, 1));
    }

    /// The failed node calls `exception` first, as a dedicated thread does
    /// right after catching, even though its failure queued the others first.
    #[test]
    fn the_failed_node_calls_exception_before_the_nodes_its_failure_woke() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
        } = probes(3);
        nodes[0].fail_on_update = Some((1, Failure::Error));
        let (rt, started) =
            Runtime::start(nodes, names, Arc::clone(&timer), pool(1, manual())).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        assert!(rt.step(Some(&[0])).wait().is_err());
        let exceptions: Vec<_> = (log.lock().unwrap().iter())
            .filter(|(_, e, _)| *e == Event::Exception)
            .map(|(i, _, _)| *i)
            .collect();
        assert_eq!(exceptions, [0, 1, 2]);
    }

    /// While one worker is held in an update, another one takes over the
    /// timekeeping and updates the other node on time.
    #[test]
    fn another_worker_keeps_the_time_while_one_is_held() {
        let timer = virtual_timer();
        let Probes {
            mut nodes, names, ..
        } = probes(2);
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        nodes[0].gate = Some((entered_tx, release_rx));
        let (tx, ticks) = mpsc::channel();
        nodes[1].ticks = Some(tx);
        let config = RuntimeConfig {
            period_ns: Some(vec![H_NS; 2]),
            ..pool(2, RuntimeConfig::default())
        };
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        timer.advance_ns(H_NS).unwrap();
        entered.recv_timeout(LONG).unwrap();
        assert_eq!(ticks.recv_timeout(LONG), Ok((1, H_NS)));
        for _ in 0..3 {
            timer.advance_ns(H_NS).unwrap();
            assert_eq!(ticks.recv_timeout(LONG), Ok((1, H_NS)));
        }
        // Node 0 returns at once from its later updates.
        drop(release);
        rt.shutdown().wait().unwrap();
    }

    /// A capture while the only worker is in an update: the update runs to its
    /// end, no update starts after the freeze, and every node's last update is
    /// at or before the frozen time.
    #[test]
    fn a_capture_freezes_the_time_before_any_later_update_in_the_pool() {
        let timer = virtual_timer();
        let Probes {
            log,
            mut nodes,
            names,
        } = probes(2);
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        nodes[0].gate = Some((entered_tx, release_rx));
        let config = pool(1, RuntimeConfig::default());
        let (rt, started) = Runtime::start(nodes, names, Arc::clone(&timer), config).unwrap();
        started.wait().unwrap();
        let t0 = timer.time_ns();
        timer.advance_ns(H_NS).unwrap();
        // Node 0 holds the worker; node 1 is due and waits for it.
        entered.recv_timeout(LONG).unwrap();
        let capture = rt.capture(|states| states, PAUSE);
        // The control thread freezes the time right after the pause request.
        let deadline = Instant::now() + LONG;
        while !timer.is_paused() {
            assert!(Instant::now() < deadline, "the capture did not freeze");
            thread::sleep(Duration::from_millis(1));
        }
        let frozen_ns = timer.time_ns();
        release.send(()).unwrap();
        assert_eq!(capture.wait().unwrap(), [1, 0]);
        let dt = Event::Update(H_NS as f64 / 1e9);
        let expected = [
            (0, Event::Start),
            (1, Event::Start),
            (0, dt),
            // Node 1 was due but pauses without updating.
            (1, Event::Pause),
            (0, Event::Pause),
            (0, Event::StateDict),
            (1, Event::StateDict),
        ];
        let log_now: Vec<_> = (log.lock().unwrap().iter())
            .filter(|(_, e, _)| !matches!(e, Event::Enter(_)))
            .map(|(i, e, _)| (*i, e.clone()))
            .take(expected.len())
            .collect();
        assert_eq!(log_now, expected);
        assert_eq!(frozen_ns, t0 + H_NS);
        assert!(rt.last_update_ns().iter().all(|&ns| ns <= frozen_ns));
        drop(release);
        rt.shutdown().wait().unwrap();
    }
}
