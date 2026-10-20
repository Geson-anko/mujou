use std::fmt;
use std::num::NonZeroUsize;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use mujou_core::runtime::{HookError, NodeHooks, Reply, Runtime, RuntimeConfig, RuntimeError};
use mujou_core::time::Timer;

mod capture;
mod failure;
mod lifecycle;
mod manual;
mod period;
mod restore;
mod sched;

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

/// `config` on a pool of `workers` workers.
fn pool(workers: usize, config: RuntimeConfig) -> RuntimeConfig {
    RuntimeConfig {
        workers: NonZeroUsize::new(workers),
        ..config
    }
}
