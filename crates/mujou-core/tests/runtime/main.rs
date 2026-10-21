use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use mujou_core::runtime::{
    Hook, HookError, NodeHooks, NodeSpec, Reply, RunConfig, Runtime, RuntimeError,
};
use mujou_core::time::{RunClock, Source, Timer, Timestamp};

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
    /// Restored this many updates as its state.
    Restore(usize),
    Update(f64),
    Pause,
    Resume,
    Exception,
    Stop,
    Capture,
}

#[derive(Debug)]
struct Boom;

impl fmt::Display for Boom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("boom")
    }
}

impl std::error::Error for Boom {}

fn boom(hook: Hook) -> HookError {
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
    fail_on_start: bool,
    /// The state `restore` restores (its count of updates), as the bindings
    /// keep it: nothing to restore if `None`.
    state: Option<usize>,
    fail_on_restore: bool,
    /// Fails the `k`-th update (counting from 1).
    fail_on_update: Option<(usize, Failure)>,
    fail_on_capture: bool,
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
            Event::Start if self.fail_on_start => Err(boom(Hook::Start)),
            Event::Restore(_) if self.fail_on_restore => Err(boom(Hook::Restore)),
            Event::Capture if self.fail_on_capture => Err(boom(Hook::Capture)),
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
                    Some((k, failure)) if self.updates == k => self.fail(Hook::Update, failure),
                    _ => Ok(()),
                }
            }
            _ => Ok(()),
        }
    }

    fn fail(&self, hook: Hook, failure: Failure) -> Result<(), HookError> {
        match failure {
            Failure::Error => Err(boom(hook)),
            Failure::Panic => panic!("panic in {hook} of node {}", self.id),
        }
    }
}

impl NodeHooks for Probe {
    type State = usize;

    fn start(&mut self) -> Result<(), HookError> {
        self.hook(Event::Start)
    }

    fn restore(&mut self) -> Result<(), HookError> {
        let Some(updates) = self.state.take() else {
            return Ok(());
        };
        self.updates = updates;
        self.hook(Event::Restore(updates))
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

    fn capture(&mut self) -> Result<usize, HookError> {
        self.hook(Event::Capture).map(|()| self.updates)
    }
}

struct Probes {
    log: Log,
    nodes: Vec<Probe>,
}

fn probes(n: usize) -> Probes {
    let log = Arc::new(Mutex::new(Vec::new()));
    let nodes = (0..n)
        .map(|id| Probe {
            id,
            log: Arc::clone(&log),
            updates: 0,
            fail_on_start: false,
            state: None,
            fail_on_restore: false,
            fail_on_update: None,
            fail_on_capture: false,
            ticks: None,
            advance: None,
            gate: None,
        })
        .collect();
    Probes { log, nodes }
}

/// The hooks of node `id`.
fn events(log: &Log, id: usize) -> Vec<Event> {
    let log = log.lock().unwrap();
    log.iter()
        .filter(|(i, _, _)| *i == id)
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
    /// Starts the nodes of `config` in this mode.
    fn start<N: NodeHooks>(
        self,
        config: RunConfig<N>,
        timer: Arc<Timer>,
    ) -> Result<(Runtime<N>, Reply<()>), RuntimeError> {
        Runtime::start(mode_config(self, config), clock(&timer))
    }
}

/// `config` with its nodes where `mode` puts them.
fn mode_config<N>(mode: Mode, mut config: RunConfig<N>) -> RunConfig<N> {
    for node in &mut config.nodes {
        node.dedicated_thread = mode == Mode::Dedicated;
    }
    config.workers = match mode {
        Mode::OneWorker => Some(NonZeroUsize::MIN),
        Mode::TwoWorkers => NonZeroUsize::new(2),
        Mode::Dedicated | Mode::Pool => None,
    };
    config
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

/// A virtual timer of its own at 0: `cargo test` runs the tests in parallel
/// threads.
fn virtual_timer() -> Arc<Timer> {
    Arc::new(Timer::new(Source::Virtual, Timestamp::from_ns(0)))
}

/// The clock of a run on `timer`.
fn clock(timer: &Arc<Timer>) -> Arc<RunClock> {
    Arc::new(RunClock::new(Arc::clone(timer)))
}

/// `hooks` as the nodes `p0`, `p1`, ... of a free-running runtime: in the
/// pool, with a period of 1 ns, and each first Δt from the start.
fn config<N>(hooks: Vec<N>) -> RunConfig<N> {
    let nodes = (hooks.into_iter().enumerate())
        .map(|(i, hooks)| NodeSpec {
            hooks,
            name: format!("p{i}"),
            period_ns: NonZeroU64::MIN,
            dedicated_thread: false,
            last_update_ns: None,
        })
        .collect();
    RunConfig {
        nodes,
        manual: false,
        workers: None,
    }
}

/// [`config`] in the manual mode.
fn manual<N>(hooks: Vec<N>) -> RunConfig<N> {
    RunConfig {
        manual: true,
        ..config(hooks)
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
fn pool<N>(workers: usize, config: RunConfig<N>) -> RunConfig<N> {
    RunConfig {
        workers: NonZeroUsize::new(workers),
        ..config
    }
}

/// `config` with each node's period, in ns.
fn with_periods<N>(mut config: RunConfig<N>, periods_ns: &[u64]) -> RunConfig<N> {
    assert_eq!(config.nodes.len(), periods_ns.len());
    for (node, &period_ns) in config.nodes.iter_mut().zip(periods_ns) {
        node.period_ns = NonZeroU64::new(period_ns).unwrap();
    }
    config
}

/// Stops the runtime and waits for its end, as the host does.
fn shut_down<N: NodeHooks>(rt: &Runtime<N>) -> Result<(), RuntimeError> {
    rt.stop();
    rt.join(None).map(drop)
}
