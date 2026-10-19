//! Bindings of `mujou_core::runtime`: Python nodes on native threads.
//!
//! Every wait on the host releases the GIL (detaches) and checks for signals
//! every `SIGNAL_CHECK_INTERVAL`, like `sleep`: the node and control threads
//! attach to call the hooks, so a host that kept the GIL while waiting for them
//! would deadlock on a GIL build.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use mujou_core::runtime::{
    self as core, HookError, NodeFailure, NodeHooks, Reply, RuntimeConfig, RuntimeError,
};
use mujou_core::time::Timer;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};

use crate::hint::Build;
use crate::time::{PyTimer, extract_state, to_ns};
use crate::wait_interruptibly;

// pyo3 does not introspect these, so `crates/mujou-stubgen` adds them to the
// stub with the same docstrings. They cannot be classes: pyo3 0.30 declares
// `PyTimeoutError` with the layout of `BaseException`, smaller than that of
// `TimeoutError`, so a class extending it fails to initialize.
create_exception!(
    mujou._core._native,
    NetworkError,
    PyRuntimeError,
    "The network stopped because a node raised. `__cause__` is the first exception."
);
create_exception!(
    mujou._core._native,
    PauseTimeoutError,
    PyTimeoutError,
    "Some nodes did not pause in time, so the state was not taken. The network keeps running."
);

/// The base of every node.
///
/// The runtime calls the hooks on the thread that runs the node: a worker of
/// the run's pool, which may differ from one hook to the next, or the node's
/// own thread if it was started as dedicated (`state_dict`, through
/// `_capture_state`, on the control thread while the node is paused). The hooks
/// of a node never overlap, and each runs in the node's own
/// `contextvars.Context` (but `_capture_state` and `_before_hook`). They do
/// nothing by default.
/// The hooks are looked up by name until `on_start` (and `load_state_dict`)
/// has returned, and once after that, so replacing one in those hooks takes
/// effect. `state_dict` itself is looked up by the default `_capture_state` at
/// each capture, so replacing it later takes effect too.
#[pyclass(subclass, frozen, module = "mujou._core._native")]
pub struct Node;

#[pymethods]
impl Node {
    /// Accepts and ignores any arguments, so that subclasses define `__init__`
    /// freely.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Self
    }

    /// Called right before a hook whenever a thread switches to this node (the
    /// first time a thread runs it, and on a worker whenever it ran another
    /// node in between), for the framework's per-thread settings, so that no
    /// override of a public hook can skip them. An error here is reported as
    /// the error of the hook that was to follow.
    fn _on_enter(&self) {}

    /// Called right before every public hook but `state_dict` (after
    /// `_on_enter` when the thread switches to this node), for the framework's
    /// settings that must hold at the start of each hook whatever the previous
    /// hook left. It runs outside the node's `contextvars.Context`, so a
    /// setting held in a context variable does not take effect here. An error
    /// here is reported as the error of that hook, which is then not called,
    /// except `on_exception` and `on_stop`, which still run.
    fn _before_hook(&self) {}

    /// Called once when the run starts (also when it is restored), before any
    /// other public hook. Allocate and initialize the state here.
    fn on_start(&self) {}

    /// Called repeatedly with `dt`, the seconds on the run's timer since the
    /// node's last update (always > 0). Read the ports, update in proportion to
    /// `dt` and send.
    #[pyo3(signature = (dt: "float"))]
    fn on_update(&self, dt: f64) {
        let _ = dt;
    }

    /// Called right after each `on_update`, for the framework's per-step work
    /// (so that overriding `on_update` cannot skip it).
    fn _after_update(&self) {}

    /// Called when the node has stopped for a pause of the run (to take the
    /// state), at the end of its current `on_update`.
    fn on_pause(&self) {}

    /// Called when the pause ends, before the next `on_update`.
    fn on_resume(&self) {}

    /// Called once on every node when a hook of any node has raised, before
    /// `on_stop`. What it raises is logged and ignored.
    fn on_exception(&self) {}

    /// Called once when the run stops, also after an exception. It must not
    /// change what `state_dict` returns. What it raises is logged and ignored.
    fn on_stop(&self) {}

    /// Returns the node's own state as a picklable object, when the run's state
    /// is taken (the node is paused or the run has ended). `None` by default.
    #[pyo3(signature = () -> "object")]
    fn state_dict(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    /// Restores `state`, a copy of what `state_dict` returned, after `on_start`
    /// when the run is restored. Does nothing by default.
    #[pyo3(signature = (state: "object"))]
    fn load_state_dict(&self, state: &Bound<'_, PyAny>) {
        let _ = state;
    }

    /// Called instead of `load_state_dict` when the runtime restores the state
    /// (after `on_start`), for the framework's work around it. Calls
    /// `self.load_state_dict(state)` by default.
    #[pyo3(signature = (state: "object"))]
    fn _restore_state(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1("load_state_dict", (state,)).map(drop)
    }

    /// Called instead of `state_dict` when the runtime takes the state (on the
    /// control thread, while the node is paused), for the framework's work
    /// around it. Returns `self.state_dict()` by default.
    #[pyo3(signature = () -> "object")]
    fn _capture_state(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        slf.call_method0("state_dict").map(Bound::unbind)
    }
}

/// A Python exception raised by a hook, kept to become `__cause__`.
#[derive(Debug)]
struct PyHookError {
    err: PyErr,
    message: String,
}

impl fmt::Display for PyHookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for PyHookError {}

/// Logs to `logging.getLogger("mujou")` with the exception's traceback.
fn log_error(py: Python<'_>, message: &str, err: &PyErr) {
    let result = (|| -> PyResult<()> {
        let logger = py
            .import("logging")?
            .call_method1("getLogger", ("mujou",))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("exc_info", err.value(py))?;
        logger.call_method("error", ("%s", message), Some(&kwargs))?;
        Ok(())
    })();
    if let Err(e) = result {
        e.write_unraisable(py, None);
    }
}

/// The hooks called more than once, looked up once at the end of `start`
/// instead of by name at every call. `PyNode::methods` holds them in this order.
const CACHED_HOOKS: [&str; 9] = [
    "_on_enter",
    "_before_hook",
    "on_update",
    "_after_update",
    "on_pause",
    "on_resume",
    "on_exception",
    "on_stop",
    "state_dict",
];

/// The attribute the runtime calls for `hook`. Taking the state goes through
/// `_capture_state`, which calls `state_dict`, and restoring it through
/// `_restore_state`, which calls `load_state_dict`; errors still name the
/// public hook.
fn attribute(hook: &str) -> &str {
    match hook {
        "state_dict" => "_capture_state",
        "load_state_dict" => "_restore_state",
        _ => hook,
    }
}

/// A Python node seen by the runtime.
struct PyNode {
    node: Py<PyAny>,
    name: String,
    /// The node's own `contextvars.Context`, which every hook runs in, so that
    /// context variables stay with the node whichever worker runs it.
    context: Py<PyAny>,
    /// The state to restore in `start`.
    state: Option<Py<PyAny>>,
    /// The methods of `CACHED_HOOKS`, once `start` has looked them up.
    methods: Option<Vec<Py<PyAny>>>,
}

impl PyNode {
    /// Every hook error is logged where it is caught, including the ones that
    /// do not end the system (a second exception, `on_exception`, `on_stop`).
    fn error(&self, py: Python<'_>, hook: &'static str, err: PyErr) -> HookError {
        let message = format!("node '{}' raised in {hook}", self.name);
        log_error(py, &message, &err);
        HookError {
            hook,
            source: Arc::new(PyHookError { err, message }),
        }
    }

    /// The method the runtime calls for `hook`.
    fn method<'py>(&self, py: Python<'py>, hook: &str) -> PyResult<Bound<'py, PyAny>> {
        let cached = (self.methods.as_ref())
            .zip(CACHED_HOOKS.iter().position(|&h| h == hook))
            .map(|(methods, i)| &methods[i]);
        match cached {
            Some(method) => Ok(method.bind(py).clone()),
            // Before the lookup, or if it failed: by name.
            None => self.node.bind(py).getattr(attribute(hook)),
        }
    }

    /// Calls `hook` in the node's context.
    fn call(
        &self,
        hook: &'static str,
        args: impl for<'py> FnOnce(Python<'py>) -> Vec<Bound<'py, PyAny>>,
    ) -> Result<Py<PyAny>, HookError> {
        self.call_as(hook, hook, args)
    }

    /// Calls `hook` in the node's context, reporting an error as `reported`.
    fn call_as(
        &self,
        hook: &'static str,
        reported: &'static str,
        args: impl for<'py> FnOnce(Python<'py>) -> Vec<Bound<'py, PyAny>>,
    ) -> Result<Py<PyAny>, HookError> {
        // Every runtime thread keeps its Python thread state for its whole life
        // (`on_thread`), so this attach only takes the GIL (on a GIL build).
        Python::attach(|py| {
            let result = (|| {
                // Outside the node's context (see `Node::_before_hook`).
                if !matches!(hook, "_on_enter" | "_after_update") {
                    match self.method(py, "_before_hook").and_then(|m| m.call0()) {
                        Ok(_) => {}
                        // Like a failed `enter`: logged, and the hook still
                        // runs so that the node can release its resources.
                        Err(e) if matches!(hook, "on_exception" | "on_stop") => {
                            self.error(py, reported, e);
                        }
                        Err(e) => return Err(e),
                    }
                }
                // `Context.run(method, *args)`.
                let mut call = vec![self.method(py, hook)?];
                call.extend(args(py));
                let call = PyTuple::new(py, call)?;
                self.context.bind(py).call_method1(intern!(py, "run"), call)
            })();
            result
                .map(Bound::unbind)
                .map_err(|e| self.error(py, reported, e))
        })
    }

    fn call0(&self, hook: &'static str) -> Result<(), HookError> {
        self.call(hook, |_| Vec::new()).map(drop)
    }
}

impl NodeHooks for PyNode {
    type State = Py<PyAny>;

    /// Attaches the thread to Python once for its whole life and detaches it
    /// for the body, so that each hook call on it only takes the GIL again (on
    /// a GIL build) instead of making a new Python thread state. On a dedicated
    /// thread the hooks of its node so share `threading.local` and
    /// `threading.current_thread()`; a worker shares them between the nodes it
    /// runs. The body waits without the GIL.
    fn on_thread<R: Send>(body: impl FnOnce() -> R + Send) -> R {
        Python::attach(|py| py.detach(body))
    }

    /// Calls `_on_enter`, whose error is reported as the error of `next_hook`.
    fn enter(&mut self, next_hook: &'static str) -> Result<(), HookError> {
        self.call_as("_on_enter", next_hook, |_| Vec::new())
            .map(drop)
    }

    fn panicked(&mut self, error: &HookError) {
        // `try_attach`: this runs outside the runtime's `catch_unwind`, and
        // `attach` panics while the interpreter finalizes.
        Python::try_attach(|py| {
            let message = format!("node '{}' panicked in {}", self.name, error.hook);
            log_error(
                py,
                &message,
                &PyRuntimeError::new_err(error.source.to_string()),
            );
        });
    }

    fn start(&mut self) -> Result<(), HookError> {
        let state = self.state.take();
        self.call0("on_start")?;
        if let Some(state) = state {
            self.call("load_state_dict", |py| vec![state.into_bound(py)])?;
        }
        // Look the hooks up last, so that the hooks above may still replace them
        // (by instance attributes).
        let methods = Python::attach(|py| {
            let node = self.node.bind(py);
            CACHED_HOOKS
                .iter()
                .map(|hook| node.getattr(attribute(hook)).map(Bound::unbind))
                .collect::<PyResult<_>>()
                .map_err(|e| self.error(py, "on_start", e))
        })?;
        self.methods = Some(methods);
        Ok(())
    }
    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.call("on_update", |py| vec![PyFloat::new(py, dt).into_any()])?;
        // The framework's per-step work: an error there is reported as
        // `on_update`'s.
        self.call_as("_after_update", "on_update", |_| Vec::new())
            .map(drop)
    }

    fn pause(&mut self) -> Result<(), HookError> {
        self.call0("on_pause")
    }

    fn resume(&mut self) -> Result<(), HookError> {
        self.call0("on_resume")
    }

    fn exception(&mut self) -> Result<(), HookError> {
        self.call0("on_exception")
    }

    fn stop(&mut self) -> Result<(), HookError> {
        self.call0("on_stop")
    }

    /// Outside the node's context: the host takes the state the same way after
    /// a normal end.
    fn state_dict(&mut self) -> Result<Py<PyAny>, HookError> {
        Python::attach(|py| {
            self.method(py, "state_dict")
                .and_then(|m| m.call0())
                .map(Bound::unbind)
                .map_err(|e| self.error(py, "state_dict", e))
        })
    }
}

/// The exception a failure is caused by: the hook's own exception, or a
/// `RuntimeError` describing a panic.
fn cause(py: Python<'_>, failure: &NodeFailure) -> PyErr {
    match failure.error.downcast_ref::<PyHookError>() {
        Some(e) => e.err.clone_ref(py),
        None => PyRuntimeError::new_err(failure.error.to_string()),
    }
}

fn runtime_error(py: Python<'_>, names: &[String], err: RuntimeError) -> PyErr {
    match err {
        RuntimeError::Failed(failure) => {
            let message = format!(
                "the network stopped because node '{}' raised in {}",
                names[failure.node], failure.hook
            );
            let err = NetworkError::new_err(message);
            err.set_cause(py, Some(cause(py, &failure)));
            err
        }
        RuntimeError::PauseTimeout { .. } => PauseTimeoutError::new_err(err.to_string()),
        RuntimeError::InvalidArgument(_) => PyValueError::new_err(err.to_string()),
        RuntimeError::NotManual
        | RuntimeError::Finished
        | RuntimeError::Reentrant
        | RuntimeError::Internal(_) => PyRuntimeError::new_err(err.to_string()),
    }
}

fn duration(seconds: f64) -> PyResult<Duration> {
    Ok(Duration::from_nanos(to_ns(seconds)?))
}

/// A period in seconds as the runtime's ns: rounded as `to_ns` rounds,
/// and at least 1 ns, since a period of 0 ns is invalid (0.0 s becomes 1 ns,
/// which updates as soon as any time has passed).
fn period_to_ns(seconds: f64) -> PyResult<u64> {
    Ok(to_ns(seconds)?.max(1))
}

/// The number of workers when `start` is given none, if not the core's
/// default (the available parallelism): half of it, at least 1, while the GIL
/// is enabled. With the GIL, more workers than that only contend for it: in the
/// benchmark (`benchmarks/mn_scheduler.py`, 14 cores), torch nodes updated
/// about twice as fast on 7 or 8 workers as on 14.
fn default_workers(py: Python<'_>) -> PyResult<Option<NonZeroUsize>> {
    let gil = py
        .import(intern!(py, "sys"))?
        .call_method0(intern!(py, "_is_gil_enabled"))?
        .extract::<bool>()?;
    if !gil {
        return Ok(None);
    }
    Ok(NonZeroUsize::new((available_parallelism() / 2).max(1)))
}

/// The available parallelism as the core reads it (on Linux it also counts a
/// CPU quota): the default number of workers for at least as many pool nodes
/// without the GIL. Private, for the tests.
#[pyfunction(name = "_available_parallelism")]
pub fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

/// Runs nodes on a pool of worker threads, or each dedicated node on a thread
/// of its own, under one control thread.
#[pyclass(frozen, module = "mujou._core._native")]
pub struct Runtime {
    /// The nodes before `start`.
    pending: Mutex<Option<Vec<Py<PyAny>>>>,
    names: Vec<String>,
    timer: Arc<Timer>,
    inner: OnceLock<core::Runtime<PyNode>>,
}

impl Runtime {
    fn core(&self) -> PyResult<&core::Runtime<PyNode>> {
        self.inner
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("the runtime has not started"))
    }

    /// Waits for a reply without the GIL, checking for signals in between.
    fn wait<T: Send>(&self, py: Python<'_>, reply: Reply<T>) -> PyResult<T> {
        // `Reply` is not `Sync`, so the closure owns it instead of borrowing it
        // across the detach.
        let result = wait_interruptibly(py, None, move |step| reply.wait_timeout(step))?;
        let result = result.expect("a wait without a deadline ends only with the reply");
        result.map_err(|e| runtime_error(py, &self.names, e))
    }
}

#[pymethods]
impl Runtime {
    /// The nodes run on `timer`: it measures their Δt, and the runtime freezes
    /// it to start and to take the state. Create the `Outlet`s and the
    /// `InputGroup` of this run with the same timer.
    #[new]
    #[pyo3(signature = (nodes, names, *, timer))]
    fn new(
        nodes: Vec<Bound<'_, Node>>,
        names: Vec<String>,
        timer: PyRef<'_, PyTimer>,
    ) -> PyResult<Self> {
        if nodes.len() != names.len() {
            return Err(PyValueError::new_err(format!(
                "got {} nodes but {} names",
                nodes.len(),
                names.len()
            )));
        }
        let nodes = nodes.into_iter().map(|n| n.into_any().unbind()).collect();
        Ok(Self {
            pending: Mutex::new(Some(nodes)),
            names,
            timer: Arc::clone(&timer.0),
            inner: OnceLock::new(),
        })
    }

    /// Starts every node and returns once all of them have started (and
    /// restored `states`) and the timer runs again. If a node raised meanwhile,
    /// raises `NetworkError` after the network has stopped. If an argument is
    /// invalid, raises `ValueError` and nothing starts, so `start` can be
    /// called again.
    ///
    /// The timer is frozen first and loaded from `timer_state`
    /// (`Timer.state_dict()`), so the time does not run while the nodes start.
    /// Loading it raises `ValueError` while another runtime runs on the same
    /// timer, whose time would jump. `last_update_ns` is where each node's
    /// first Δt starts (the time at the start by default). `periods` is each
    /// node's period on the timer in seconds (>= 0), rounded to the nearest ns
    /// and at least 1 ns, so that 0.0 updates as soon as any time has passed,
    /// as every node does without `periods`; the manual mode ignores it.
    ///
    /// `dedicated` tells which nodes run on a thread of their own; the others,
    /// all of them by default, share a pool of `workers` threads (>= 1), by
    /// default as many as the available parallelism (half of it, at least 1,
    /// while the GIL is enabled), never more than the nodes in the pool.
    ///
    /// If the wait is interrupted (Ctrl-C), the start goes on in the
    /// background; `shutdown` stops it.
    #[pyo3(signature = (
        *,
        manual = false,
        states = None,
        last_update_ns = None,
        periods = None,
        timer_state: "dict[str, int | float] | None" = None,
        workers = None,
        dedicated = None,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "the keyword arguments of the Python method"
    )]
    fn start(
        &self,
        py: Python<'_>,
        manual: bool,
        states: Option<Vec<Py<PyAny>>>,
        last_update_ns: Option<Vec<u64>>,
        periods: Option<Vec<f64>>,
        timer_state: Option<Bound<'_, PyAny>>,
        workers: Option<usize>,
        dedicated: Option<Vec<bool>>,
    ) -> PyResult<()> {
        let n = self.names.len();
        let workers = match workers {
            Some(w) => Some(
                NonZeroUsize::new(w)
                    .ok_or_else(|| PyValueError::new_err("workers must be at least 1, got 0"))?,
            ),
            None => default_workers(py)?,
        };
        if let Some(states) = &states {
            if states.len() != n {
                return Err(PyValueError::new_err(format!(
                    "got {n} nodes but {} states",
                    states.len()
                )));
            }
        }
        let timer_state = timer_state.as_ref().map(extract_state).transpose()?;
        let period_ns = periods
            .map(|periods| {
                periods
                    .into_iter()
                    .map(period_to_ns)
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let config = RuntimeConfig {
            manual,
            last_update_ns,
            period_ns,
            timer_state,
            workers,
            dedicated,
        };
        // An empty context for each node, as a new thread has.
        let context = py.import("contextvars")?.getattr("Context")?;
        let contexts = (0..n)
            .map(|_| context.call0().map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let nodes = pending
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("the runtime has already started"))?;
        // Kept to undo the take if the core rejects the arguments.
        let kept = nodes.iter().map(|n| n.clone_ref(py)).collect();
        let mut states = states.into_iter().flatten();
        let hooks = (nodes.into_iter().zip(self.names.iter().cloned()))
            .zip(contexts)
            .map(|((node, name), context)| PyNode {
                node,
                name,
                context,
                state: states.next(),
                methods: None,
            })
            .collect();
        let (runtime, started) = match core::Runtime::start(
            hooks,
            self.names.clone(),
            Arc::clone(&self.timer),
            config,
        ) {
            Ok(started) => started,
            Err(e) => {
                if matches!(e, RuntimeError::InvalidArgument(_)) {
                    *pending = Some(kept);
                }
                return Err(runtime_error(py, &self.names, e));
            }
        };
        drop(pending);
        let _ = self.inner.set(runtime);
        self.wait(py, started)
    }

    /// Manual mode: lets the nodes in `order` (all, in index order, if `None`)
    /// update once each, one after another.
    #[pyo3(signature = (order = None))]
    fn step(&self, py: Python<'_>, order: Option<Vec<usize>>) -> PyResult<()> {
        let reply = self.core()?.step(order.as_deref());
        self.wait(py, reply)
    }

    /// Pauses every node, calls `build` with their `state_dict()` in node
    /// order while paused, resumes, and returns what `build` returned.
    ///
    /// `build` runs on the control thread and must not call this runtime's
    /// waiting methods. It may capture another runtime, but two builds must
    /// not wait for each other's runtime: that cycle deadlocks, and the control
    /// threads cannot see Ctrl-C. If `build` raises, the exception propagates
    /// and the network keeps running. Gives up with `PauseTimeoutError` after
    /// `timeout` real seconds if some node has not paused.
    fn capture(&self, py: Python<'_>, build: Build, timeout: f64) -> PyResult<Py<PyAny>> {
        let Build(build) = build;
        let timeout = duration(timeout)?;
        let reply = self.core()?.capture(
            move |states| {
                Python::attach(|py| {
                    let states = PyList::new(py, states)?;
                    build.call1(py, (states,))
                })
            },
            timeout,
        );
        self.wait(py, reply)?
    }

    /// Stops every node and waits until the network has stopped. Does nothing
    /// before `start` or after a normal stop.
    fn shutdown(&self, py: Python<'_>) -> PyResult<()> {
        let Some(runtime) = self.inner.get() else {
            return Ok(());
        };
        let reply = runtime.shutdown();
        self.wait(py, reply)
    }

    /// Asks every node to stop and returns without waiting; `join` waits for
    /// the end. Does nothing before `start` or once the network has stopped.
    fn request_stop(&self) {
        if let Some(runtime) = self.inner.get() {
            runtime.request_stop();
        }
    }

    /// Waits until the network has stopped, for at most `timeout` seconds, and
    /// returns whether it has.
    #[pyo3(signature = (timeout = None))]
    fn join(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let runtime = self.core()?;
        let timeout = timeout.map(duration).transpose()?;
        let deadline = timeout.and_then(|t| Instant::now().checked_add(t));
        let ended = wait_interruptibly(py, deadline, |step| match runtime.join(Some(step)) {
            Ok(true) => Some(Ok(())),
            Ok(false) => None,
            Err(e) => Some(Err(e)),
        })?;
        match ended {
            Some(Ok(())) => Ok(true),
            None => Ok(false),
            Some(Err(e)) => Err(runtime_error(py, &self.names, e)),
        }
    }

    /// Raises `NetworkError` if a node has raised.
    fn check(&self, py: Python<'_>) -> PyResult<()> {
        match self.inner.get().and_then(core::Runtime::failure) {
            Some(failure) => Err(runtime_error(
                py,
                &self.names,
                RuntimeError::Failed(failure.clone()),
            )),
            None => Ok(()),
        }
    }

    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.get().is_some_and(core::Runtime::is_finished)
    }

    /// The time of each node's last update on the runtime's timer, in
    /// nanoseconds.
    fn last_update_ns(&self) -> PyResult<Vec<u64>> {
        Ok(self.core()?.last_update_ns())
    }
}
