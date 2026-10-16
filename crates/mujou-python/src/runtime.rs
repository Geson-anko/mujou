//! Bindings of `mujou_core::runtime`: Python nodes on native threads.
//!
//! Every wait on the host releases the GIL (detaches) and checks for signals
//! every `SIGNAL_CHECK_INTERVAL`, like `sleep`: the node and control threads
//! attach to call the hooks, so a host that kept the GIL while waiting for them
//! would deadlock on a GIL build.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use mujou_core::runtime::{
    self as core, ExceptionCapture, HookError, NodeFailure, NodeHooks, Reply, RuntimeConfig,
    RuntimeError,
};
use mujou_core::time::{Timer, seconds_to_ns};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::hint::Build;
use crate::time::{PyTimer, extract_state, timer_or_global};
use crate::{SIGNAL_CHECK_INTERVAL, timer_error};

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
/// The runtime calls the hooks on the node's own thread (`state_dict`, through
/// `_capture_state`, on the control thread while the node is paused). They do
/// nothing by default. The hooks other than `on_thread_start`, `on_start` and
/// `load_state_dict` are looked up once, after those three, so replacing one
/// there takes effect. `state_dict` itself is looked up by the default
/// `_capture_state` at each capture, so replacing it later takes effect too.
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

    /// Called on the node's thread right before `on_start`, for per-thread
    /// settings. A subclass that overrides it calls the base's first, so that
    /// the base's settings are not skipped.
    fn on_thread_start(&self) {}

    fn on_start(&self) {}

    #[pyo3(signature = (dt: "float"))]
    fn on_update(&self, dt: f64) {
        let _ = dt;
    }

    /// Called on the node's thread right after each `on_update`, for the
    /// framework's per-step work (so that overriding `on_update` cannot skip it).
    fn _after_update(&self) {}

    fn on_pause(&self) {}

    fn on_resume(&self) {}

    fn on_exception(&self) {}

    fn on_stop(&self) {}

    #[pyo3(signature = () -> "object")]
    fn state_dict(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    #[pyo3(signature = (state: "object"))]
    fn load_state_dict(&self, state: &Bound<'_, PyAny>) {
        let _ = state;
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

/// The hooks called more than once, looked up once on the node's thread at the
/// end of `start` instead of by name at every call.
struct Methods {
    on_update: Py<PyAny>,
    after_update: Py<PyAny>,
    on_pause: Py<PyAny>,
    on_resume: Py<PyAny>,
    on_exception: Py<PyAny>,
    on_stop: Py<PyAny>,
    capture_state: Py<PyAny>,
}

impl Methods {
    fn of(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        let get = |hook: &str| node.getattr(attribute(hook)).map(Bound::unbind);
        Ok(Self {
            on_update: get("on_update")?,
            after_update: get("_after_update")?,
            on_pause: get("on_pause")?,
            on_resume: get("on_resume")?,
            on_exception: get("on_exception")?,
            on_stop: get("on_stop")?,
            capture_state: get("state_dict")?,
        })
    }

    fn get(&self, hook: &str) -> Option<&Py<PyAny>> {
        match hook {
            "on_update" => Some(&self.on_update),
            "_after_update" => Some(&self.after_update),
            "on_pause" => Some(&self.on_pause),
            "on_resume" => Some(&self.on_resume),
            "on_exception" => Some(&self.on_exception),
            "on_stop" => Some(&self.on_stop),
            "state_dict" => Some(&self.capture_state),
            _ => None,
        }
    }
}

/// The attribute the runtime calls for `hook`. Taking the state goes through
/// `_capture_state`, which calls `state_dict`; errors still name `state_dict`.
fn attribute(hook: &str) -> &str {
    match hook {
        "state_dict" => "_capture_state",
        _ => hook,
    }
}

/// A Python node seen by the runtime.
struct PyNode {
    node: Py<PyAny>,
    name: String,
    /// The state to restore in `start`.
    state: Option<Py<PyAny>>,
    methods: Option<Methods>,
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

    fn call(
        &self,
        hook: &'static str,
        args: impl for<'py> FnOnce(Python<'py>) -> Bound<'py, PyTuple>,
    ) -> Result<Py<PyAny>, HookError> {
        // The node's thread keeps its Python thread state for its whole life
        // (`on_thread`), so this attach only takes the GIL (on a GIL build).
        Python::attach(|py| {
            let result = match self.methods.as_ref().and_then(|m| m.get(hook)) {
                Some(method) => method.bind(py).call1(args(py)),
                // Before the lookup, or if it failed: by name.
                None => self.node.bind(py).call_method1(attribute(hook), args(py)),
            };
            result
                .map(Bound::unbind)
                .map_err(|e| self.error(py, hook, e))
        })
    }

    fn call0(&self, hook: &'static str) -> Result<(), HookError> {
        self.call(hook, PyTuple::empty).map(drop)
    }
}

impl NodeHooks for PyNode {
    type State = Py<PyAny>;

    /// Attaches the thread to Python once for its whole life and detaches it
    /// for the body, so that the hooks of a node share one Python thread state:
    /// `threading.local`, `contextvars` and `threading.current_thread()` set or
    /// seen in `on_start` are the same in `on_update`. The body waits without
    /// the GIL; each hook call takes it again on this same thread state.
    fn on_thread<R: Send>(body: impl FnOnce() -> R + Send) -> R {
        Python::attach(|py| py.detach(body))
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
        self.call0("on_thread_start")?;
        self.call0("on_start")?;
        if let Some(state) = state {
            self.call("load_state_dict", |py| {
                PyTuple::new(py, [state]).expect("a tuple of one object")
            })?;
        }
        // Look the hooks up last, so that `on_thread_start`, `on_start` and
        // `load_state_dict` may still replace them (by instance attributes).
        let methods = Python::attach(|py| Methods::of(self.node.bind(py)));
        self.methods =
            Some(methods.map_err(|e| Python::attach(|py| self.error(py, "on_start", e)))?);
        Ok(())
    }
    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.call("on_update", |py| {
            PyTuple::new(py, [dt]).expect("a tuple of one float")
        })?;
        self.call0("_after_update")
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

    fn state_dict(&mut self) -> Result<Py<PyAny>, HookError> {
        self.call("state_dict", PyTuple::empty)
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
    Ok(Duration::from_nanos(
        seconds_to_ns(seconds).map_err(timer_error)?,
    ))
}

/// Runs nodes, each on its own thread, under one control thread.
#[pyclass(frozen, module = "mujou._core._native")]
pub struct Runtime {
    /// The nodes before `start`.
    pending: Mutex<Option<Vec<Py<PyAny>>>>,
    names: Vec<String>,
    timer: Arc<Timer>,
    inner: OnceLock<core::Runtime<PyNode>>,
    exception_state: Arc<OnceLock<Py<PyAny>>>,
}

impl Runtime {
    fn core(&self) -> PyResult<&core::Runtime<PyNode>> {
        self.inner
            .get()
            .ok_or_else(|| PyRuntimeError::new_err("the runtime has not started"))
    }

    /// Waits for a reply without the GIL, checking for signals in between.
    fn wait<T: Send>(&self, py: Python<'_>, mut reply: Reply<T>) -> PyResult<T> {
        loop {
            // `Reply` is not `Sync`, so it moves in and out instead of being
            // borrowed across the detach.
            let (back, result) = py.detach(move || {
                let result = reply.wait_timeout(SIGNAL_CHECK_INTERVAL);
                (reply, result)
            });
            if let Some(result) = result {
                return result.map_err(|e| runtime_error(py, &self.names, e));
            }
            reply = back;
            py.check_signals()?;
        }
    }

    fn exception_capture(&self, callback: Py<PyAny>) -> ExceptionCapture<Py<PyAny>> {
        let store = Arc::clone(&self.exception_state);
        Box::new(
            move |states: BTreeMap<usize, Py<PyAny>>, failure: &NodeFailure| {
                Python::attach(|py| {
                    let result = (|| -> PyResult<Py<PyAny>> {
                        let dict = PyDict::new(py);
                        for (i, state) in states {
                            dict.set_item(i, state)?;
                        }
                        let exc = cause(py, failure).into_value(py);
                        callback.call1(py, (dict, failure.node, failure.hook, exc))
                    })();
                    match result {
                        Ok(value) => {
                            let _ = store.set(value);
                        }
                        Err(e) => log_error(py, "on_exception_capture raised", &e),
                    }
                });
            },
        )
    }
}

#[pymethods]
impl Runtime {
    /// The nodes run on `timer` (the global timer if `None`): it measures their
    /// Δt, and the runtime freezes it to start, to take the state, and on an
    /// exception. Create the `Outlet`s and the `InputGroup` of this run with
    /// the same timer.
    #[new]
    #[pyo3(signature = (nodes, names, *, timer = None))]
    fn new(
        nodes: Vec<Bound<'_, Node>>,
        names: Vec<String>,
        timer: Option<PyRef<'_, PyTimer>>,
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
            timer: timer_or_global(timer),
            inner: OnceLock::new(),
            exception_state: Arc::new(OnceLock::new()),
        })
    }

    /// Starts every node on its own thread and returns once all of them have
    /// started (and restored `states`) and the timer runs again. If a node
    /// raised meanwhile, raises `NetworkError` after the network has stopped.
    /// If an argument is invalid, raises `ValueError` and nothing starts, so
    /// `start` can be called again.
    ///
    /// The timer is frozen first and loaded from `timer_state`
    /// (`Timer.state_dict()`), so the time does not run while the nodes start.
    /// Loading it raises `ValueError` while another runtime runs on the same
    /// timer, whose time would jump. `last_update_ns` is where each node's
    /// first Δt starts (the time at the start by default).
    /// `on_exception_capture(states, node, hook, exception)` takes the states
    /// (by node index, without those whose `state_dict` failed) when a node
    /// raises.
    ///
    /// If the wait is interrupted (Ctrl-C), the start goes on in the
    /// background; `shutdown` stops it.
    #[pyo3(signature = (
        *,
        manual = false,
        states = None,
        last_update_ns = None,
        timer_state: "dict[str, int | float] | None" = None,
        pause_timeout = 10.0,
        on_exception_capture = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn start(
        &self,
        py: Python<'_>,
        manual: bool,
        states: Option<Vec<Py<PyAny>>>,
        last_update_ns: Option<Vec<u64>>,
        timer_state: Option<Bound<'_, PyAny>>,
        pause_timeout: f64,
        on_exception_capture: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let n = self.names.len();
        if let Some(states) = &states {
            if states.len() != n {
                return Err(PyValueError::new_err(format!(
                    "got {n} nodes but {} states",
                    states.len()
                )));
            }
        }
        let timer_state = timer_state.as_ref().map(extract_state).transpose()?;
        let config = RuntimeConfig {
            manual,
            pause_timeout: duration(pause_timeout)?,
            last_update_ns,
            timer_state,
        };
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let nodes = pending
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("the runtime has already started"))?;
        // Kept to undo the take if the core rejects the arguments.
        let kept = nodes.iter().map(|n| n.clone_ref(py)).collect();
        let mut states = states.map(Vec::into_iter);
        let hooks = (nodes.into_iter().zip(self.names.iter().cloned()))
            .map(|(node, name)| PyNode {
                node,
                name,
                state: states.as_mut().and_then(Iterator::next),
                methods: None,
            })
            .collect();
        let capture = on_exception_capture.map(|callback| self.exception_capture(callback));
        let (runtime, started) = match core::Runtime::start(
            hooks,
            self.names.clone(),
            Arc::clone(&self.timer),
            config,
            capture,
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
    /// and the network keeps running.
    #[pyo3(signature = (build, *, timeout = None))]
    fn capture(&self, py: Python<'_>, build: Build, timeout: Option<f64>) -> PyResult<Py<PyAny>> {
        let timeout = timeout.map(duration).transpose()?;
        let Build(build) = build;
        let reply = self.core()?.capture(timeout, move |states| {
            Python::attach(|py| {
                let states = PyList::new(py, states)?;
                build.call1(py, (states,))
            })
        });
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
        loop {
            let remaining = deadline.map_or(SIGNAL_CHECK_INTERVAL, |d| {
                d.saturating_duration_since(Instant::now())
            });
            let wait = remaining.min(SIGNAL_CHECK_INTERVAL);
            match py.detach(|| runtime.join(Some(wait))) {
                Ok(true) => return Ok(true),
                Ok(false) if remaining.is_zero() => return Ok(false),
                Ok(false) => py.check_signals()?,
                Err(e) => return Err(runtime_error(py, &self.names, e)),
            }
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
    fn is_running(&self) -> bool {
        self.inner.get().is_some_and(core::Runtime::is_running)
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

    /// What `on_exception_capture` returned, if it was called and returned.
    fn exception_state(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.exception_state.get().map(|state| state.clone_ref(py))
    }
}
