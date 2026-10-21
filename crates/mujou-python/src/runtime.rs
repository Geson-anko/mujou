//! Bindings of `mujou_core::runtime`: Python nodes on native threads, each run
//! through its driver.
//!
//! Every wait on the host releases the GIL (detaches) and checks for signals
//! every `SIGNAL_CHECK_INTERVAL`, like `sleep`: the node and control threads
//! attach to call the hooks, so a host that kept the GIL while waiting for them
//! would deadlock on a GIL build.

use std::error::Error;
use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use mujou_core::runtime::{
    self as core, Hook, HookError, NodeFailure, NodeHooks, NodeSpec, Reply, RunConfig, RuntimeError,
};
use pyo3::call::PyCallArgs;
use pyo3::exceptions::{PyRuntimeError, PyTimeoutError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::current;
use crate::errors::NetworkError;
use crate::hint::Build;
use crate::time::PyReadOnlyTimer;
use crate::wait::{to_duration, to_ns, wait_interruptibly};

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

/// The method of the driver that runs `hook`.
fn method(py: Python<'_>, hook: Hook) -> &Bound<'_, PyString> {
    match hook {
        Hook::Start => intern!(py, "start"),
        Hook::Restore => intern!(py, "restore"),
        Hook::Update => intern!(py, "update"),
        Hook::Pause => intern!(py, "pause"),
        Hook::Resume => intern!(py, "resume"),
        Hook::Exception => intern!(py, "exception"),
        Hook::Stop => intern!(py, "stop"),
        Hook::Capture => intern!(py, "capture"),
    }
}

/// A node's driver (`mujou.driver.Driver`) as the runtime sees it: each
/// hook is one call of the driver's method for it, which `current()` returns
/// the driver during.
struct PyDriver {
    driver: Py<PyAny>,
    name: String,
    /// The state to restore in `restore`.
    state: Option<Py<PyAny>>,
}

impl PyDriver {
    /// Every hook error is logged where it is caught, including the ones that
    /// do not end the system (a second exception, `on_exception`, `on_stop`).
    fn error(&self, py: Python<'_>, hook: Hook, err: PyErr) -> HookError {
        let message = format!("node '{}' raised in {hook}", self.name);
        log_error(py, &message, &err);
        HookError {
            hook,
            source: Arc::new(PyHookError { err, message }),
        }
    }

    /// Calls the driver's method for `hook` with `args`, as the hook of the
    /// node, and returns what `keep` keeps of what it returned.
    ///
    /// `keep` runs attached, so that dropping the result there releases it at
    /// once: dropped detached, pyo3 would queue the release in its one global
    /// pool, which every thread that runs a node then contends for.
    fn call<A, T>(
        &self,
        hook: Hook,
        args: A,
        keep: impl for<'py> FnOnce(Bound<'py, PyAny>) -> T,
    ) -> Result<T, HookError>
    where
        A: for<'py> PyCallArgs<'py>,
    {
        // Every runtime thread keeps its Python thread state for its whole life
        // (`on_thread`), so this attach only takes the GIL (on a GIL build).
        Python::attach(|py| {
            current::running(&self.driver, py, || {
                self.driver.bind(py).call_method1(method(py, hook), args)
            })
            .map(keep)
            .map_err(|e| self.error(py, hook, e))
        })
    }
}

/// What a hook returns but the capture: nothing to keep.
fn ignore(_: Bound<'_, PyAny>) {}

impl NodeHooks for PyDriver {
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
        self.call(Hook::Start, (), ignore)
    }

    /// Restores the state, if any: no call without one.
    fn restore(&mut self) -> Result<(), HookError> {
        match self.state.take() {
            Some(state) => self.call(Hook::Restore, (state,), ignore),
            None => Ok(()),
        }
    }

    fn update(&mut self, dt: f64) -> Result<(), HookError> {
        self.call(Hook::Update, (dt,), ignore)
    }

    fn pause(&mut self) -> Result<(), HookError> {
        self.call(Hook::Pause, (), ignore)
    }

    fn resume(&mut self) -> Result<(), HookError> {
        self.call(Hook::Resume, (), ignore)
    }

    fn exception(&mut self) -> Result<(), HookError> {
        self.call(Hook::Exception, (), ignore)
    }

    fn stop(&mut self) -> Result<(), HookError> {
        self.call(Hook::Stop, (), ignore)
    }

    fn capture(&mut self) -> Result<Py<PyAny>, HookError> {
        self.call(Hook::Capture, (), |state| state.unbind())
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

fn runtime_error(py: Python<'_>, err: RuntimeError) -> PyErr {
    match err {
        RuntimeError::Failed(failure) => {
            let message = format!(
                "the network stopped because node '{}' raised in {}",
                failure.name, failure.hook
            );
            let err = NetworkError::new_err(message);
            err.set_cause(py, Some(cause(py, &failure)));
            err
        }
        // The run had ended normally: the host's call raises what the node's
        // `state_dict` raised, as any call on its thread would.
        RuntimeError::CaptureFailed(failure) => cause(py, &failure),
        RuntimeError::PauseTimeout { .. } => PyTimeoutError::new_err(err.to_string()),
        RuntimeError::Finished | RuntimeError::Reentrant | RuntimeError::Internal(_) => {
            PyRuntimeError::new_err(err.to_string())
        }
    }
}

/// A period in seconds as the runtime's ns: rounded as `to_ns` rounds, at
/// least 1 ns (0.0 s becomes 1 ns, which updates as soon as any time has
/// passed), and at most `u64::MAX` ns, which infinity is too, as any period
/// longer than that.
fn period_to_ns(seconds: f64) -> PyResult<NonZeroU64> {
    let ns = if seconds == f64::INFINITY {
        u64::MAX
    } else {
        to_ns(seconds)?
    };
    Ok(NonZeroU64::new(ns).unwrap_or(NonZeroU64::MIN))
}

/// The number of workers when `start` is given none: the available
/// parallelism (which on Linux also counts a CPU quota), or half of it, at
/// least 1, while the GIL is enabled. With the GIL, more workers than that
/// only contend for it: in the benchmark (`benchmarks/mn_scheduler.py`, 14
/// cores), torch nodes updated about twice as fast on 7 or 8 workers as on 14.
fn default_workers(py: Python<'_>) -> PyResult<NonZeroUsize> {
    let gil = py
        .import(intern!(py, "sys"))?
        .call_method0(intern!(py, "_is_gil_enabled"))?
        .extract::<bool>()?;
    let cpus = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    if !gil {
        return Ok(cpus);
    }
    Ok(NonZeroUsize::new(cpus.get() / 2).unwrap_or(NonZeroUsize::MIN))
}

/// A node of `start`: `(driver, name, period, dedicated_thread,
/// last_update_ns, state)`.
type Record<'py> = (
    Bound<'py, PyAny>,
    String,
    f64,
    bool,
    Option<u64>,
    Option<Py<PyAny>>,
);

/// Starts `nodes` on `clock` and returns once all of them have started (and
/// restored their states) and the time runs again. If a node raised
/// meanwhile, raises `NetworkError` after the network has stopped.
///
/// Each node is a record `(driver, name, period, dedicated_thread,
/// last_update_ns, state)`. The runtime runs each hook of the node as one call
/// of a method of `driver` (`mujou.driver.Driver`): `start()`,
/// `restore(state)`, `update(dt)`, `pause()`, `resume()`, `exception()`,
/// `stop()` and `capture()`, which returns the node's state. `name` names the
/// node in the errors and the logs, which name the hooks as the node's author
/// knows them (`on_update`, `load_state_dict`, `state_dict`, ...). `period` is
/// its period on the timer in seconds (>= 0), rounded to the nearest ns, at
/// least 1 ns, so that 0.0 updates as soon as any time has passed, and at most
/// `u64::MAX` ns (infinity included); the manual mode ignores it.
/// `dedicated_thread` runs the node on a thread of its own instead of the
/// pool. `last_update_ns` is where its first Δt starts (the
/// time at the start if `None`), and `state` what `restore` restores after
/// `start` (no call if `None`).
///
/// The nodes run on `clock`, the reading side of the run's timer: it measures
/// their Δt, and the runtime freezes its timer to start (so the time does not
/// run while the nodes start) and to take the state. Create the `Outlet`s and
/// the `InputGroup` of this run with the same clock. The runtime stops the
/// clock when it is asked to stop or a node raises, so that no time of the run
/// is later than its end.
///
/// `current()` returns the driver while one of its methods runs.
///
/// The nodes not on a thread of their own share a pool of `workers` threads,
/// by default as many as the available parallelism (half of it, at least 1,
/// while the GIL is enabled), never more than the nodes in the pool.
///
/// If the wait is interrupted (Ctrl-C), stops the network and raises once it
/// has stopped; interrupting that wait too leaves it stopping in the background.
#[pyfunction]
#[pyo3(signature = (nodes, clock, *, manual = false, workers = None))]
pub fn start(
    py: Python<'_>,
    nodes: Vec<Record<'_>>,
    clock: Bound<'_, PyReadOnlyTimer>,
    manual: bool,
    workers: Option<NonZeroUsize>,
) -> PyResult<Runtime> {
    let workers = match workers {
        Some(workers) => workers,
        None => default_workers(py)?,
    };
    let nodes = (nodes.into_iter())
        .map(
            |(driver, name, period, dedicated_thread, last_update_ns, state)| {
                Ok(NodeSpec {
                    hooks: PyDriver {
                        driver: driver.unbind(),
                        name: name.clone(),
                        state,
                    },
                    name,
                    period_ns: period_to_ns(period)?,
                    dedicated_thread,
                    last_update_ns,
                })
            },
        )
        .collect::<PyResult<_>>()?;
    let config = RunConfig {
        nodes,
        manual,
        workers,
    };
    let (runtime, started) = core::Runtime::start(config, Arc::clone(&clock.get().0))
        .map_err(|e| runtime_error(py, e))?;
    if let Err(e) = wait(py, started) {
        // A node raised, and the network has stopped, or the wait was
        // interrupted: stop and wait for the end, so that no node runs once
        // this has raised (nor into the interpreter's exit).
        runtime.stop();
        wait_interruptibly(py, None, |step| {
            runtime
                .join(Some(step))
                .map_or(Some(()), |ended| ended.then_some(()))
        })?;
        return Err(e);
    }
    Ok(Runtime(runtime))
}

/// Waits for a reply without the GIL, checking for signals in between.
fn wait<T: Send>(py: Python<'_>, reply: Reply<T>) -> PyResult<T> {
    // `Reply` is not `Sync`, so the closure owns it instead of borrowing it
    // across the detach.
    let result = wait_interruptibly(py, None, move |step| reply.wait_timeout(step))?;
    let result = result.expect("a wait without a deadline ends only with the reply");
    result.map_err(|e| runtime_error(py, e))
}

/// The running nodes of one run, from `start`: on a pool of worker threads,
/// or each dedicated node on a thread of its own, under one control thread.
#[pyclass(frozen, module = "mujou._native")]
pub struct Runtime(core::Runtime<PyDriver>);

#[pymethods]
impl Runtime {
    /// Manual mode: lets the nodes in `order` (all, in index order, if `None`)
    /// update once each, one after another.
    #[pyo3(signature = (order = None))]
    fn step(&self, py: Python<'_>, order: Option<Vec<usize>>) -> PyResult<()> {
        wait(py, self.0.step(order.as_deref()))
    }

    /// Pauses every node, calls `build` with the states their drivers'
    /// `capture()` return, in node order, while paused, resumes, and returns
    /// what `build` returned. After a normal end, calls their `capture()` and
    /// `build` on this thread instead, at once.
    ///
    /// `build` runs on the control thread and must not call this runtime's
    /// waiting methods. It may capture another runtime, but two builds must
    /// not wait for each other's runtime: that cycle deadlocks, and the control
    /// threads cannot see Ctrl-C. If `build` raises, the exception propagates
    /// and the network keeps running. Gives up with `TimeoutError` after
    /// `timeout` real seconds if some node has not paused; the network keeps
    /// running.
    fn capture(&self, py: Python<'_>, build: Build, timeout: f64) -> PyResult<Py<PyAny>> {
        let Build(build) = build;
        let timeout = to_duration(timeout)?;
        let build = move |states| {
            Python::attach(|py| {
                let states = PyList::new(py, states)?;
                build.call1(py, (states,))
            })
        };
        // Detached: after the end, this thread takes the states itself and may
        // wait for a node that another thread captures meanwhile, which holds
        // the GIL to run `capture`.
        let reply = py.detach(|| self.0.capture(build, timeout));
        wait(py, reply)?
    }

    /// Asks every node to stop, then waits as `join` does and returns whether
    /// the network has stopped. After a timeout the stop stays asked. A bad
    /// `timeout` raises before anything changes.
    #[pyo3(signature = (timeout = None))]
    fn shutdown(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let timeout = timeout.map(to_duration).transpose()?;
        self.0.stop();
        self.wait_for_end(py, timeout)
    }

    /// Waits until the network has stopped, for at most `timeout` seconds (no
    /// limit if `None`), and returns whether it has. Raises `NetworkError` if
    /// it stopped because a node raised.
    #[pyo3(signature = (timeout = None))]
    fn join(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let timeout = timeout.map(to_duration).transpose()?;
        self.wait_for_end(py, timeout)
    }

    /// Raises `NetworkError` if a node has raised.
    fn check(&self, py: Python<'_>) -> PyResult<()> {
        self.0.check().map_err(|e| runtime_error(py, e))
    }

    /// The network has stopped.
    #[getter]
    fn finished(&self) -> bool {
        self.0.finished()
    }

    /// The number of worker threads of the pool: the default or the given
    /// number, but not more than the nodes in the pool (0 if every node has a
    /// thread of its own).
    #[getter]
    fn workers(&self) -> usize {
        self.0.workers()
    }

    /// The time of each node's last update on the runtime's timer, in
    /// nanoseconds.
    fn last_update_ns(&self) -> Vec<u64> {
        self.0.last_update_ns()
    }
}

impl Runtime {
    fn wait_for_end(&self, py: Python<'_>, timeout: Option<Duration>) -> PyResult<bool> {
        let deadline = timeout.and_then(|t| Instant::now().checked_add(t));
        let runtime = &self.0;
        let ended = wait_interruptibly(py, deadline, |step| match runtime.join(Some(step)) {
            Ok(true) => Some(Ok(())),
            Ok(false) => None,
            Err(e) => Some(Err(e)),
        })?;
        match ended {
            Some(Ok(())) => Ok(true),
            None => Ok(false),
            Some(Err(e)) => Err(runtime_error(py, e)),
        }
    }
}
