# 🚨 Errors and Shutdown

This page shows what happens when a hook of a node raises, and where the host receives the exception. It also covers the timeouts for waiting for the end and for taking the state.

## One node's exception stops everything

Whichever hook of whichever node raises (not only `on_update`, but also `on_start`, `on_pause`, `state_dict` and the rest), every node of the run stops. It goes as follows.

1. Mujou catches the exception and logs the node's path, the hook's name and the traceback to the logger `"mujou"` at ERROR
2. Each node finishes the hook it is running, stops, and calls its own `on_exception()` without waiting for the other nodes. Nodes that did not raise are called too
3. Once `on_exception()` has finished on every node, `on_stop()` is called on every node. The node that raised is called too
4. The host gets a `NetworkError`. Its `__cause__` is the first exception. If several nodes raise at almost the same time, the exceptions after the first are only logged

!!! note "Why everything stops"

    Mujou never stops only the node that raised and keeps the others running. Its neighbors would keep reading the stopped node's old values and drift away from the equilibrium without anyone noticing.

## Where the host receives it

| The exception happens                    | What raises `NetworkError`                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| While starting (`on_start` and the like) | `mujou.run` itself. It finishes the ending work (`on_exception`, `on_stop`) before it raises                                                                              |
| After the start                          | The next `write`, `read`, `subscribe` (and its iterator), `wait_for`, `state_dict`, `shutdown` or `join` the host calls, or `step` of a `ManualRun` (`mujou.testing.run`) |

An exception after the start does not reach the host until it next calls a method of `Run`.

!!! warning "Wait with run.join, not sleep"

    If the host waits without calling anything, it does not notice the exception. To wait until the run stops, use `run.join(timeout)`, not `run.timer.sleep`.

```python
--8<-- "examples/errors.py"
```

`NetworkError` is a subclass of `RuntimeError`. To tell causes apart by type, look at `error.__cause__`.

### Inside a with statement

The `with` statement calls `run.shutdown()` when it exits.

- If the body of the `with` finishes without an exception: the node's exception arrives at the end of the `with` statement, as the `NetworkError` raised by `shutdown()`
- If the body of the `with` raises too: the body's exception propagates as it is, and the `NetworkError` is discarded. Check the node's exception in the log (see "Logging" below)

## on_exception and on_stop

| Hook             | Called                                                                    | What to write                                                      |
| ---------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `on_exception()` | Only when the run ends with an exception. On every node, before `on_stop` | Work specific to an abnormal end, such as a safe stop of actuators |
| `on_stop()`      | Once at every end, normal or by an exception                              | Release of resources such as I/O. Do not change the state          |

- `on_exception` takes no arguments. A node cannot know which exception of which node stopped the run. The host handles the cause, by looking at `__cause__`
- If you only want to handle an exception in your own hook, use `try`/`except` in that hook. An exception you catch does not stop the run
- If `on_exception` or `on_stop` raises, the exception is only logged, and the ending goes on
- The time of the run stands still once the run ends, so `self.timer` does not measure elapsed time in `on_exception` and `on_stop`. Measure real time there, for example to slow a robot down over a second

## Logging

A node's exception is logged with its traceback to the logger `"mujou"` at the ERROR level. In a program that has not configured logging, Python's default behavior prints it to standard error.

To send it elsewhere, configure Python's `logging`. The example above (`examples/errors.py`) sends it to standard output with the following lines.

```python
--8<-- "examples/errors.py:logging"
```

## The state after an exception

The state of a run that ended with an exception is not saved. `run.state_dict()` raises `NetworkError` as well. To keep the state, take it regularly while the network runs (see [Saving and restoring state](state.md)).

To run again, create new instances, nodes included. A network can run only once.

## Timeouts

`shutdown`, `join`, `subscribe` and `wait_for` take an argument `timeout` in seconds of real time. The default is `math.inf` (wait forever).

| Call                                     | When the time runs out                                                                                                                                                                                                     |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.shutdown(timeout)`                  | `TimeoutError`. The request to stop is not withdrawn, and the nodes update no more. The ending goes on in the background; wait for it with `join`                                                                          |
| `run.join(timeout)`                      | `TimeoutError`. Nothing changes, and the run keeps running                                                                                                                                                                 |
| `run.subscribe(port, timeout)`           | `TimeoutError` from the iterator, at a step that begins after the time has run out or would wait past it. The time counts the whole iteration, the body of the loop included. The iterator ends, and the run keeps running |
| `run.wait_for(port, condition, timeout)` | `TimeoutError`. The run keeps running                                                                                                                                                                                      |

`run.state_dict(timeout)` on a running network waits until every node has paused, at most `timeout` seconds of real time (30 by default). This `timeout` must be finite and greater than 0. After the run has ended there is nothing to wait for, and `timeout` is only checked.

If some node does not pause in time, `run.state_dict()` raises `TimeoutError`. The message names the nodes that did not pause. No state is taken, the nodes that had paused resume too, and the run keeps running.

!!! note "Long updates delay the pause"

    A `state_dict` timeout is likely with a node whose single `on_update` is long (heavy BP, for example), because a node finishes the running `on_update` before it pauses.

Passing a negative number or NaN as `timeout` raises `ValueError`; for `state_dict`, so do 0 and `math.inf`.

## Interpreter exit

If the interpreter exits while a run is running, Mujou stops the run and waits for `on_stop` on every node.

!!! tip "End each run yourself"

    Still, end each run yourself with `shutdown()` or a `with` statement.
