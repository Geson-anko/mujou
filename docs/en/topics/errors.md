# 🚨 Errors and Shutdown

This page shows what happens when a hook of a node raises, where the host receives the exception, and the timeouts of the host's calls.

## One node's exception stops everything

Whichever hook of whichever node raises (`on_update`, `on_start`, `state_dict` or any other), the whole run stops:

1. Mujou logs the node's path, the hook and the traceback to the logger `"mujou"` at the ERROR level
2. Each node finishes the hook it is running, stops, and calls its `on_exception()`, without waiting for the others. The nodes that did not raise are called too
3. Once every `on_exception()` has returned, every node's `on_stop()` is called, the one that raised included
4. The host gets a `NetworkError`, a subclass of `RuntimeError`, whose `__cause__` is the first exception. Exceptions of other nodes at about the same time are only logged

Mujou never stops only the node that raised: its neighbors would keep reading its old values and drift away from the equilibrium without anyone noticing.

```python
--8<-- "examples/errors.py"
```

## Where the host receives it

| The exception happens                  | What raises `NetworkError`                                                                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| While the run starts (`on_start`, ...) | `mujou.run` itself, after the ending work (`on_exception`, `on_stop`)                                                                                  |
| After the start                        | The next `write`, `read`, `read_with_time`, `subscribe` (and its iterator), `wait_for`, `state_dict`, `shutdown` or `join`, or `step` of a `ManualRun` |

An exception after the start reaches the host at its next call, so a host that only waits does not notice it: wait with `run.join(timeout)`, not with `run.timer.sleep`. At the end of a `with` block, the `shutdown` raises it; if the body of the block raises too, the body's exception goes on and the `NetworkError` is dropped (the log still has the node's exception).

## `on_exception` and `on_stop`

| Hook             | Called                                                  | What to write                                                     |
| ---------------- | ------------------------------------------------------- | ----------------------------------------------------------------- |
| `on_exception()` | When the run ends by an exception, on every node, first | The work of an abnormal end, such as a safe stop of the actuators |
| `on_stop()`      | Once at every end, normal or not                        | The release of resources such as I/O; not a change of the state   |

- `on_exception` takes no arguments: a node does not learn which exception stopped the run. The host handles the cause, from `__cause__`
- To handle an exception within a node, catch it with `try` in the hook: an exception caught there does not stop the run
- If `on_exception` or `on_stop` raises, the exception is logged, and the ending goes on
- `self.timer` does not move in these hooks: see the [rules a node keeps](../concepts.md#rules-a-node-keeps)

## Logging

The node's exception goes to the logger `"mujou"` with its traceback; a program that has not configured logging prints it to standard error. Configure Python's `logging` to send it elsewhere:

```python
--8<-- "examples/errors.py:logging"
```

## The state after an exception

The state of a run that ended by an exception is not kept: `run.state_dict()` raises `NetworkError` too. Take the state you want to keep while the network runs, regularly during a long run. To run again, build new instances.

## Timeouts

`shutdown`, `join`, `subscribe`, `wait_for` and `state_dict` take a `timeout` in real seconds; the default `math.inf` waits as long as it takes.

| Call                                         | When the time runs out                                                                                                             |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `run.shutdown(timeout=...)`                  | `TimeoutError`. The request to stop stands, and the nodes update no more; the ending goes on, and `join` waits for it              |
| `run.join(timeout=...)`                      | `TimeoutError`. Nothing changes                                                                                                    |
| `run.subscribe(port, timeout=...)`           | `TimeoutError` from the iterator, counting the whole iteration, the body of the loop included. The run keeps running               |
| `run.wait_for(port, condition, timeout=...)` | `TimeoutError`. The run keeps running                                                                                              |
| `run.state_dict(timeout=...)`                | `TimeoutError`, while some node has not paused (each finishes its `on_update` first). No state is taken, and the run keeps running |

A node whose single `on_update` is long (heavy BP, say) delays every pause for the state.

## Interpreter exit

If the interpreter exits while a run is running, Mujou shuts the run down and waits for every `on_stop`, once the other non-daemon threads have ended. A thread that iterates over `subscribe` keeps the exit waiting until its run ends, so end each run yourself, with `shutdown` or a `with` block.
