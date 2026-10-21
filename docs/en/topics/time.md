# ⏱️ Time

Each run has one time axis, its timer: every node's `dt`, every message's time and every pause of the run is on it. This page covers the timers, their units, the time scale, pausing, and the timer of a resumed run.

## The timers

| Name                        | What it is                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `mujou.time.global_timer`   | The timer of every run not given one of its own. It starts at 0 when the process first uses it                |
| `mujou.time.Timer()`        | A timer for a run of its own: `mujou.run(net, timer=Timer())`. It starts at `start` (0) with `time_scale` (1) |
| `mujou.time.VirtualTimer()` | A `Timer` that moves only when advanced (`advance(seconds)`); `mujou.testing.run` uses one                    |
| `mujou.time.ReadOnlyTimer`  | The reading side of a timer: what a node sees as `self.timer`, with only `now()` and `time_scale()`           |
| `mujou.time.Timestamp`      | A time on a timer                                                                                             |

The host controls the time of a run through `run.timer`: `now()`, `sleep(seconds)`, `pause()`, `resume()`, `is_paused()`, `time_scale()` and `set_time_scale(scale)`. A node only reads it, with `self.timer.now()`, in its hooks.

## Units

| Unit                       | Used by                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Seconds of the run's timer | `dt`, `period`, `Timer.sleep`, `VirtualTimer.advance`, `pc.show(seconds=)`, the time constants of `mujou.pc` |
| Real seconds               | Every `timeout`                                                                                              |

A timeout is real time so that a stuck run is given up on whatever its time scale and pauses.

## Timestamp

```python
--8<-- "examples/timers.py:timestamp"
```

`write`, `read_with_time`, `subscribe`, `wait_for`, `recv_with_time` and `now()` return `Timestamp`s (or `None` for a value not sent yet). Compare them with `<` and the like, and subtract them for the seconds between them. A `Timestamp` does not record which timer it is on: compare only times of one timer, such as those of one run and its `run.timer.now()`.

```python
t_in = run.write({net.prev: x})
run.timer.sleep(0.1)
y, t_out = run.read_with_time(net.next)
if t_out is not None and t_out >= t_in:  # y was sent after x was written
    print(t_out - t_in)  # seconds from the write until y was sent
```

## A timer of a run's own

```python
--8<-- "examples/timers.py:own-timer"
```

Give a run a timer of its own to speed up or slow down only that run (for debugging, or for trials faster than real time), or to keep it from pausing with other runs. The time scale is how many seconds of the timer pass per real second: at 10, a real second is 10 seconds of the timer. It is a finite number greater than 0.

The global timer can be scaled too, which changes every run on it:

```python
--8<-- "examples/timers.py:global-scale"
```

## Pausing the time

`timer.pause()` stops the time of the timer until `timer.resume()`; `global_timer.pause()` stops every run on the global timer. While it is paused:

- the nodes of every run on the timer do not update, and the first `dt` after `resume` does not count the pause
- `sleep` on the timer does not return, and `VirtualTimer.advance` raises `RuntimeError`
- `run.state_dict()` still works, and gives the state at the time of the pause

Taking the state pauses the run's timer too, apart from your `pause`: `pause` is a switch, not a count, and `resume` does not cut a pause for the state short. `is_paused()` is true while either one holds the time.

Use it to take the states of several runs on one timer at one time. Taken one after the other, the second state would be of a later time:

```python
--8<-- "examples/timers.py:pause"
```

!!! tip "Put `resume` in `finally`"

    Then an exception does not leave the time stopped.

A pause is for taking the state and for observing. Do not make pausing part of how a network learns.

## The timer of a resumed run

A run resumed with `mujou.run(net, state=tree)` runs on a new `Timer` that starts at the time and the time scale of the tree, so the global timer and the other runs do not go back:

```python
--8<-- "examples/timers.py:restore"
```

Passing `timer=` with `state=` raises `ValueError`, and `global_timer.pause()` does not pause a resumed run: pause its `run.timer`. Runs resumed from trees taken at one time therefore no longer share a time axis: their timers drift apart, and pausing one does not pause the others. Runs that must stay on one axis after a resume run as one run (`mujou.run(Group(a=a, b=b))`), whose single tree resumes onto one timer.

## Nodes only read the time

A node's `self.timer` is a `ReadOnlyTimer` of its run, readable in every hook (`on_start` included) and nowhere else: reading it in `__init__` or from the host raises `RuntimeError`. It stops at the time the run ends, so `on_exception` and `on_stop` see the time stand still.

Only the host moves the time ([rules a node keeps](../concepts.md#rules-a-node-keeps)). To update at intervals, give the node a `period`.
