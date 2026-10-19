# ⏱️ Controlling Time

This page shows how to read the time, wait on it, change how fast it runs, and give a run its own timer.

Why a node updates in proportion to `dt`, and why a pause does not make time jump, is in [Global time](../concepts.md#global-time).

## What to use where

| To                       | Host (default global timer)                  | Host (run on a `Timer` of its own) | In a node's hook              |
| ------------------------ | -------------------------------------------- | ---------------------------------- | ----------------------------- |
| Read the time            | `mujou.time.now()` (a `Timestamp`)           | `timer.now()`                      | `self.timer.now()`            |
| Wait on the time         | `mujou.time.sleep(seconds)`                  | `timer.sleep(seconds)`             | Not allowed                   |
| Read the time scale      | `mujou.time.get_time_scale()`                | `timer.get_time_scale()`           | `self.timer.get_time_scale()` |
| Change the time scale    | `mujou.time.set_time_scale(scale)`           | `timer.set_time_scale(scale)`      | Not allowed                   |
| Stop and restart time    | `mujou.time.pause()` / `mujou.time.resume()` | `timer.pause()` / `timer.resume()` | Not allowed                   |
| See whether time stopped | `mujou.time.is_paused()`                     | `timer.is_paused()`                | Not available                 |

!!! warning "Do not wait with time.sleep"

    The host waits with the `sleep` of the run's timer (to show an input for a certain time, for example), not with the standard `time.sleep` of Python. The standard `time.sleep` waits in real time: it ignores the time scale, and it goes on counting while the run pauses (a run pauses while its state is taken).

The scale is how many seconds of the timer pass per real second (at 5, one real second is 5 seconds of the timer). It is a finite number greater than 0. Anything else raises `ValueError`.

## Timestamp

The times that `run.write`, `run.read_with_time`, `run.subscribe` and `run.wait_for` return, and the values of `mujou.time.now()`, `timer.now()` and `self.timer.now()`, are all `mujou.time.Timestamp`.

| Form              | Value                                                                           |
| ----------------- | ------------------------------------------------------------------------------- |
| `ts.float()`      | Seconds (`float`). `float(ts)` is the same                                      |
| `ts.ns`           | Nanoseconds (`int`). The exact value                                            |
| `later - earlier` | The seconds from `earlier` to `later` (`float`). Negative if `later` is earlier |
| `a < b` and so on | Which time is earlier                                                           |

!!! warning "Compare only times from the same timer"

    A `Timestamp` does not record which timer it is from. Compare only times from the same timer. Compare with `mujou.time.now()` only the times of a run on the global timer.

Comparing the time that `write` returns with the time that `read_with_time` returns tells whether an output was sent after the input was switched.

```python
t_in = run.write({net.prev: x})
mujou.time.sleep(0.1)
y, t_out = run.read_with_time(net.next)
if t_out >= t_in:               # y was sent after x was written
    print(t_out - t_in)  # seconds from writing x until y was sent
```

## A timer for one run

Create `mujou.time.Timer()` and pass it as `mujou.run(..., timer=timer)`, and the run runs on that timer. A new timer starts at time 0 with scale 1.

```python
--8<-- "examples/timers.py:own-timer"
```

Give a run its own timer in these cases.

- You want to speed up or slow down only that run (for debugging, or for trials faster than real time)
- You want to restore a state while another run is running on the global timer (see [Saving and restoring state](state.md))
- You run several networks in separate runs that should not stop each other (see [Grouping networks](group.md))

The host then waits on that run with `timer.sleep`, and compares the times from `run.read_with_time` with `timer.now()`. `run.timer` returns the timer of a run, the global timer if you passed none.

!!! warning "Do not use the functions of mujou.time for this run"

    The functions of `mujou.time` (`mujou.time.sleep`, `mujou.time.now()`, ...) work on the global timer, not on this run's timer, so do not use them for this run.

## Changing the speed of the global timer

`mujou.time.set_time_scale` changes the speed of every run on the global timer.

```python
--8<-- "examples/timers.py:global-scale"
```

The global time starts at 0 when the process first uses it.

## Pausing the time

`timer.pause()` stops the time of the timer until `timer.resume()`; `mujou.time.pause()` and `mujou.time.resume()` do the same for the global timer.

Use it when the host does something that should happen at one time, such as taking the states of several runs on the same timer (see [Saving and restoring state](state.md#taking-the-states-of-several-runs-at-one-time)).

While the time is paused:

- The nodes of every run on the timer do not update. They wait until the time moves again, and the first `dt` after `resume` does not count the pause
- `sleep` on the timer does not return until the time moves again
- `VirtualTimer.advance` raises `RuntimeError`
- `run.state_dict()` still works, and gives the state at the time of the pause

!!! note "pause is a switch, not a count"

    `pause` is a switch, not a count: calling it twice still needs only one `resume`. Taking the state pauses the timer too, apart from this switch, so `resume` does not cut that short, and the end of taking the state does not end your pause. `is_paused()` is true while either one stops the time.

## Nodes only read the time

A node's `self.timer` is the read-only side of the run's timer (`mujou.time.ReadOnlyTimer`). It has only `now()` and `get_time_scale()`.

Calling `sleep`, `set_time_scale`, `pause` or `resume` of `mujou.time` or of a `Timer`, or `VirtualTimer.advance`, in a hook raises `RuntimeError`. If one node stopped the time or changed its speed, it would change the `dt` of the other nodes in the same run too.

- To update a node at regular intervals, do not wait in a hook. Pass a period with `super().__init__(period=seconds)`. The period is the average time between two updates, in seconds of the run's timer (see [Writing a node](writing-a-node.md))
- `self.timer` can be read in every hook, `on_start` included. Reading it in `__init__` raises `RuntimeError`

## Restoring sets the time back

Restoring with `mujou.run(..., state=tree)` sets the run's timer back to the time the tree was taken. On the global timer, `mujou.time.now()` goes back too. Details are in [Saving and restoring state](state.md).

## Use a virtual timer in tests

To give `dt` a fixed value in a test, use `mujou.testing.VirtualTimer`. The steps are in [Testing](testing.md).
