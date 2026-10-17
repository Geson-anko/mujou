# Global Time

Mujou treats the learning of a network as a system that changes continuously over time. How much a node updates depends on the time that has passed, not on the number of iterations.

## Updates are proportional to the elapsed time

A node is called with `on_update(dt)` over and over. `dt` is the time (in seconds) since that node's last update. The node updates its state in proportion to `dt`. For example, if the state $x$ follows the differential equation $\frac{dx}{dt} = f(x)$, the explicit Euler method updates it as follows.

$$
x \leftarrow x + \Delta t \, f(x)
$$

In code, this is `x = x + dt * f(x)`.

The time a node takes for one update differs from node to node and varies with the load. If the update size were set per iteration, a node that loops faster would change its state more. With updates proportional to the elapsed time, nodes that update at different rates still approximate the same dynamics on the same flow of time. A higher load only makes `dt` longer.

However, the explicit Euler method becomes unstable when `dt` is too large. In that case, the node splits its step into smaller ones within one update. Do not cap `dt`: a capped `dt` drops part of the time that actually passed, and the node falls behind the dynamics in continuous time.

By default, `on_update` is called with as short an interval as possible. Give a node a period of $P$ seconds, and Mujou sets the next target to "the previous target + $P$" and waits until that time before calling it. The time an update takes and a late wake-up count toward the period, so the node updates once every $P$ seconds on average. The period is also counted in seconds of the run's timer. How to pass a period is in [Writing a node](../user-guide/writing-a-node.md).

## Timers

`dt` and the time are measured on the run's timer. "The run's timer" is the timer that one run, started by `mujou.run`, uses; by default it is the global timer. All nodes and ports of a run use the same timer.

| Argument of `mujou.run`    | Timer used                                                    |
| -------------------------- | ------------------------------------------------------------- |
| `timer=None` (default)     | The global timer, one per process                             |
| `timer=mujou.time.Timer()` | The timer you pass. It runs independently of the global timer |

A time is a `mujou.time.Timestamp`. `float(ts)` gives seconds and `ts.ns` nanoseconds. Compare only times from the same timer.

## The time scale

A timer's time can run faster or slower than real time. After `mujou.time.set_time_scale(2.0)`, the global timer advances 2 seconds per second of real time. For a `Timer` you created, use that timer's `set_time_scale`.

## The host waits on the run's timer

When the host wants to show an input for a certain time, it waits on the run's timer. Use `mujou.time.sleep(seconds)` for the global timer, and `timer.sleep(seconds)` for a timer you created. The standard `time.sleep` waits in real time, so it ignores changes of the time scale and the pauses described below.

## Nodes only read the time

A node reads the time of the run's timer (`self.timer.now()`) and its scale (`self.timer.get_time_scale()`) through the attribute `self.timer`. Only the host moves the time. Calling `sleep` or `set_time_scale` in a hook of a node raises `RuntimeError`. If one node changed the scale, it would change the `dt` of the other nodes in the same run too.

## A pause does not make time jump

When the state is taken, Mujou pauses the run for a moment (see [State](state.md)). While paused, the run's timer stops as well. So the paused time does not enter `dt` after the run resumes. To the nodes' updates, it is as if the pause never happened.

A pause reaches every run that uses the same timer. By default, every run uses the global timer, so taking the state of one run stops the time of the other runs too. To keep two runs from pausing each other, it is enough to pass `timer=mujou.time.Timer()` to one of them. The two runs then use different timers, and a pause of one does not stop the other.

A pause is for taking the state or observing. Do not make pausing part of how the network learns, for example by pausing all nodes regularly to average their weights.

## Time in tests

In tests, use `mujou.testing.VirtualTimer`. This virtual timer advances only by what you pass to `advance(seconds)`. Step a run by hand with `mujou.testing.run`, and every node updates with a fixed `dt` in a fixed order, so the results are the same every time.

Controlling time in detail is in [Controlling time](../user-guide/time.md) and [Testing](../user-guide/testing.md).
