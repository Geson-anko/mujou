# 🧩 2. Write a Node

This page writes a node whose state follows a differential equation, the leaky integrator, and runs two of them in a row. It covers `dt`, the state and the period. The rules every node keeps are in [How Mujou Works](../concepts.md#rules-a-node-keeps).

## The node

The node follows its input $u$ with the time constant $\tau$: its state $x$ follows $\frac{dx}{dt} = \frac{u - x}{\tau}$.

```python
--8<-- "examples/leaky.py:node"
```

The node writes three things.

1. In `__init__`, the settings (`tau`), the state (`x`) and the ports (`prev`, `next`)
2. In `on_update(dt)`, it reads the input, moves the state by `dt`, and sends the result
3. Nothing else: Mujou calls `on_update` again and again, so the node has no loop

## Update in proportion to `dt`

`dt` is the time in seconds since this node's previous update, on the run's timer. The first `dt` counts from the start of the run, once every node's `on_start` has returned.

Make every update proportional to `dt`. For $\frac{dx}{dt} = f(x)$, the simplest update is explicit Euler, `x += dt * f(x)`. A node updates at its own rate, and that rate varies with the load. With updates proportional to `dt`, every node follows the same dynamics on the same flow of time; a busy machine only makes `dt` longer. With a fixed step per update instead, a node that loops faster would change faster.

`dt` is not constant, and it gets longer when the machine is busy. Do not cut it: the time you cut off is lost, and the result drifts from the continuous-time solution. Stay stable for the longest `dt` that can come instead, in one of two ways.

- **Exact solution**: if the system has one, advance the state by `dt` with it. `Leaky` moves $x$ by the fraction `1 - exp(-dt / tau)` of its distance to $u$, which is exact while $u$ stays constant. The fraction is below 1 for any `dt`, so $x$ never overshoots $u$
- **Split the step**: otherwise, split `dt` into `n` steps no longer than a stable length inside one `on_update`, and update `n` times in a `for` loop ([4. Learn in a Node](4-learn-in-a-node.md) does this)

Splitting does not help when the instability comes from the coupling with other nodes, whose messages stay as they were over `dt`. Such a node may cap the `dt` it uses, at the cost of the time it drops (the nodes of `mujou.pc` do, with `max_dt`).

For noise in a continuous-time update, scale its standard deviation by $\sqrt{dt}$, not by `dt`, so that the result does not depend on how often the node updates.

## The state

`self.x` is a tensor attribute, so it is in the node's state by itself: `run.state_dict()` takes it, and a restore loads it back (see [6. Save and Resume](6-save-and-resume.md)). Create the state in `__init__`, and update it in place in `on_update` (`+=`, `copy_` and the like).

Assigning a new tensor to the attribute raises no error, and the new tensor is the state from then on. What breaks is any other reference to the first tensor, from another attribute or an optimizer: it still points at the old one. A restore also writes in place into the tensor the node holds.

`self.tau` is a number set from an argument: a setting, not state. A node restored with another `tau` keeps its own. A plain value that changes while the node runs, such as a counter, joins the state with `register_state` ([4. Learn in a Node](4-learn-in-a-node.md) does this).

The node sends `self.x`, and changes it in place afterwards. That is fine: `send` copies the value.

## The period

A node updates once every `period` seconds on average, 0.01 s by default. The time `on_update` takes counts toward the period, and a node that falls behind does not update several times in a row to catch up. `Leaky` takes a period of 1 ms by default, well below `tau`: a node downstream sees the output at most about one period late, a small delay next to `tau`.

Even with a period, `dt` is not exactly `period`: it is the time that actually passed, longer after a delay and shorter after that. Keep updating in proportion to `dt`.

In a chain fed by a node with a period (a sensor, say), the nodes after it read the same input until the next one arrives. Give them a period no longer than the period of their input.

!!! warning "`period=0.0` uses up a CPU core"

    `period=0.0` calls the next `on_update` as soon as any time has passed. Such a node keeps a CPU core busy, takes time from the other nodes and makes their `dt` irregular. Pass it only to a node that really must not wait.

## Two in a row

```python
--8<-- "examples/leaky.py:chain"
```

`leaky.py` steps the input of the chain from 0 to 1, and compares the outputs with the exact solution at the time each was sent:

```python
--8<-- "examples/leaky.py:main"
```

```sh
uv run python examples/leaky.py
```

`run.write` returns the time of the step, and `run.read_with_time` the time of each output: their difference is the time since the step, on the run's timer, however the nodes got scheduled.

## Next

- Test this node deterministically: [3. Test a Node](3-test.md)
- More about ports (specs, initial values, one message for values that belong together): [Ports and Channels](../topics/ports.md)
- The seed, the device and the threads of a node: [Threads and Devices](../topics/threads-and-devices.md)
