# 🧪 3. Test a Node

In a run started with `mujou.run`, `dt` follows the real time that passes, and the order in which the nodes update differs every time, so the results differ slightly from run to run. This page tests the node of [2. Write a Node](2-node.md) with `mujou.testing`, which runs nodes with the same result every time.

## The tools

Import it with `import mujou.testing`; `import mujou` alone does not.

| Name                           | Role                                                                                                                                                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mujou.testing.run(root, ...)` | Runs the tree with the `timer`, `inputs` and `state` of `mujou.run` (no `workers`) and returns a `ManualRun`. No node updates until `step` updates it. It runs on a new `mujou.time.VirtualTimer` by default |
| `ManualRun.step(dt, *nodes)`   | Moves the timer `dt` seconds forward, then updates the given nodes once each, in the given order                                                                                                             |
| `ManualRun.timer`              | The run's `VirtualTimer`                                                                                                                                                                                     |

`ManualRun` is a `Run`, so `write`, `read`, `state_dict` and the rest work as with `mujou.run`.

## A test

```python
--8<-- "examples/test_leaky.py:test"
```

```sh
uv run pytest examples/test_leaky.py
```

1. `mujou.testing.run(node, inputs=...)` runs the node on a new `VirtualTimer` at time 0. `inputs` are the values of the input ports from the start
2. Each `run.step(h)` moves the time `h` seconds forward and updates every node once, so the `dt` of the node is exactly `h`
3. The reference is the exact solution, computed with the same coefficient. Compare it with `torch.testing.assert_close`, or with `torch.equal` when the reference does the same operations in the same order (below)
4. The time the node sent the output is `steps` milliseconds: no real time enters a manual run

## The order of the updates

```python
--8<-- "examples/test_leaky.py:chain"
```

- `step(dt)` without nodes updates every node once, in the order of the tree (the order the attributes were assigned)
- The order changes the result. With `step(h, net.a, net.b)`, `b` reads the value `a` sent in this step; with `step(h, net.b, net.a)`, the one from the step before
- A node's `dt` is the time since its last update, the first from the start of the run. A node passed twice in one `step` updates once: the second time is skipped, since its time has not moved
- The period of a node is not used in a manual run
- A node that raises makes `step` raise `NetworkError` at once

## Save and restore

A run that takes the state midway and goes on in a new instance matches a run that went on without stopping, to the last bit:

```python
--8<-- "examples/test_leaky.py:restore"
```

A run restored from a state runs on a `VirtualTimer` of its own, which starts at the time of the state.

## Exceptions

When a node raises, `step` raises `NetworkError`, whose `__cause__` is the node's exception. Every later call of the run raises it again, including the `shutdown` at the end of the `with` block, so put `pytest.raises` around the whole block, in the same `with` statement:

```python
with pytest.raises(NetworkError) as info, mujou.testing.run(net) as run:
    run.step(0.001)  # a node raises ValueError here
assert isinstance(info.value.__cause__, ValueError)
```

## Other conditions for determinism

- Give a `seed` to the nodes that draw random numbers, and draw them from `self.generator` (see [Threads and Devices](../topics/threads-and-devices.md#seeds-and-random-numbers))
- Move the time only with `step`. Nothing is sent unless `step` is called, so `wait_for` and `subscribe` without a `timeout` wait forever on the thread that steps
- To put several runs on one time axis, pass the same `mujou.time.VirtualTimer()` to each: a `step` of one moves the others too

## Next

- A node that learns: [4. Learn in a Node](4-learn-in-a-node.md)
