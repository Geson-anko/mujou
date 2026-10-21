# 🧪 Testing

In a network run the usual way, `dt` follows the real time that passes, and the order in which the nodes update differs every time. So the results differ slightly from run to run.

This page shows how to use `mujou.testing` to run nodes and networks with the same result every time, and test them. Mujou's own tests use the same API.

## Tools

Import `mujou.testing` with `import mujou.testing`. `import mujou` alone does not import it.

| Name                          | Role                                                                                                                                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mujou.testing.run(net, ...)` | Runs the network with the same arguments as `mujou.run` and returns a `ManualRun`. No node updates until `step` lets it. The run is on a new `mujou.time.VirtualTimer` unless you pass one |
| `ManualRun.timer`             | The timer of the run: the `VirtualTimer` that `mujou.testing.run` made, or the one you passed                                                                                              |
| `ManualRun.step(dt, *nodes)`  | Moves the timer `dt` seconds forward, then updates the given nodes once each, in the given order, and returns when all are done                                                            |

`ManualRun` is a subclass of `Run`, so `write`, `read`, `state_dict` and the rest work as they do with `mujou.run`.

## Steps

The following test compares a network of two leaky integrators in a row with a discrete algorithm written with the same exact coefficient as the nodes, `1 - exp(-h / tau)`.

```python
--8<-- "examples/testing_a_node.py:test"
```

1. Write the reference with the same operations as the nodes. Here it updates `a`, then updates `b` with the new value of `a`
2. Run the network with `mujou.testing.run(net, inputs=...)`. It runs on a new `VirtualTimer` at time 0, which `run.timer` returns
3. At each step, `run.step(h, ...)` moves the time `h` seconds forward and updates the nodes. The `dt` of each node is then exactly `h`
4. Read the output with `run.read` and compare it with the reference

!!! tip "Comparing with the reference"

    If the reference and the nodes do the same operations in the same order, the results match to the last bit, so you can compare them with `torch.equal`. If the order of operations differs, compare with `torch.testing.assert_close`.

## Rules of step

- `step(dt)` with no nodes updates every node once, in the order of the network's tree
- The order changes the result. With `step(h, net.a, net.b)` in the example above, `b` reads the value `a` sent in this step. With `step(h, net.b, net.a)`, `b` reads the value of `a` from the previous step
- The `dt` of a node is the time since its last update; the first one is from the start of the run. If you pass the same node twice in one `step`, the second one is skipped (its time has not moved)
- The period of a node (`period`) is not used in a manual run
- A `VirtualTimer` shared by several runs moves for all of them

!!! warning "Do not wait without a timeout on the stepping thread"

    Nothing is sent unless `step` is called. So `wait_for` and `subscribe` without a `timeout` on the thread that calls `step` wait forever.

## Choosing the timer

By default, `mujou.testing.run` makes a new `VirtualTimer` for each run, unlike `mujou.run`, whose default is the global timer. To run several runs on one time axis, create a `mujou.time.VirtualTimer()` and pass the same one to each run. A run restored from a state makes its own, at the time of the state.

## Other conditions for determinism

- Give a `seed` to nodes that use random numbers (see [PyTorch in nodes](torch.md)). Without a `seed`, the generator is made with a different seed every time
- In tests too, the host does not wait with `time.sleep` to decide a result. Move the time with `step`

## Testing save and restore

You can check that a run that takes the state midway and continues in a new instance matches a run that went on without stopping. On a `VirtualTimer`, the two match to the last bit.

The following test continues the program above. The second half runs on a `VirtualTimer` of its own, which starts at the time of the state.

```python
--8<-- "examples/testing_a_node.py:restore"
```

## Testing exceptions

When a node raises, the exception reaches you from `step` as a `NetworkError`. Its `__cause__` is the node's exception (see [Errors and shutdown](errors.md)).

After that, every call of the run raises it again, including the `shutdown` at the end of the `with` block, so put `pytest.raises` around the whole block, in the same `with` statement:

```python
import pytest

import mujou.testing
from mujou import NetworkError

with pytest.raises(NetworkError) as info, mujou.testing.run(net) as run:
    run.step(0.001)  # a node raises ValueError here
assert isinstance(info.value.__cause__, ValueError)
```
