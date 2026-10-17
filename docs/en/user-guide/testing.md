# Testing

In a network run the usual way, `dt` follows the real time that passes, and the order in which the nodes update differs every time. So the results differ slightly from run to run. This page shows how to use `mujou.testing` to run nodes and networks with the same result every time, and test them. Mujou's own tests use the same API.

## Tools

Import `mujou.testing` with `import mujou.testing`. `import mujou` alone does not import it.

| Name                           | Role                                                                                                                                                                                  |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mujou.testing.run(net, ...)`  | Runs the network with the same arguments as `mujou.run` and returns a `ManualRun`. No node updates until `step` lets it. The run is on a new `VirtualTimer` unless you pass a `timer` |
| `mujou.testing.VirtualTimer()` | A virtual timer. It starts at time 0 and moves only by what you pass to `advance(seconds)`                                                                                            |
| `ManualRun.timer`              | The timer of the run: the `VirtualTimer` that `mujou.testing.run` made, or the `timer` you passed                                                                                     |
| `ManualRun.step(*nodes)`       | Updates the given nodes once each, in the given order, and returns when all are done                                                                                                  |

`ManualRun` is a subclass of `Run`, so `write`, `read`, `state_dict` and the rest work as they do with `mujou.run`.

## Steps

The following test compares a network of two leaky integrators in a row with a discrete algorithm written with the same exact coefficient as the nodes, `1 - exp(-h / tau)`.

```python
--8<-- "examples/testing_a_node.py:test"
```

1. Write the reference with the same operations as the nodes. Here it updates `a`, then updates `b` with the new value of `a`
2. Run the network with `mujou.testing.run(net, inputs=...)`. It runs on a new `VirtualTimer` at time 0, which `run.timer` returns
3. At each step, first move the time `h` seconds forward with `run.timer.advance(h)`, then update the nodes with `run.step(...)`. The `dt` of each node is then exactly `h`
4. Read the output with `run.read` and compare it with the reference

If the reference and the nodes do the same operations in the same order, the results match to the last bit, so you can compare them with `torch.equal`. If the order of operations differs, compare with `torch.testing.assert_close`.

## Rules of step

- `step()` with no arguments updates every node once, in the order of the network's tree
- The order changes the result. With `step(net.a, net.b)` in the example above, `b` reads the value `a` sent in this step. With `step(net.b, net.a)`, `b` reads the value of `a` from the previous step
- A node whose timer has not moved since its last update is skipped (it would see `dt = 0`). If you forget `advance`, nothing happens, and if you pass the same node twice in one `step`, the second one is skipped
- The period of a node (`period`) is not used in a manual run
- `step` does not move the time. Only `advance` moves it

## Choosing the timer

By default, `mujou.testing.run` makes a new `VirtualTimer` for each run, unlike `mujou.run`, whose default is the global timer. Pass a `timer` in these cases.

- To run several runs on one time axis, for example to restore a state on the timer of the run that took it (see "Testing save and restore" below), create a `VirtualTimer()` and pass the same one to each run
- To pass a `Timer` that is not virtual. Its clock then moves by itself, and each `dt` is the real time that passed between steps. Such a test is not deterministic

## Other conditions for determinism

- Give a `seed` to nodes that use random numbers (see [PyTorch in nodes](torch.md)). Without a `seed`, the generator is made with a different seed every time
- In tests too, the host does not wait with `time.sleep` to decide a result. Move the time with `advance`

## Testing save and restore

You can check that a run that takes the state midway and continues in a new instance matches a run that went on without stopping. On a `VirtualTimer`, the two match to the last bit. The following test continues the program above. The second half restores onto the same `timer` as the first half; this is allowed because the first half has already ended.

```python
--8<-- "examples/testing_a_node.py:restore"
```

## Testing exceptions

When a node raises, the exception reaches you from `step` as a `NetworkError`. Its `__cause__` is the node's exception (see [Errors and shutdown](errors.md)). After that, every call of the run raises it again, including the `shutdown` at the end of the `with` block, so put `pytest.raises` around the whole block, in the same `with` statement:

```python
import pytest

import mujou.testing
from mujou import NetworkError

with pytest.raises(NetworkError) as info, mujou.testing.run(net) as run:
    run.timer.advance(0.001)
    run.step()  # a node raises ValueError here
assert isinstance(info.value.__cause__, ValueError)
```
