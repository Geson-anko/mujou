# Testing

In a network run the usual way, `dt` follows the real time that passes, and the order in which the nodes update differs every time. So the results differ slightly from run to run. This page shows how to use `mujou.testing` to run nodes and networks with the same result every time, and test them. Mujou's own tests use the same API.

## Tools

Import `mujou.testing` with `import mujou.testing`. `import mujou` alone does not import it.

| Name                                     | Role                                                                                                                    |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `mujou.testing.VirtualTimer()`           | A virtual timer. It starts at time 0 and moves only by what you pass to `advance(seconds)`                              |
| `mujou.testing.run(net, timer=..., ...)` | Runs the network with the same arguments as `mujou.run` and returns a `ManualRun`. No node updates until `step` lets it |
| `ManualRun.step(*nodes)`                 | Updates the given nodes once each, in the given order, and returns when all are done                                    |

`ManualRun` is a subclass of `Run`, so `write`, `read`, `state_dict` and the rest work as they do with `mujou.run`.

## Steps

The following test compares a network of two leaky integrators in a row with a discrete algorithm written with the same exact coefficient as the nodes, `1 - exp(-h / tau)`.

```python
--8<-- "examples/testing_a_node.py"
```

1. Write the reference with the same operations as the nodes. Here it updates `a`, then updates `b` with the new value of `a`
2. Create a `VirtualTimer()` and pass it as `mujou.testing.run(net, timer=timer, inputs=...)`
3. At each step, first move the time `h` seconds forward with `timer.advance(h)`, then update the nodes with `run.step(...)`. The `dt` of each node is then exactly `h`
4. Read the output with `run.read` and compare it with the reference

If the reference and the nodes do the same operations in the same order, the results match to the last bit, so you can compare them with `torch.equal`. If the order of operations differs, compare with `torch.testing.assert_close`.

## Rules of step

- `step()` with no arguments updates every node once, in the order of the network's tree
- The order changes the result. With `step(net.a, net.b)` in the example above, `b` reads the value `a` sent in this step. With `step(net.b, net.a)`, `b` reads the value of `a` from the previous step
- A node whose timer has not moved since its last update is not updated. If you forget `advance`, nothing happens, and if you pass the same node twice in one `step`, the second one is skipped
- The period of a node (`period`) is not used in a manual run
- `step` does not move the time. Only `advance` moves it

## Do not forget the VirtualTimer

The default `timer` of `mujou.testing.run` is the global timer, as with `mujou.run`. Without a `VirtualTimer`, the clock moves by itself and each `dt` is the real time that passed between steps. Such a test is not deterministic. To fix it, pass a `VirtualTimer()`, and at each step call `advance` before `step` (steps 2 and 3 of "Steps" above).

## Other conditions for determinism

- Give a `seed` to nodes that use random numbers (see [PyTorch in nodes](torch.md)). Without a `seed`, the generator is made with a different seed every time
- In tests too, the host does not wait with `time.sleep` to decide a result. Move the time with `advance`

## Testing save and restore

You can check that a run that takes the state midway and continues in a new instance matches a run that went on without stopping. On a `VirtualTimer`, the two match to the last bit. The following example uses the `Chain` class of the example above. The second half restores onto the same `timer` as the first half; this is allowed because the first half has already ended.

```python
import torch

import mujou.testing
from mujou.testing import VirtualTimer

u = torch.tensor([1.0, -2.0], dtype=torch.float64)


def steps(net, timer, n, state=None):
    with mujou.testing.run(net, timer=timer, inputs={net.prev: u}, state=state) as run:
        for _ in range(n):
            timer.advance(0.001)
            run.step()
    return run.state_dict()


straight = steps(Chain(0.01), VirtualTimer(), 20)
timer = VirtualTimer()
half = steps(Chain(0.01), timer, 10)
resumed = steps(Chain(0.01), timer, 10, state=half)
assert torch.equal(resumed["nodes"]["b"]["x"], straight["nodes"]["b"]["x"])
```

## Testing exceptions

When a node raises, the exception reaches you from `step` as a `NetworkError`. Its `__cause__` is the node's exception (see [Errors and shutdown](errors.md)).

```python
import pytest

from mujou import NetworkError

with pytest.raises(NetworkError) as info:
    run.step()
assert isinstance(info.value.__cause__, ValueError)
```
