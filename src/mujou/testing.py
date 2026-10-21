"""Deterministic tests of nodes and networks.

Run the networks with ``run``, which puts them on a new ``VirtualTimer``, and
``step`` the nodes by hand by a fixed ``dt`` in a fixed order: every update then
sees exactly that ``dt``, and the result can be compared with the discrete
algorithm. Mujou tests its own nodes with the same API.

Example::

    from typing import override

    import torch

    import mujou
    import mujou.testing
    from mujou import Node, Port, TensorSpec


    class Double(Node):
        def __init__(self) -> None:
            super().__init__()
            self.prev = Port(recv=TensorSpec((2,)))
            self.next = Port(send=TensorSpec((2,)))

        @override
        def on_update(self, dt: float) -> None:
            super().on_update(dt)
            self.next.send(2 * self.prev.recv())


    class Chain(mujou.Network):
        def __init__(self) -> None:
            super().__init__()
            self.a, self.b = Double(), Double()
            self.a.next.connect(self.b.prev)
            self.prev, self.next = self.a.prev, self.b.next


    net = Chain()
    x = torch.ones(2)
    with mujou.testing.run(net, inputs={net.prev: x}) as run:
        run.step(0.001, net.a, net.b)  # 1 ms later, a, then b reads a's new value
        y, t = run.read_with_time(net.next)
    assert torch.equal(y, 4 * x) and t is not None and t.ns == 1_000_000
"""

from collections.abc import Mapping
from typing import Any

import torch

from mujou._core.network import Network
from mujou._core.node import Node
from mujou._core.port import Port
from mujou._core.run import ManualRun, VirtualTimer, start

__all__ = ["ManualRun", "run"]


# P: as in Run.write.
def run[P: Port[Any, Any]](
    root: Node | Network,
    *,
    timer: VirtualTimer | None = None,
    inputs: Mapping[P, torch.Tensor | Mapping[str, object]] | None = None,
    state: Mapping[str, Any] | None = None,
) -> ManualRun:
    """Run ``root`` as ``mujou.run`` does, with the updates left to
    ``ManualRun.step``.

    The default ``timer`` is a new ``mujou.time.VirtualTimer`` at time 0 (with
    ``state``, at the time of the state), which ``ManualRun.timer`` returns and
    ``ManualRun.step`` advances. Pass a ``VirtualTimer`` of your own to share it
    between runs: a ``step`` of one then moves the time of the others too.

    The other arguments and the errors are those of ``mujou.run``.
    """
    return start(ManualRun, root, timer=timer, inputs=inputs, state=state, workers=None)
