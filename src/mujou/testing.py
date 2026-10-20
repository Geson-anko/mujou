"""Deterministic tests of nodes and networks.

Run the networks with ``run``, which puts them on a new ``VirtualTimer``, advance
the timer by a fixed ``dt`` and ``step`` the nodes by hand in a fixed order: every
update then sees exactly that ``dt``, and the result can be compared with the
discrete algorithm. Mujou tests its own nodes with the same API.

Example::

    from typing import override

    import torch

    import mujou
    import mujou.testing
    from mujou import NextPort, Node, PrevPort, TensorSpec


    class Double(Node):
        def __init__(self) -> None:
            super().__init__()
            self.prev = PrevPort(recv=TensorSpec((2,)))
            self.next = NextPort(send=TensorSpec((2,)))

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
        run.timer.advance(0.001)  # the run's VirtualTimer
        run.step(net.a, net.b)  # a, then b reads a's new value
        y, t = run.read_with_time(net.next)
    assert torch.equal(y, 4 * x) and t.ns == 1_000_000
"""

from collections.abc import Mapping
from typing import Any, cast

import torch

from mujou._core.network import Network
from mujou._core.node import Node
from mujou._core.port import Port
from mujou._core.run import ManualRun, start_run
from mujou._core.timer import Timer, VirtualTimer

__all__ = ["ManualRun", "VirtualTimer", "run"]


# P: as in Run.write. T: the timer's type, which ManualRun.timer gives back.
def run[P: Port[Any, Any, Any, Any, Any], T: Timer = VirtualTimer](
    component: Node | Network,
    *components: Node | Network,
    timer: T | None = None,
    inputs: Mapping[P, torch.Tensor | Mapping[str, object]] | None = None,
    state: Mapping[str, Any] | None = None,
) -> ManualRun[T]:
    """Run the given nodes and networks as ``mujou.run`` does, with the updates
    left to ``ManualRun.step``.

    The default ``timer`` is a new ``VirtualTimer`` at time 0, which
    ``ManualRun.timer`` returns: advance it before each ``step``, and every
    ``dt`` is exactly what the test advanced. Pass a ``VirtualTimer`` of your own
    to share it between runs (e.g. to restore a state on it). On any other timer
    the clock runs by itself, and each ``dt`` is whatever real time passed between
    the steps.

    The other arguments and the errors are those of ``mujou.run``.
    """
    # The class itself: a subscripted generic is not a class to instantiate.
    return start_run(
        cast(type[ManualRun[T]], ManualRun),
        (component, *components),
        timer=cast(T, VirtualTimer()) if timer is None else timer,
        inputs=inputs,
        state=state,
        manual=True,
        workers=None,
    )
