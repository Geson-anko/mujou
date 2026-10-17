"""Deterministic tests of nodes and networks.

Run the networks with ``run`` on a ``VirtualTimer``, advance the timer by a fixed
``dt`` and ``step`` the nodes by hand in a fixed order: every update then sees
exactly that ``dt``, and the result can be compared with the discrete algorithm.
Mujou tests its own nodes with the same API.

Example::

    timer = VirtualTimer()
    net = Chain()
    with mujou.testing.run(net, timer=timer, inputs={net.prev: x}) as run:
        for _ in range(10):
            timer.advance(0.001)
            run.step(net.a, net.b)  # a, then b reads a's new value
        y, t = run.read(net.next)
"""

from collections.abc import Mapping
from typing import Any

import torch

from mujou._core.network import Network
from mujou._core.node import Node
from mujou._core.port import Port
from mujou._core.run import ManualRun, start
from mujou._core.timer import Timer, VirtualTimer

__all__ = ["ManualRun", "VirtualTimer", "run"]


# P: as in Run.write.
def run[P: Port[Any, Any, Any]](
    component: Node | Network,
    *components: Node | Network,
    timer: Timer | None = None,
    inputs: Mapping[P, torch.Tensor | Mapping[str, object]] | None = None,
    state: Mapping[str, Any] | None = None,
    pause_timeout: float = 30.0,
) -> ManualRun:
    """Run the given nodes and networks as ``mujou.run`` does, with the updates
    left to ``ManualRun.step``.

    Pass a ``VirtualTimer`` as ``timer`` for a deterministic test: on any other
    timer the clock runs by itself, and each ``dt`` is whatever real time passed
    between the steps. The default is the global timer, as in ``mujou.run``.

    The arguments and the errors are those of ``mujou.run``: ``pause_timeout``
    too is in real seconds, finite and > 0 (``ValueError`` otherwise).
    """
    return start(
        ManualRun,
        (component, *components),
        timer=timer,
        inputs=inputs,
        state=state,
        pause_timeout=pause_timeout,
        manual=True,
    )
