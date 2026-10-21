"""A leaky integrator: a node whose dynamics is defined in continuous time."""

import math
from typing import override

import torch

from mujou import Node, Port, TensorSpec

SCALAR = TensorSpec((), torch.float64)


class Leaky(Node):
    """Follows its input ``u`` with the time constant ``tau`` (in seconds):

        dx/dt = (u - x) / tau

    ``on_update`` advances ``x`` by ``dt``, the time that passed since the last
    update, with the exact solution for an input held constant over the step:
    ``x`` moves the fraction ``1 - exp(-dt / tau)`` of the way to ``u``. That
    fraction stays below 1 for any ``dt``, so a long ``dt`` (a busy machine) never
    overshoots ``u``, and none of the elapsed time is lost.

    ``period`` is well below ``tau`` by default: a node downstream sees this
    output at most about one period late, a small delay next to ``tau``.
    """

    def __init__(self, tau: float, period: float = 0.001) -> None:
        super().__init__(period=period)
        self.tau = tau
        self.prev = Port(recv=SCALAR)
        self.next = Port(send=SCALAR)
        self.x = SCALAR.zeros()  # the state: collected by the node automatically

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        u = self.prev.recv()
        self.x += (1 - math.exp(-dt / self.tau)) * (u - self.x)  # in place
        self.next.send(self.x)
