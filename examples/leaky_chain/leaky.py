"""A leaky integrator: a node whose dynamics is defined in continuous time."""

import math
from typing import override

import torch

from mujou import NextPort, Node, PrevPort, TensorSpec

SCALAR = TensorSpec((), torch.float64)


class Leaky(Node):
    """Follows its input ``u`` with the time constant ``tau`` (in seconds):

        dx/dt = (u - x) / tau

    ``on_update`` advances ``x`` by ``dt``, the time that passed since the last
    update, with the exact solution for an input held constant over the step:
    ``x`` moves the fraction ``1 - exp(-dt / tau)`` of the way to ``u``. That
    fraction stays below 1 for any ``dt``, so a long ``dt`` (a busy machine) never
    overshoots ``u``, and none of the elapsed time is lost.
    """

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.prev = PrevPort(recv=SCALAR)
        self.next = NextPort(send=SCALAR)

    @override
    def on_start(self) -> None:
        super().on_start()
        self.x = SCALAR.zeros()  # the state: collected by the node automatically

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        u = self.prev.recv()
        self.x += (1 - math.exp(-dt / self.tau)) * (u - self.x)  # in place
        self.next.send(self.x)
