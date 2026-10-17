"""A leaky integrator: a node whose dynamics is defined in continuous time."""

from typing import override

import torch

from mujou import NextPort, Node, PrevPort, TensorSpec

SCALAR = TensorSpec((), torch.float64)


class Leaky(Node):
    """Follows its input ``u`` with the time constant ``tau`` (in seconds):

        dx/dt = (u - x) / tau

    ``on_update`` takes one explicit Euler step of length ``dt``, the time that
    passed since the last update. The rate ``dt / tau`` is clipped at 1, so that a
    long ``dt`` (a busy machine) jumps to ``u`` instead of overshooting it.
    """

    def __init__(self, tau: float) -> None:
        self.tau = tau
        self.prev = PrevPort(recv=SCALAR)
        self.next = NextPort(send=SCALAR)

    @override
    def on_start(self) -> None:
        self.x = SCALAR.zeros()  # the state: collected by the node automatically

    @override
    def on_update(self, dt: float) -> None:
        u = self.prev.recv()
        self.x += min(dt / self.tau, 1.0) * (u - self.x)  # in place
        self.next.send(self.x)
