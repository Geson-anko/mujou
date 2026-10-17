"""Test a node deterministically with ``mujou.testing``.

On a ``VirtualTimer`` the time moves only when the test advances it, and
``ManualRun.step`` updates the nodes by hand in a fixed order. Every ``dt`` is then
exactly the step, and the run can be compared with the discrete algorithm.
"""

from typing import override

import torch

import mujou
import mujou.testing
from mujou import NextPort, Node, PrevPort, TensorSpec
from mujou.testing import VirtualTimer

VECTOR = TensorSpec((2,), torch.float64)


class Leaky(Node):
    """dx/dt = (u - x) / tau, by explicit Euler."""

    def __init__(self, tau: float) -> None:
        self.tau = tau
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR)

    @override
    def on_start(self) -> None:
        self.x = VECTOR.zeros()

    @override
    def on_update(self, dt: float) -> None:
        self.x += min(dt / self.tau, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)


class Chain(mujou.Network):
    def __init__(self, tau: float) -> None:
        self.a = Leaky(tau)
        self.b = Leaky(tau)
        self.a.connect(self.b)
        self.prev = self.a.prev
        self.next = self.b.next


def test_chain_follows_explicit_euler() -> None:
    tau, h, steps = 0.01, 0.001, 50
    u = torch.tensor([1.0, -2.0], dtype=torch.float64)

    # The reference: the discrete algorithm, a then b at every step.
    a, b = VECTOR.zeros(), VECTOR.zeros()
    for _ in range(steps):
        a += h / tau * (u - a)
        b += h / tau * (a - b)  # b reads the new a

    timer = VirtualTimer()
    net = Chain(tau)
    with mujou.testing.run(net, timer=timer, inputs={net.prev: u}) as run:
        for _ in range(steps):
            timer.advance(h)  # every node's next dt is exactly h
            run.step(net.a, net.b)  # a, then b, which reads a's new value
        y, t = run.read(net.next)

    assert torch.equal(y, b)  # the same operations: equal to the last bit
    assert t.ns == steps * 1_000_000  # sent at the time of the last step
    print(f"after {steps} steps of {h} s: y = {y}, equal to the reference")


if __name__ == "__main__":
    test_chain_follows_explicit_euler()
