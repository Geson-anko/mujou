# --8<-- [start:test]
"""Test a node deterministically with ``mujou.testing``.

``mujou.testing.run`` runs on a ``VirtualTimer``, whose time moves only when the
test advances it, and ``ManualRun.step`` updates the nodes by hand in a fixed order.
Every ``dt`` is then exactly the step, and the run can be compared with the discrete
algorithm, or with a run restored from a state taken midway.
"""

import math
from typing import Any, override

import torch

import mujou
import mujou.testing
from mujou import NextPort, Node, PrevPort, TensorSpec

VECTOR = TensorSpec((2,), torch.float64)


class Leaky(Node):
    """dx/dt = (u - x) / tau, exact for an input held constant over each ``dt``."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR)

    @override
    def on_start(self) -> None:
        super().on_start()
        self.x = VECTOR.zeros()

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.x += (1 - math.exp(-dt / self.tau)) * (self.prev.recv() - self.x)
        self.next.send(self.x)


class Chain(mujou.Network):
    def __init__(self, tau: float) -> None:
        super().__init__()
        self.a = Leaky(tau)
        self.b = Leaky(tau)
        self.a.next.connect(self.b.prev)
        self.prev = self.a.prev
        self.next = self.b.next


def test_chain_follows_the_discrete_algorithm() -> None:
    tau, h, steps = 0.01, 0.001, 50
    u = torch.tensor([1.0, -2.0], dtype=torch.float64)

    # The reference: the discrete algorithm, a then b at every step.
    a, b = VECTOR.zeros(), VECTOR.zeros()
    for _ in range(steps):
        a += (1 - math.exp(-h / tau)) * (u - a)
        b += (1 - math.exp(-h / tau)) * (a - b)  # b reads the new a

    net = Chain(tau)
    # On a new VirtualTimer at time 0 (run.timer), which moves only by advance.
    with mujou.testing.run(net, inputs={net.prev: u}) as run:
        for _ in range(steps):
            run.timer.advance(h)  # every node's next dt is exactly h
            run.step(net.a, net.b)  # a, then b, which reads a's new value
        y, t = run.read_with_time(net.next)

    assert torch.equal(y, b)  # the same operations: equal to the last bit
    assert t.ns == steps * 1_000_000  # sent at the time of the last step
    print(f"after {steps} steps of {h} s: y = {y}, equal to the reference")


# --8<-- [end:test]


# --8<-- [start:restore]
def run_steps(
    net: Chain,
    timer: mujou.testing.VirtualTimer,
    n: int,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run ``net`` for ``n`` steps of 1 ms on ``timer`` and return its state."""
    u = torch.tensor([1.0, -2.0], dtype=torch.float64)
    with mujou.testing.run(net, timer=timer, inputs={net.prev: u}, state=state) as run:
        for _ in range(n):
            timer.advance(0.001)
            run.step()
    return run.state_dict()


def test_restored_run_matches_a_straight_run() -> None:
    straight = run_steps(Chain(0.01), mujou.testing.VirtualTimer(), 20)
    # Ten steps, then ten more in a new instance restored from the first half. The
    # second half runs on the timer of the first, which has ended.
    timer = mujou.testing.VirtualTimer()
    half = run_steps(Chain(0.01), timer, 10)
    resumed = run_steps(Chain(0.01), timer, 10, state=half)

    assert torch.equal(resumed["nodes"]["b"]["x"], straight["nodes"]["b"]["x"])
    print("save and restore: equal to the run that did not stop")


# --8<-- [end:restore]


if __name__ == "__main__":
    test_chain_follows_the_discrete_algorithm()
    test_restored_run_matches_a_straight_run()
