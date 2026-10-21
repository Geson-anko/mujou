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
from mujou import Node, Port, TensorSpec

VECTOR = TensorSpec((2,), torch.float64)


class Leaky(Node):
    """dx/dt = (u - x) / tau, exact for an input held constant over each ``dt``."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.prev = Port(recv=VECTOR)
        self.next = Port(send=VECTOR)

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
    # On a new VirtualTimer at time 0 (run.timer), which moves only by step.
    with mujou.testing.run(net, inputs={net.prev: u}) as run:
        for _ in range(steps):
            # Every node's next dt is exactly h: a, then b, which reads a's new value.
            run.step(h, net.a, net.b)
        y, t = run.read_with_time(net.next)

    assert torch.equal(y, b)  # the same operations: equal to the last bit
    assert t is not None and t.ns == steps * 1_000_000  # sent at the last step
    print(f"after {steps} steps of {h} s: y = {y}, equal to the reference")


# --8<-- [end:test]


# --8<-- [start:restore]
def run_steps(
    net: Chain, n: int, state: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run ``net`` for ``n`` steps of 1 ms and return its state."""
    u = torch.tensor([1.0, -2.0], dtype=torch.float64)
    with mujou.testing.run(net, inputs={net.prev: u}, state=state) as run:
        for _ in range(n):
            run.step(0.001)
    return run.state_dict()


def test_restored_run_matches_a_straight_run() -> None:
    straight = run_steps(Chain(0.01), 20)
    # Ten steps, then ten more in a new instance restored from the first half, on
    # a VirtualTimer of its own that starts at the time of the state.
    half = run_steps(Chain(0.01), 10)
    resumed = run_steps(Chain(0.01), 10, state=half)

    assert torch.equal(resumed["nodes"]["b"]["x"], straight["nodes"]["b"]["x"])
    print("save and restore: equal to the run that did not stop")


# --8<-- [end:restore]


if __name__ == "__main__":
    test_chain_follows_the_discrete_algorithm()
    test_restored_run_matches_a_straight_run()
