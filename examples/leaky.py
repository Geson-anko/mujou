"""A node defined in continuous time: the leaky integrator.

``Leaky`` follows its input with a time constant, and ``LeakyChain`` puts two in
a row. Run this file to watch the chain follow a step of its input, against the
exact solution; ``test_leaky.py`` tests the same node.
"""

import math
from typing import override

import torch

import mujou
from mujou import Network, Node, Port, TensorSpec

SCALAR = TensorSpec((), torch.float64)


# --8<-- [start:node]
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

    def __init__(self, tau: float, *, period: float = 0.001) -> None:
        super().__init__(period=period)
        self.tau = tau  # a setting, not state
        self.x = SCALAR.zeros()  # a tensor attribute: in the state by itself
        self.prev = Port(recv=SCALAR)
        self.next = Port(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        u = self.prev.recv()
        self.x += (1 - math.exp(-dt / self.tau)) * (u - self.x)  # in place
        self.next.send(self.x)  # a copy: changing x later does not change it


# --8<-- [end:node]


# --8<-- [start:chain]
class LeakyChain(Network):
    """``u -> a -> b ->``: two leaky integrators with the time constant ``tau``."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.a = Leaky(tau)
        self.b = Leaky(tau)
        self.a.next.connect(self.b.prev)
        self.prev, self.next = self.a.prev, self.b.next


# --8<-- [end:chain]


def step_response(t: float, tau: float) -> tuple[float, float]:
    """The exact outputs of a and b at ``t`` seconds after the input steps 0 -> 1
    (both 0 before the step, at ``t < 0``)."""
    s = max(t, 0.0) / tau
    return 1 - math.exp(-s), 1 - math.exp(-s) * (1 + s)


# --8<-- [start:main]
if __name__ == "__main__":
    tau = 0.05
    net = LeakyChain(tau)
    with mujou.run(net) as run:
        # Step the input from 0 to 1 (write returns the time of the step), then
        # watch both integrators rise on the run's timer.
        start = run.write({net.prev: torch.tensor(1.0, dtype=torch.float64)})
        for _ in range(4):
            run.timer.sleep(tau / 2)
            a, t_a = run.read_with_time(net.a.next)
            b, t_b = run.read_with_time(net.b.next)
            # Each output against the exact one at the time it was sent. On a busy
            # machine a node may not have sent since the step: its time is then
            # earlier than start, and the difference is negative (None, before
            # its first send, counts as long before the step).
            exact_a, _ = step_response(-math.inf if t_a is None else t_a - start, tau)
            _, exact_b = step_response(-math.inf if t_b is None else t_b - start, tau)
            print(
                f"a = {a.item():.3f} (exact {exact_a:.3f}), "
                f"b = {b.item():.3f} (exact {exact_b:.3f})"
            )
        # Hold the input for 10 time constants: both reach the equilibrium x = u.
        run.timer.sleep(10 * tau)
        b = run.read(net.next)
    print(f"after {10 * tau:.1f} s more: b = {b.item():.4f}")
# --8<-- [end:main]
