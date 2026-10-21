"""Two leaky integrators in a row approach their input, in global time.

The input is a signal on the global time: the host holds each value for a while
with ``run.timer.sleep``. However often the nodes update, the outputs follow the
continuous-time solution, because every update advances the state by its ``dt``.
"""

import math

import torch
from leaky import SCALAR, Leaky

import mujou

TAU = 0.05  # seconds


class LeakyChain(mujou.Network):
    """``u -> a -> b ->``: two leaky integrators with the time constant ``TAU``."""

    def __init__(self) -> None:
        super().__init__()
        self.a = Leaky(TAU)
        self.b = Leaky(TAU)
        self.a.next.connect(self.b.prev)
        self.prev = self.a.prev
        self.next = self.b.next


def step_response(t: float) -> tuple[float, float]:
    """The exact outputs of a and b at ``t`` seconds after the input steps 0 -> 1
    (both 0 before the step, at ``t < 0``)."""
    s = max(t, 0.0) / TAU
    return 1 - math.exp(-s), 1 - math.exp(-s) * (1 + s)


net = LeakyChain()
with mujou.run(net, inputs={net.prev: SCALAR.zeros()}) as run:
    # Step the input from 0 to 1 (write returns the time of the step), then watch
    # both integrators rise.
    start = run.write({net.prev: torch.tensor(1.0, dtype=torch.float64)})
    for _ in range(4):
        run.timer.sleep(TAU / 2)
        a, t_a = run.read_with_time(net.a.next)
        b, t_b = run.read_with_time(net.b.next)
        # Each output against the exact one at the time it was sent. On a busy
        # machine a node may not have sent since the step: its time is then
        # earlier than start, and the difference is negative.
        exact_a, _ = step_response(t_a - start)
        _, exact_b = step_response(t_b - start)
        print(
            f"a = {a.item():.3f} (exact {exact_a:.3f}), "
            f"b = {b.item():.3f} (exact {exact_b:.3f})"
        )
        assert abs(a.item() - exact_a) < 0.05 and abs(b.item() - exact_b) < 0.05

    # Hold the input for 10 time constants: both reach the equilibrium x = u.
    run.timer.sleep(10 * TAU)
    b = run.read(net.next)
    print(f"after {10 * TAU:.1f} s more: b = {b.item():.4f}")
    assert abs(b.item() - 1.0) < 1e-3
