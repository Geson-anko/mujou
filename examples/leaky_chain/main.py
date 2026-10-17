"""Two leaky integrators in a row approach their input, in global time.

The input is a signal on the global time: the host holds each value for a while
with ``mujou.time.sleep``. However often the nodes update, the outputs follow the
continuous-time solution, because every update is proportional to ``dt``.
"""

import math

import torch
from leaky import SCALAR, Leaky

import mujou

TAU = 0.05  # seconds


class LeakyChain(mujou.Network):
    """``u -> a -> b ->``: two leaky integrators with the time constant ``TAU``."""

    def __init__(self) -> None:
        self.a = Leaky(TAU)
        self.b = Leaky(TAU)
        self.a.connect(self.b)
        self.prev = self.a.prev
        self.next = self.b.next


def step_response(t: float) -> tuple[float, float]:
    """The exact outputs of a and b at ``t`` seconds after the input steps 0 -> 1."""
    s = t / TAU
    return 1 - math.exp(-s), 1 - math.exp(-s) * (1 + s)


net = LeakyChain()
with mujou.run(net, inputs={net.prev: SCALAR.zeros()}) as run:
    # Step the input from 0 to 1, then watch both integrators rise.
    run.write({net.prev: torch.tensor(1.0, dtype=torch.float64)})
    start = mujou.time.time()
    for _ in range(4):
        mujou.time.sleep(TAU / 2)
        (a, t_a), (b, t_b) = run.read(net.a.next), run.read(net.b.next)
        exact_a, _ = step_response(float(t_a) - start)
        _, exact_b = step_response(float(t_b) - start)
        print(
            f"a = {a.item():.3f} (exact {exact_a:.3f}), "
            f"b = {b.item():.3f} (exact {exact_b:.3f})"
        )
        assert abs(a.item() - exact_a) < 0.05 and abs(b.item() - exact_b) < 0.05

    # Hold the input for 10 time constants: both reach the equilibrium x = u.
    mujou.time.sleep(10 * TAU)
    b, _ = run.read(net.next)
    print(f"after {10 * TAU:.1f} s more: b = {b.item():.4f}")
    assert abs(b.item() - 1.0) < 1e-3
