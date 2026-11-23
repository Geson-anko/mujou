"""Simulate a network on virtual time, as fast as the machine computes.

``mujou.run_manually`` runs the ``LeakyChain`` of ``leaky.py`` on a
``VirtualTimer``: ``ManualRun.step`` moves the time and updates the nodes, with no
sleeping in between. The program simulates 2 s of a sine input in steps of 1 ms
and prints how long that took in real time. Then it compares the two kinds of
step: every node at once, and the nodes one by one in a given order.
"""

import math
import time

import torch
from leaky import SCALAR, LeakyChain

import mujou

# Each worker runs torch operations with threads of torch's own: keep those few
# (a setting of the whole process, made by the host before the run).
torch.set_num_threads(1)

TAU = 0.05  # the time constant of the integrators, in seconds
H = 0.001  # one step: 1 ms of the run's timer
ONE = torch.tensor(1.0, dtype=SCALAR.dtype)


# --8<-- [start:simulate]
def simulate(seconds: float, workers: int | None = None) -> torch.Tensor:
    """Feed a sine of 1 Hz to a new chain for ``seconds`` of virtual time and
    return its last output."""
    net = LeakyChain(TAU)
    with mujou.run_manually(net, workers=workers) as run:
        for i in range(round(seconds / H)):
            # Write between steps: the input for the step about to run.
            u = math.sin(2 * math.pi * i * H)
            run.write({net.prev: torch.tensor(u, dtype=SCALAR.dtype)})
            run.step(H)  # 1 ms later, then every node once, in parallel
        return run.read(net.next)


start = time.perf_counter()
y = simulate(2.0)
print(f"2 s simulated in {time.perf_counter() - start:.2f} s of real time")
# The same result, bit for bit, with any number of workers.
assert torch.equal(y, simulate(2.0, workers=1))
# --8<-- [end:simulate]


# --8<-- [start:orders]
def output_after(steps: int, order: str) -> torch.Tensor:
    """Step a new chain ``steps`` times with a constant input of 1 and return its
    output. ``order`` is "at once", "a, b" or "b, a"."""
    net = LeakyChain(TAU)
    nodes = {"at once": (), "a, b": (net.a, net.b), "b, a": (net.b, net.a)}[order]
    with mujou.run_manually(net, inputs={net.prev: ONE}) as run:
        for _ in range(steps):
            run.step(H, *nodes)
        return run.read(net.next)


at_once = output_after(3, "at once")  # b reads the a of the start of each step
assert torch.equal(at_once, output_after(3, "b, a"))  # b, then a: the same
assert not torch.equal(at_once, output_after(3, "a, b"))  # b reads the new a
print(f"after 3 steps: b = {at_once.item():.6f} (at once and b, a)")
print(f"after 3 steps: b = {output_after(3, 'a, b').item():.6f} (a, b)")
# --8<-- [end:orders]
