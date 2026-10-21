"""Receive the outputs of a run as they are sent.

``Run.subscribe`` yields the latest message now, then each newer one as it comes
(skipping those sent in between). Leaving the loop with ``break`` stops the
iteration; the run keeps running until it shuts down.
"""

import itertools
from typing import override

import torch

import mujou
from mujou import NextPort, Node, TensorSpec

SCALAR = TensorSpec((), torch.float64)


class Clock(Node):
    """Sends its run's time, in seconds."""

    def __init__(self) -> None:
        super().__init__()
        self.next = NextPort(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        now = self.timer.now()  # a node reads the time, but never waits on it
        self.next.send(torch.tensor(float(now), dtype=torch.float64))


# --8<-- [start:subscribe]
clock = Clock()  # a node runs by itself, without a network
with mujou.run(clock) as run:
    received: list[tuple[float, float]] = []
    # timeout: give up after 5 s of real time instead of waiting forever.
    for value, t in run.subscribe(clock.next, timeout=5.0):
        if t is None:  # the zeros the port holds until the clock first sends
            continue
        received.append((value.item(), float(t)))
        if len(received) == 20:
            break
# --8<-- [end:subscribe]

print(f"received {len(received)} messages, the last at t = {received[-1][1]:.3f} s")
# Each message is newer than the one before, and was sent after the clock read.
assert all(a[1] < b[1] for a, b in itertools.pairwise(received))
assert all(value <= t for value, t in received)
