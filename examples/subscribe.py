"""Receive the outputs of a run as they are sent, and stop it from another thread.

``Run.subscribe`` yields the latest message now, then each newer one as it comes
(skipping those sent in between). The iteration ends when the run shuts down.
"""

import itertools
import threading
from typing import override

import torch

import mujou
from mujou import NextPort, Node, TensorSpec

SCALAR = TensorSpec((), torch.float64)


class Clock(Node):
    """Sends its run's time, in seconds."""

    def __init__(self) -> None:
        self.next = NextPort(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        now = self.timer.now()  # a node reads the time, but never waits on it
        self.next.send(torch.tensor(float(now), dtype=torch.float64))


clock = Clock()  # a node runs by itself, without a network
with mujou.run(clock) as run:

    def stop_later() -> None:
        mujou.time.sleep(0.1)
        run.shutdown()  # ends the iteration below

    stopper = threading.Thread(target=stop_later)
    stopper.start()
    received = [(value.item(), float(t)) for value, t in run.subscribe(clock.next)]
    stopper.join()

print(f"received {len(received)} messages, the last at t = {received[-1][1]:.3f} s")
# Each message is newer than the one before, and was sent after the clock read.
assert all(a[1] < b[1] for a, b in itertools.pairwise(received))
assert all(value <= t for value, t in received)
