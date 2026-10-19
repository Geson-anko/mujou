"""The smallest program: two nodes in a network, run in the background.

The host writes an input, follows the output until it arrives and stops.
"""

from typing import override

import torch

import mujou
from mujou import NextPort, Node, PrevPort, TensorSpec

VECTOR = TensorSpec((3,))  # every message here is a float32 vector of 3


class Scale(Node):
    """Sends its input times ``factor``."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor
        self.prev = PrevPort(recv=VECTOR)  # receives from the previous side
        self.next = NextPort(send=VECTOR)  # sends to the next side

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x = self.prev.recv()  # the latest input; never waits
        self.next.send(self.factor * x)


class Shift(Node):
    """Sends its input plus ``offset``."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.prev.recv() + self.offset)


# --8<-- [start:network]
class Affine(mujou.Network):
    """``x -> scale -> shift ->``: y = 2 x + 1."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Scale(2.0)
        self.shift = Shift(1.0)
        self.scale.next.connect(self.shift.prev)
        self.prev = self.scale.prev  # the input of the network
        self.next = self.shift.next  # the output of the network


# --8<-- [end:network]


# --8<-- [start:run]
net = Affine()
x = torch.tensor([1.0, 2.0, 3.0])
# The nodes run on the run's worker threads until the end of the `with` block.
with mujou.run(net) as run:
    run.write({net.prev: x})  # the unconnected input port of scale
    # x takes a few updates to pass through the nodes: wait until y arrives, with
    # the time it was sent (TimeoutError after 5 s of real time).
    y, t = run.wait_for(net.next, lambda y: torch.equal(y, 2 * x + 1), timeout=5.0)

print(f"y = {y} at t = {float(t):.3f} s")
# --8<-- [end:run]
assert torch.equal(y, 2 * x + 1)
