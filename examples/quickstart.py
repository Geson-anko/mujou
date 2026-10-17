"""The smallest program: two nodes in a network, run in the background.

The host writes an input, waits on the global time, reads the output and stops.
"""

from typing import override

import torch

import mujou
from mujou import NextPort, Node, PrevPort, TensorSpec

VECTOR = TensorSpec((3,))  # every message here is a float32 vector of 3


class Scale(Node):
    """Sends its input times ``factor``."""

    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.prev = PrevPort(recv=VECTOR)  # receives from the previous side
        self.next = NextPort(send=VECTOR)  # sends to the next side

    @override
    def on_update(self, dt: float) -> None:
        x = self.prev.recv()  # the latest input; never waits
        self.next.send(self.factor * x)


class Shift(Node):
    """Sends its input plus ``offset``."""

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        self.next.send(self.prev.recv() + self.offset)


class Affine(mujou.Network):
    """``x -> scale -> shift ->``: y = 2 x + 1."""

    def __init__(self) -> None:
        self.scale = Scale(2.0)
        self.shift = Shift(1.0)
        self.scale.connect(self.shift)  # scale.next <-> shift.prev
        self.prev = self.scale.prev  # the input of the network
        self.next = self.shift.next  # the output of the network


net = Affine()
x = torch.tensor([1.0, 2.0, 3.0])
# The nodes run on threads of their own until the end of the `with` block.
with mujou.run(net) as run:
    run.write({net.prev: x})  # the unconnected input port of scale
    mujou.time.sleep(0.1)  # 0.1 s of the global time
    y, t = run.read(net.next)  # the latest message and when it was sent

print(f"y = {y} at t = {float(t):.3f} s")
assert torch.equal(y, 2 * x + 1)
