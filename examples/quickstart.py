"""The smallest program: two nodes in a network, running in the background.

The host writes an input, waits until the output arrives, and stops.
"""

from typing import override

import torch

import mujou
from mujou import Network, Node, Port, TensorSpec

VECTOR = TensorSpec((3,))  # every message here is a float32 vector of 3


# --8<-- [start:node]
class Affine(Node):
    """Sends ``a * x + b``, where ``x`` is the latest input."""

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a, self.b = a, b  # settings, not state
        self.prev = Port(recv=VECTOR)  # receives from the previous node
        self.next = Port(send=VECTOR)  # sends to the next node

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.a * self.prev.recv() + self.b)  # recv() never waits


# --8<-- [end:node]


# --8<-- [start:network]
class Pipeline(Network):
    """``x -> scale -> shift ->``: y = 2 x + 1."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Affine(2.0, 0.0)
        self.shift = Affine(1.0, 1.0)
        self.scale.next.connect(self.shift.prev)
        self.prev, self.next = self.scale.prev, self.shift.next  # exposed as its own


# --8<-- [end:network]


# --8<-- [start:run]
net = Pipeline()
x = torch.tensor([1.0, 2.0, 3.0])
# The nodes run on the run's worker threads until the end of the `with` block.
with mujou.run(net) as run:
    run.write({net.prev: x})
    # x takes a few updates to pass through the nodes: wait until y arrives, with
    # the time it was sent (TimeoutError after 5 s of real time).
    y, t = run.wait_for(net.next, lambda y: torch.equal(y, 2 * x + 1), timeout=5.0)
    assert t is not None  # None only for the zeros before the first send

print(f"y = {y} at t = {float(t):.3f} s")
# --8<-- [end:run]
