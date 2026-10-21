"""Run several networks together with ``Group``.

A network given by keyword is at the path of its keyword (``"enc.scale"``), and
one given by position at the path of its position (``"0.scale"``).
``mujou.run(a, b)`` is ``mujou.run(Group(a, b))``.
"""

from typing import override

import torch

import mujou
from mujou import Group, Node, Port, TensorSpec

VECTOR = TensorSpec((2,))


class Scale(Node):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor
        self.prev = Port(recv=VECTOR)
        self.next = Port(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.factor * self.prev.recv())


class Stage(mujou.Network):
    """One node, with its ports exposed as the sides of the network."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.scale = Scale(factor)
        self.prev = self.scale.prev
        self.next = self.scale.next


# --8<-- [start:group]
# By keyword: two networks connected before they are grouped.
enc, dec = Stage(2.0), Stage(3.0)
enc.next.connect(dec.prev)  # the ports a network exposes, as for nodes
x = torch.tensor([1.0, -1.0])
with mujou.run(Group(enc=enc, dec=dec), inputs={enc.prev: x}) as run:
    # x takes a few updates to pass through the nodes, and how many fit in a given
    # time depends on how the threads get scheduled: follow the output until y
    # arrives (TimeoutError after 5 s).
    # The ports are those of the networks given.
    y, _ = run.wait_for(dec.next, lambda y: torch.equal(y, 6 * x), timeout=5.0)
    nodes = list(run.state_dict()["nodes"])
print(f"keyword: nodes {nodes}, y = {y}")
assert nodes == ["enc.scale", "dec.scale"]
assert torch.equal(y, 6 * x)

# By position: two independent networks in one run.
first, second = Stage(2.0), Stage(3.0)
with mujou.run(first, second, inputs={first.prev: x, second.prev: x}) as run:
    y1, _ = run.wait_for(first.next, lambda y: torch.equal(y, 2 * x), timeout=5.0)
    y2, _ = run.wait_for(second.next, lambda y: torch.equal(y, 3 * x), timeout=5.0)
    nodes = list(run.state_dict()["nodes"])
print(f"position: nodes {nodes}, y1 = {y1}, y2 = {y2}")
assert nodes == ["0.scale", "1.scale"]
assert torch.equal(y1, 2 * x) and torch.equal(y2, 3 * x)
# --8<-- [end:group]
