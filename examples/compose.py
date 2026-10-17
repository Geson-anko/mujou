"""Run several networks together with ``Compose``.

A network given by keyword is at the path of its keyword (``"enc.scale"``), and
one given by position at the path of its position (``"0.scale"``).
``mujou.run(a, b)`` is ``mujou.run(Compose(a, b))``.
"""

from typing import override

import torch

import mujou
from mujou import Compose, NextPort, Node, PrevPort, TensorSpec

VECTOR = TensorSpec((2,))


class Scale(Node):
    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        self.next.send(self.factor * self.prev.recv())


class Stage(mujou.Network):
    """One node, with its ports exposed as the sides of the network."""

    def __init__(self, factor: float) -> None:
        self.scale = Scale(factor)
        self.prev = self.scale.prev
        self.next = self.scale.next


# By keyword: two networks connected before they are composed.
enc, dec = Stage(2.0), Stage(3.0)
enc.connect(dec)  # enc.next <-> dec.prev, as for nodes
x = torch.tensor([1.0, -1.0])
with mujou.run(Compose(enc=enc, dec=dec), inputs={enc.prev: x}) as run:
    mujou.time.sleep(0.1)
    y, _ = run.read(dec.next)  # the ports are those of the networks given
    nodes = list(run.state_dict()["nodes"])
print(f"keyword: nodes {nodes}, y = {y}")
assert nodes == ["enc.scale", "dec.scale"]
assert torch.equal(y, 6 * x)

# By position: two independent networks in one run.
first, second = Stage(2.0), Stage(3.0)
with mujou.run(first, second, inputs={first.prev: x, second.prev: x}) as run:
    mujou.time.sleep(0.1)
    y1, _ = run.read(first.next)
    y2, _ = run.read(second.next)
    nodes = list(run.state_dict()["nodes"])
print(f"position: nodes {nodes}, y1 = {y1}, y2 = {y2}")
assert nodes == ["0.scale", "1.scale"]
assert torch.equal(y1, 2 * x) and torch.equal(y2, 3 * x)
