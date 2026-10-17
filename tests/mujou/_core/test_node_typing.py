# pyright: strict
"""Checked by pyright in strict mode, as users' code and mujou's own nodes are:
subclassing ``mujou.Node`` and writing to a run must type-check cleanly."""

import torch

import mujou
from mujou import Network, Node, Port, PrevPort, TensorSpec


class Layer(Node):
    def __init__(self, n: int) -> None:
        super().__init__(seed=0)
        self.prev = PrevPort(recv=TensorSpec((n,)))
        self.out = Port(send=TensorSpec((n,)))


class Single(Network):
    def __init__(self) -> None:
        self.layer = Layer(2)
        self.prev = self.layer.prev


class TestStrictTyping:
    def test_a_subclass_and_a_prebuilt_write(self) -> None:
        net = Single()
        values = {net.prev: torch.ones(2)}  # a dict built before the call
        with mujou.run(net, inputs=values) as run:
            run.write(values)
            _, t = run.read(net.layer.out)
            assert t.ns >= 0
