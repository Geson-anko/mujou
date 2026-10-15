# pyright: strict
"""Checked by pyright in strict mode, as users' code and mujou's own nodes are:
subclassing ``mujou.Node`` and writing to a network must type-check cleanly."""

import torch

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
        net.build()
        values = {net.prev: torch.ones(2)}  # a dict built before the call
        net.write(values)
