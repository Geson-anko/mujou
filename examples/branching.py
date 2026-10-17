"""A network that branches and merges.

A ``NextPort`` takes many peers, so a node can feed several branches; a
``PrevPort(multi=True)`` merges them, and its ``recv`` returns one message per
channel. ``send_to`` sends on one channel only: one that ``connect`` returned.
"""

from collections.abc import Callable
from typing import override

import torch

import mujou
from mujou import Channel, NextPort, Node, PrevPort, TensorSpec

VECTOR = TensorSpec((2,))


class Split(Node):
    """Sends ``x`` on the channel ``plus`` and ``-x`` on the channel ``minus``."""

    # Channels of self.next, set by the network to the ones connect returned.
    plus: Channel
    minus: Channel

    def __init__(self) -> None:
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR)  # many peers by default

    @override
    def on_update(self, dt: float) -> None:
        x = self.prev.recv()
        self.next.send_to(self.plus, x)
        self.next.send_to(self.minus, -x)


class Apply(Node):
    """Sends ``fn(x)``."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.fn = fn
        self.prev = PrevPort(recv=VECTOR)
        self.next = NextPort(send=VECTOR, multi=False)  # exactly one peer here

    @override
    def on_update(self, dt: float) -> None:
        x = self.prev.recv()
        self.next.send(self.fn(x))


class Sum(Node):
    """Sends the sum of the messages of all its peers."""

    def __init__(self) -> None:
        self.prev = PrevPort(recv=VECTOR, multi=True)  # merges any number of peers
        self.next = NextPort(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        received = self.prev.recv()  # dict[Channel, torch.Tensor]
        self.next.send(torch.stack(list(received.values())).sum(dim=0))


class Diamond(mujou.Network):
    """``split -> (double, square) -> sum``: y = 2 x + (-x)^2."""

    def __init__(self) -> None:
        self.split = Split()
        self.double = Apply(lambda x: 2 * x)
        self.square = Apply(lambda x: x**2)
        self.sum = Sum()
        # Hand split the channels it sends on: the order of connect does not matter.
        self.split.plus = self.split.connect(self.double)
        self.split.minus = self.split.connect(self.square)
        self.double.connect(self.sum)
        self.square.connect(self.sum)
        self.prev = self.split.prev
        self.next = self.sum.next


net = Diamond()
graph = net.build()  # what mujou.run runs; mujou.run(net) builds the same graph
print("nodes:", [path for path, _ in graph.nodes])
print("output port:", graph.port_path(net.next))

x = torch.tensor([1.0, 3.0])
with mujou.run(net, inputs={net.prev: x}) as run:
    mujou.time.sleep(0.1)
    y, _ = run.read(net.next)

print(f"x = {x}, y = {y}")
assert torch.equal(y, 2 * x + (-x) ** 2)
