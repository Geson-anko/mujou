"""Networks that branch and merge.

A ``Port`` that only sends takes any number of peers, so one node can feed
several branches (fan-out); a ``MultiPort`` merges them, and its ``recv``
returns one message per channel. A ``MultiPort`` can also send to one channel
at a time with ``send_to``, given a channel that ``connect`` returned.
"""

from collections.abc import Callable
from typing import override

import torch

import mujou
from mujou import Channel, MultiPort, Node, Port, TensorSpec

VECTOR = TensorSpec((2,))


# --8<-- [start:fan-out]
class Apply(Node):
    """Sends ``fn(x)``."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.prev = Port(recv=VECTOR)
        self.next = Port(send=VECTOR)  # only sends: any number of peers

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.fn(self.prev.recv()))


class Sum(Node):
    """Sends the sum of the latest messages of all its peers."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = MultiPort(recv=VECTOR)  # merges any number of peers
        self.next = Port(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        received = self.prev.recv()  # dict[Channel, torch.Tensor], in connect order
        self.next.send(torch.stack(list(received.values())).sum(dim=0))


class Diamond(mujou.Network):
    """``shift -> (double, square) -> sum``: y = 2 (x + 1) + (x + 1)^2."""

    def __init__(self) -> None:
        super().__init__()
        self.shift = Apply(lambda x: x + 1)
        self.double = Apply(lambda x: 2 * x)
        self.square = Apply(lambda x: x**2)
        self.sum = Sum()
        self.shift.next.connect(self.double.prev)  # one port, two peers
        self.shift.next.connect(self.square.prev)
        self.double.next.connect(self.sum.prev)  # two peers, one MultiPort
        self.square.next.connect(self.sum.prev)
        self.prev, self.next = self.shift.prev, self.sum.next


# --8<-- [end:fan-out]


# --8<-- [start:send-to]
class Split(Node):
    """Sends ``x`` on the channel ``plus`` and ``-x`` on the channel ``minus``."""

    # Channels of self.next, set by the network to the ones connect returned.
    plus: Channel
    minus: Channel

    def __init__(self) -> None:
        super().__init__()
        self.prev = Port(recv=VECTOR)
        self.next = MultiPort(send=VECTOR)  # sends to each peer on its own

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x = self.prev.recv()
        self.next.send_to(self.plus, x)
        self.next.send_to(self.minus, -x)


class Magnitude(mujou.Network):
    """``split -> (relu, relu) -> sum``: y = relu(x) + relu(-x) = |x|."""

    def __init__(self) -> None:
        super().__init__()
        self.split = Split()
        self.pos, self.neg = Apply(torch.relu), Apply(torch.relu)
        self.sum = Sum()
        # Hand split the channels it sends on: the order of connect does not matter.
        self.split.plus = self.split.next.connect(self.pos.prev)
        self.split.minus = self.split.next.connect(self.neg.prev)
        self.pos.next.connect(self.sum.prev)
        self.neg.next.connect(self.sum.prev)
        self.prev, self.next = self.split.prev, self.sum.next


# --8<-- [end:send-to]


x = torch.tensor([1.0, -3.0])
for net, want in ((Diamond(), 2 * (x + 1) + (x + 1) ** 2), (Magnitude(), x.abs())):
    with mujou.run(net, inputs={net.prev: x}) as run:
        # x takes a few updates to pass through the nodes, and how many fit in a
        # given time depends on how the threads get scheduled: follow the output
        # until y arrives (TimeoutError after 5 s of real time).
        y, _ = run.wait_for(net.next, lambda y: torch.equal(y, want), timeout=5.0)
        # --8<-- [start:paths]
        print("nodes:", list(run.state_dict()["nodes"]))  # their paths from the root
        # --8<-- [end:paths]
    print(f"{type(net).__name__}: x = {x}, y = {y}")
