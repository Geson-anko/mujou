"""A network that branches and merges.

A ``Port`` that only sends takes any number of peers, so a node can feed several
branches; a ``MultiPort`` merges them, and its ``recv`` returns one message per
channel. A ``MultiPort``'s ``send_to`` sends on one channel only: one that
``connect`` returned.
"""

from collections.abc import Callable
from typing import override

import torch

import mujou
from mujou import Channel, MultiPort, Node, Port, TensorSpec

VECTOR = TensorSpec((2,))


# --8<-- [start:network]
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


class Apply(Node):
    """Sends ``fn(x)``."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.prev = Port(recv=VECTOR)
        self.next = Port(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x = self.prev.recv()
        self.next.send(self.fn(x))


class Sum(Node):
    """Sends the sum of the messages of all its peers."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = MultiPort(recv=VECTOR)  # merges any number of peers
        self.next = Port(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        received = self.prev.recv()  # dict[Channel, torch.Tensor]
        self.next.send(torch.stack(list(received.values())).sum(dim=0))


class Diamond(mujou.Network):
    """``split -> (double, square) -> sum``: y = 2 x + (-x)^2."""

    def __init__(self) -> None:
        super().__init__()
        self.split = Split()
        self.double = Apply(lambda x: 2 * x)
        self.square = Apply(lambda x: x**2)
        self.sum = Sum()
        # Hand split the channels it sends on: the order of connect does not matter.
        self.split.plus = self.split.next.connect(self.double.prev)
        self.split.minus = self.split.next.connect(self.square.prev)
        self.double.next.connect(self.sum.prev)
        self.square.next.connect(self.sum.prev)
        self.prev = self.split.prev
        self.next = self.sum.next


# --8<-- [end:network]


net = Diamond()
x = torch.tensor([1.0, 3.0])
want = 2 * x + (-x) ** 2
with mujou.run(net, inputs={net.prev: x}) as run:
    # x takes a few updates to pass through the nodes, and how many fit in a given
    # time depends on how the threads get scheduled: follow the output until y
    # arrives (TimeoutError after 5 s).
    y, _ = run.wait_for(net.next, lambda y: torch.equal(y, want), timeout=5.0)
    # --8<-- [start:paths]
    print("nodes:", list(run.state_dict()["nodes"]))  # their paths from the root
    # --8<-- [end:paths]

print(f"x = {x}, y = {y}")
assert torch.equal(y, want)
