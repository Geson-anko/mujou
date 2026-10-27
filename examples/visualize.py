"""Draw the graph of a network as a page of HTML, and open it in a browser.

``mujou.viz.to_html`` reads the network as ``mujou.run`` would, without running
it, and returns a self-contained page. This program writes ``graph.html`` for a
network of two networks and a node, and a ``pc.Chain``.
"""

from pathlib import Path
from typing import override

import torch
from torch import nn

import mujou
import mujou.pc as pc
import mujou.viz
from mujou import Node, Port, TensorSpec

VECTOR = TensorSpec((2,))


class Smooth(Node):
    """Follows its input with the time constant ``tau``."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.prev = Port(recv=VECTOR)
        self.next = Port(send=VECTOR)
        self.x = torch.zeros(2)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.x += min(dt / self.tau, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)


class Filter(mujou.Network):
    """Two ``Smooth`` in a row."""

    def __init__(self) -> None:
        super().__init__()
        self.fast = Smooth(0.01)
        self.slow = Smooth(0.1)
        self.fast.next.connect(self.slow.prev)
        self.prev, self.next = self.fast.prev, self.slow.next


class Monitor(Node):
    """Receives what the filters send."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = mujou.MultiPort(recv=VECTOR)


# --8<-- [start:viz]
class Plant(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.left = Filter()  # a network in a network
        self.right = Filter()
        self.monitor = Monitor()
        self.left.next.connect(self.monitor.prev)
        self.right.next.connect(self.monitor.prev)


net = Plant()
Path("graph.html").write_text(mujou.viz.to_html(net), "utf-8")  # open it in a browser
# --8<-- [end:viz]

# --8<-- [start:chain]
B = 8
chain = pc.Chain(
    nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
    nn.Linear(256, 10),
    x=TensorSpec((B, 784)),
    y=TensorSpec((B,), torch.int64),
    batch=B,
    loss=pc.loss.CrossEntropyLoss(B),
    tau=pc.TimeConstants(h=4e-3, lam=8e-3),
    optimizer=pc.optim.SGD(tau=1.28),
    learn_interval=0.15,
)
Path("chain.html").write_text(mujou.viz.to_html(chain), "utf-8")
# --8<-- [end:chain]
print("wrote graph.html and chain.html")
