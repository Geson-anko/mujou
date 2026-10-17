"""A node that learns with PyTorch: backpropagation inside one node.

The framework turns gradient recording off on every node's thread. A node that
backpropagates within itself turns it on for that part, marked
``# mujou: local-autograd``. The update follows the gradient flow
dW/dt = -rate * dL/dW, so its size is proportional to ``dt``.
"""

from typing import override

import torch
from torch import nn

import mujou
from mujou import NextPort, Node, Port, PrevPort, TensorSpec

BATCH, N_IN, N_OUT = 16, 4, 2
INPUT, OUTPUT = TensorSpec((BATCH, N_IN)), TensorSpec((BATCH, N_OUT))


class Regressor(Node):
    """Learns to map its input to the target online, and sends its prediction."""

    def __init__(self, rate: float) -> None:
        super().__init__(seed=0)  # the seed of self.generator; computes on the CPU
        self.rate = rate
        self.prev = PrevPort(recv=INPUT)
        self.target = Port(recv=OUTPUT)
        self.next = NextPort(send=OUTPUT)

    @override
    def on_start(self) -> None:
        self.model = nn.Linear(N_IN, N_OUT, device=self.device)
        # The node's own generator, not the global one that all nodes share.
        nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
        nn.init.zeros_(self.model.bias)

    @override
    def on_update(self, dt: float) -> None:
        x, y = self.prev.recv(), self.target.recv()
        params = list(self.model.parameters())
        with torch.enable_grad():  # mujou: local-autograd
            loss = nn.functional.mse_loss(self.model(x), y)
            grads = torch.autograd.grad(loss, params)
        step = min(dt, 0.01) * self.rate  # clip dt to keep explicit Euler stable
        for p, g in zip(params, grads, strict=True):
            p -= step * g  # in place; no gradient is recorded outside the block
        self.next.send(self.model(x))


class Learner(mujou.Network):
    def __init__(self) -> None:
        self.regressor = Regressor(rate=20.0)
        self.prev = self.regressor.prev
        self.target = self.regressor.target
        self.next = self.regressor.next


# Each node computes on its own thread, and torch adds threads of its own for
# each: keep those few (a setting of the whole process, made by the host).
torch.set_num_threads(1)

generator = torch.Generator().manual_seed(0)
teacher = torch.randn(N_OUT, N_IN, generator=generator)
x = torch.randn(BATCH, N_IN, generator=generator)
y = x @ teacher.T

net = Learner()
with mujou.run(net, inputs={net.prev: x, net.target: y}) as run:
    losses: list[float] = []
    for _ in range(5):
        prediction, t = run.read(net.next)
        losses.append(nn.functional.mse_loss(prediction, y).item())
        print(f"t = {float(t):.2f} s: loss = {losses[-1]:.5f}")
        mujou.time.sleep(0.1)

assert losses[-1] < 1e-2 * losses[0]
