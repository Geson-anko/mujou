"""A node that learns with PyTorch: backpropagation inside one node.

A hook records gradients as plain PyTorch does, and the tensors a node sends
are detached, so the graph stays inside the node. The update follows the
gradient flow dW/dt = -rate * dL/dW over the whole ``dt``, in explicit Euler
steps no longer than ``MAX_STEP``.
"""

import math
from typing import override

import torch
from torch import nn

import mujou
from mujou import NextPort, Node, Port, PrevPort, TensorSpec

# --8<-- [start:node]
BATCH, N_IN, N_OUT = 16, 4, 2
INPUT, OUTPUT = TensorSpec((BATCH, N_IN)), TensorSpec((BATCH, N_OUT))
MAX_STEP = 0.01  # seconds: a longer explicit Euler step could diverge


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
        super().on_start()
        self.model = nn.Linear(N_IN, N_OUT, device=self.device)
        # The node's own generator, not the global one that all nodes share.
        nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
        nn.init.zeros_(self.model.bias)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x, y = self.prev.recv(), self.target.recv()
        params = list(self.model.parameters())
        # Split a long dt into n equal steps instead of clipping it: none of the
        # elapsed time is lost.
        n = max(1, math.ceil(dt / MAX_STEP))
        h = dt / n
        for _ in range(n):
            loss = nn.functional.mse_loss(self.model(x), y)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():  # the step itself is not part of any graph
                for p, g in zip(params, grads, strict=True):
                    p -= h * self.rate * g
        with torch.no_grad():  # the prediction needs no graph: it is sent detached
            self.next.send(self.model(x))


# --8<-- [end:node]


# --8<-- [start:threads]
# The nodes compute on the run's worker threads, and torch adds threads of its
# own for each: keep those few (a setting of the whole process, made by the host).
torch.set_num_threads(1)
# --8<-- [end:threads]

generator = torch.Generator().manual_seed(0)
teacher = torch.randn(N_OUT, N_IN, generator=generator)
x = torch.randn(BATCH, N_IN, generator=generator)
y = x @ teacher.T

regressor = Regressor(rate=20.0)  # a node runs by itself, without a network
with mujou.run(regressor, inputs={regressor.prev: x, regressor.target: y}) as run:
    losses: list[float] = []
    # How far the node has learned by a given time depends on how it gets
    # scheduled: read every 0.1 s until the loss has fallen, giving up after 5 s.
    for _ in range(50):
        prediction, t = run.read_with_time(regressor.next)
        if t is None:  # the zeros the port holds until the node first sends
            run.timer.sleep(0.1)
            continue
        losses.append(nn.functional.mse_loss(prediction, y).item())
        print(f"t = {float(t):.2f} s: loss = {losses[-1]:.5f}")
        if losses[-1] < 1e-2 * losses[0]:
            break
        run.timer.sleep(0.1)

assert losses[-1] < 1e-2 * losses[0]
