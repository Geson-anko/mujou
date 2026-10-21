"""Learning while the world changes: backpropagation inside one node.

A teacher sends a batch of samples from y = a x + b every 10 ms, with the ``a``
that the host writes; the host changes ``a`` from 2 to -1 while the network
runs. The learner learns from each new batch as it comes, with plain PyTorch
autograd inside the node: there is no training phase and no inference phase.
The host follows the learner's loss as it falls, jumps when the target changes,
and falls again.
"""

import math
from typing import TypedDict, override

import torch
from torch import nn

import mujou
from mujou import DictSpec, Node, Port, TensorSpec

BATCH = 8
A_BEFORE, A_AFTER, B = 2.0, -1.0, 0.5


class Batch(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


BATCH_SPEC = DictSpec(Batch, x=TensorSpec((BATCH, 1)), y=TensorSpec((BATCH, 1)))
SCALAR = TensorSpec(())


class Teacher(Node):
    """Sends a batch of y = a x + b each update, with the ``a`` the host writes."""

    def __init__(self) -> None:
        super().__init__(period=0.01, seed=0)
        self.a = Port(recv=SCALAR)  # unconnected: the host writes it
        self.next = Port(send=BATCH_SPEC)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x = torch.randn(BATCH, 1, generator=self.generator)
        self.next.send({"x": x, "y": self.a.recv() * x + B})


# --8<-- [start:node]
class Learner(Node):
    """Follows the gradient flow dW/dt = -rate * dL/dW, learning once from each
    new batch for the time since the last one, and sends its loss on each batch
    before learning from it."""

    MAX_STEP = 0.01  # seconds: a longer explicit Euler step could diverge

    def __init__(self, rate: float) -> None:
        # Updating more often than the teacher sends would only re-read its batch.
        super().__init__(period=0.005, seed=0)
        self.rate = rate  # a setting, not state
        self.model = nn.Linear(1, 1)  # has state_dict: in the state by itself
        # seed reproduces only self.generator: initialize the module from it.
        nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
        nn.init.zeros_(self.model.bias)
        self.last = -1  # the time (ns) of the batch learned from last
        self.elapsed = 0.0  # seconds since then
        self.register_state("last")  # plain values that change: register them
        self.register_state("elapsed")
        self.prev = Port(recv=BATCH_SPEC)
        self.loss = Port(send=SCALAR, initial=torch.tensor(math.inf))  # no batch yet

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.elapsed += dt
        batch, t = self.prev.recv_with_time()
        if t is None or t.ns == self.last:  # nothing sent yet, or nothing new
            return
        self.last = t.ns
        x, y = batch["x"], batch["y"]
        with torch.no_grad():  # the loss on an unseen batch needs no graph
            self.loss.send(nn.functional.mse_loss(self.model(x), y))
        # How far to learn is set by the time, not by the number of batches: move
        # by the elapsed time, in n equal steps no longer than MAX_STEP.
        n = max(1, math.ceil(self.elapsed / self.MAX_STEP))
        for _ in range(n):
            loss = nn.functional.mse_loss(self.model(x), y)  # records the graph
            loss.backward()
            with torch.no_grad():  # the step itself is not part of any graph
                for p in self.model.parameters():
                    assert p.grad is not None
                    p -= self.elapsed / n * self.rate * p.grad
                    p.grad = None
        self.elapsed = 0.0


# --8<-- [end:node]


def learned(loss: torch.Tensor) -> bool:
    return loss.item() < 1e-4


# --8<-- [start:threads]
# The nodes compute on the run's worker threads, and torch adds threads of its
# own for each: keep those few (a setting of the whole process, made by the host).
torch.set_num_threads(1)
# --8<-- [end:threads]

# --8<-- [start:run]
teacher, learner = Teacher(), Learner(rate=5.0)
teacher.next.connect(learner.prev)
net = mujou.Group(teacher=teacher, learner=learner)
with mujou.run(net, inputs={teacher.a: torch.tensor(A_BEFORE)}) as run:
    # How fast the learner learns depends on time, not on how often it runs:
    # follow its loss, giving each phase up to 5 s of real time.
    loss, t = run.wait_for(learner.loss, learned, timeout=5.0)
    assert t is not None  # the loss on a batch, not the initial inf
    print(f"t = {float(t):.2f} s: loss = {loss.item():.6f}  (learned a = {A_BEFORE})")
    # The world changes: the same learner keeps learning, with no new phase.
    changed_at = run.write({teacher.a: torch.tensor(A_AFTER)})
    loss, t = run.wait_for(learner.loss, lambda loss: loss.item() > 1e-2, timeout=5.0)
    assert t is not None
    print(
        f"t = {float(t):.2f} s: loss = {loss.item():.6f}  (a changed at {float(changed_at):.2f} s)"
    )
    loss, t = run.wait_for(learner.loss, learned, timeout=5.0)
    assert t is not None
    print(f"t = {float(t):.2f} s: loss = {loss.item():.6f}  (learned a = {A_AFTER})")

# After the run, the nodes' attributes are the host's to read.
weight, bias = learner.model.weight.item(), learner.model.bias.item()
print(f"the learner's weight = {weight:.3f}, bias = {bias:.3f}")
# --8<-- [end:run]
