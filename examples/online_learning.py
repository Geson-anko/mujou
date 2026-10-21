"""Learning while the world changes: a learner follows a target that drifts.

A teacher sends one batch of samples every 10 ms from y = a x + b, where the host
sets ``a`` and changes it from 2 to -1 while the network runs. The learner learns
from each new batch as it comes: there is no training phase and no inference
phase. The host follows the learner's loss as it falls, jumps when the target
changes, and falls again.
"""

import math
from typing import TypedDict, override

import torch
from torch import nn

import mujou
from mujou import DictSpec, NextPort, Node, Port, PrevPort, TensorSpec
from mujou.time import Timestamp

BATCH = 8
A_BEFORE, A_AFTER, B = 2.0, -1.0, 0.5
MAX_STEP = 0.01  # seconds: a longer explicit Euler step could diverge


class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((BATCH, 1)), y=TensorSpec((BATCH, 1)))
SCALAR = TensorSpec(())


class Teacher(Node):
    """Sends a batch of y = a x + b each update, with the ``a`` the host writes."""

    def __init__(self) -> None:
        super().__init__(seed=0, period=0.01)
        self.a = Port(recv=SCALAR)  # unconnected: the host writes it
        self.next = NextPort(send=SAMPLE)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x = torch.randn(BATCH, 1, generator=self.generator)
        self.next.send({"x": x, "y": self.a.recv() * x + B})


class Learner(Node):
    """Learns once from each new batch, and sends its loss on it before learning."""

    def __init__(self, rate: float) -> None:
        # Updating more often than the teacher sends would only re-read its batch.
        super().__init__(seed=0, period=0.005)
        self.rate = rate
        self.prev = PrevPort(recv=SAMPLE)
        self.loss = Port(send=SCALAR, initial=torch.tensor(math.inf))  # no batch yet

    @override
    def on_start(self) -> None:
        super().on_start()
        self.model = nn.Linear(1, 1, device=self.device)
        nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
        nn.init.zeros_(self.model.bias)
        self.last: Timestamp | None = None  # the time of the last batch learned from
        self.elapsed = 0.0  # seconds since the last batch learned from

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.elapsed += dt
        sample, t = self.prev.recv_with_time()
        if t is None or t == self.last:  # the teacher has not sent, or nothing new
            return
        self.last = t
        x, y = sample["x"], sample["y"]
        with torch.no_grad():  # the loss on an unseen batch, before learning
            self.loss.send(nn.functional.mse_loss(self.model(x), y))
        # Follow the gradient flow dW/dt = -rate * dL/dW over the time since the
        # last batch, in n equal steps no longer than MAX_STEP.
        params = list(self.model.parameters())
        n = max(1, math.ceil(self.elapsed / MAX_STEP))
        h = self.elapsed / n
        for _ in range(n):
            loss = nn.functional.mse_loss(self.model(x), y)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():  # the step itself is not part of any graph
                for p, g in zip(params, grads, strict=True):
                    p -= h * self.rate * g
        self.elapsed = 0.0


torch.set_num_threads(1)  # the nodes compute on the run's worker threads


def learned(loss: torch.Tensor) -> bool:
    return loss.item() < 1e-4


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
model = run.state_dict()["nodes"]["learner"]["model"]  # the learner's state at the end

weight, bias = model["weight"].item(), model["bias"].item()
print(f"the learner's weight = {weight:.3f}, bias = {bias:.3f}")
assert abs(weight - A_AFTER) < 0.1 and abs(bias - B) < 0.1
