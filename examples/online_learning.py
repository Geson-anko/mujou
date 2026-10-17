"""Learning while the world changes: a learner follows a target that drifts.

A teacher sends one batch of samples every 10 ms from y = a x + b, and switches
``a`` from 2 to -1 partway through the run. The learner learns from each new batch
as it comes, while it runs: there is no training phase and no inference phase.
The host follows the learner's loss as it falls, jumps when the target changes,
and falls again.
"""

import math
from typing import TypedDict, override

import torch
from torch import nn

import mujou
from mujou import DictSpec, NextPort, Node, Port, PrevPort, TensorSpec

BATCH = 8
A_BEFORE, A_AFTER, B = 2.0, -1.0, 0.5
SWITCH = 1.5  # seconds after the teacher starts: when `a` changes
MAX_STEP = 0.01  # seconds: a longer explicit Euler step could diverge


class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((BATCH, 1)), y=TensorSpec((BATCH, 1)))
SCALAR = TensorSpec(())


class Teacher(Node):
    """Sends a batch of y = a x + b each update, with ``a`` switching at SWITCH."""

    def __init__(self) -> None:
        super().__init__(seed=0, period=0.01)
        self.next = NextPort(send=SAMPLE)

    @override
    def on_start(self) -> None:
        super().on_start()
        self.start = self.timer.now()

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        a = A_BEFORE if float(self.timer.now() - self.start) < SWITCH else A_AFTER
        x = torch.randn(BATCH, 1, generator=self.generator)
        self.next.send({"x": x, "y": a * x + B})


class Learner(Node):
    """Learns once from each new batch, and sends its loss on it before learning."""

    def __init__(self, rate: float) -> None:
        super().__init__(seed=0)
        self.rate = rate
        self.prev = PrevPort(recv=SAMPLE)
        self.loss = Port(send=SCALAR, initial=torch.tensor(math.inf))  # no batch yet

    @override
    def on_start(self) -> None:
        super().on_start()
        self.model = nn.Linear(1, 1, device=self.device)
        nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
        nn.init.zeros_(self.model.bias)
        _, self.last = self.prev.recv_with_time()  # skip the value there before
        self.elapsed = 0.0  # seconds since the last batch learned from

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.elapsed += dt
        sample, t = self.prev.recv_with_time()
        if t == self.last:  # nothing new since the last update
            return
        self.last = t
        x, y = sample["x"], sample["y"]
        self.loss.send(nn.functional.mse_loss(self.model(x), y))  # an unseen batch
        # Follow the gradient flow dW/dt = -rate * dL/dW over the time since the
        # last batch, in n equal steps no longer than MAX_STEP.
        params = list(self.model.parameters())
        n = max(1, math.ceil(self.elapsed / MAX_STEP))
        h = self.elapsed / n
        for _ in range(n):
            with torch.enable_grad():  # mujou: local-autograd
                loss = nn.functional.mse_loss(self.model(x), y)
                grads = torch.autograd.grad(loss, params)
            for p, g in zip(params, grads, strict=True):
                p -= h * self.rate * g
        self.elapsed = 0.0


torch.set_num_threads(1)  # each node computes on a thread of its own

teacher, learner = Teacher(), Learner(rate=5.0)
teacher.connect(learner)
with mujou.run(teacher, learner) as run:
    start = mujou.time.now()  # the run is on the global timer
    # How fast the learner learns depends on time, not on how often its thread
    # runs: follow its loss, giving each phase up to 5 s of real time.
    phases = [
        ("first batch", lambda loss: bool(loss.isfinite())),
        (f"learned a = {A_BEFORE}", lambda loss: loss.item() < 1e-4),
        ("the target changed", lambda loss: loss.item() > 1.0),
        (f"learned a = {A_AFTER} again", lambda loss: loss.item() < 1e-4),
    ]
    for what, condition in phases:
        loss, t = run.wait_for(learner.loss, condition, timeout=5.0)
        print(f"t = {float(t - start):.2f} s: loss = {loss.item():.6f}  ({what})")
model = run.state_dict()["nodes"]["1"]["model"]  # the learner's state at the end

weight, bias = model["weight"].item(), model["bias"].item()
print(f"the learner's weight = {weight:.3f}, bias = {bias:.3f}")
assert abs(weight - A_AFTER) < 0.1 and abs(bias - B) < 0.1
