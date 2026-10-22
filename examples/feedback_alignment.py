"""A learning rule across nodes that is not PC: feedback alignment.

Two nodes learn the two-layer network y_hat = W2 tanh(W1 x) from a stream of
samples that the host writes. ``hidden`` sends its activity forward, and
``output`` sends the error back on the same two-way port. ``hidden`` never sees
``W2`` (no node reads another's weights), so instead of the transpose of ``W2``
it multiplies the error by a fixed random matrix ``B`` of its own: ``W2`` comes
to align with the transpose of ``B``, and the network learns anyway (Lillicrap et
al., 2016, https://doi.org/10.1038/ncomms13276).

What a rule across nodes needs, and how this program does it:

- A two-way port: ``hidden.next`` sends the activity and receives the error.
- Relaying the target: the activity carries the target ``y`` and the time of the
  write it came from, so ``output`` pairs each prediction with its own target,
  and ``hidden`` each error with its own input.
- Telling what is new: each node learns once per sample, a while after its write,
  as a ``pc.LearnClock`` tells it, by the time since it last learned; ``hidden``
  learns only from the error of the sample it holds.

The host shows each batch for 20 ms of the run's timer and prints the loss of
``output`` on each batch before learning from it: it falls from about 1.5 to
about 0.02 in 6 s. With ``hidden`` frozen, it stays above 0.15: the error that
comes back through ``B`` teaches ``hidden`` too.
"""

from typing import TypedDict, override

import torch

import mujou
import mujou.pc as pc
from mujou import DictSpec, Node, Port, TensorSpec, TimestampSpec
from mujou.time import Timestamp

BATCH, N_IN, N_HIDDEN, N_OUT = 32, 8, 64, 2
SHOWN = 0.02  # seconds of the run's timer each batch is shown
LEARN = 0.75 * SHOWN  # a node learns this long after each write
TAU = SHOWN / 0.2  # the time constant of the weights: a rate of 0.2 per batch
PERIOD = 0.002


class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


class Activity(TypedDict):
    h: torch.Tensor
    y: torch.Tensor  # the target of the sample h was computed from
    time: Timestamp  # the time the sample was written at


class Error(TypedDict):
    e: torch.Tensor  # y_hat - y
    time: Timestamp  # the time of the sample, as the activity had it


TIME = TimestampSpec()
SAMPLE = DictSpec(Sample, x=TensorSpec((BATCH, N_IN)), y=TensorSpec((BATCH, N_OUT)))
ACTIVITY = DictSpec(
    Activity, h=TensorSpec((BATCH, N_HIDDEN)), y=TensorSpec((BATCH, N_OUT)), time=TIME
)
ERROR = DictSpec(Error, e=TensorSpec((BATCH, N_OUT)), time=TIME)


# --8<-- [start:hidden]
class Hidden(Node):
    """h = tanh(W1 x); learns W1 from the error that comes back through ``B``."""

    def __init__(self) -> None:
        super().__init__(period=PERIOD, seed=1)
        g = self.generator
        self.w1 = torch.randn(N_HIDDEN, N_IN, generator=g) / N_IN**0.5
        self.b = torch.randn(N_OUT, N_HIDDEN, generator=g) / N_OUT**0.5  # fixed
        self.clock = pc.LearnClock(
            period=PERIOD, learn_interval=LEARN, max_dt=2 * PERIOD
        )
        self.prev = Port(recv=SAMPLE)  # the host writes it
        self.next = Port(send=ACTIVITY, recv=ERROR)  # two-way

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, t = self.prev.recv_with_time()
        if t is None:  # nothing written yet
            return
        x = sample["x"]
        h = torch.tanh(x @ self.w1.T)
        self.next.send({"h": h, "y": sample["y"], "time": t})  # relays y
        error = self.next.recv()
        tick = self.clock.tick(dt, t.ns, self.timer.now().ns)
        if tick.learn and error["time"] == t:  # the error of this sample
            delta = (error["e"] @ self.b) * (1 - h**2)  # B e, not W2^T e
            self.w1 -= tick.elapsed / TAU * (delta.T @ x) / BATCH


# --8<-- [end:hidden]


# --8<-- [start:output]
class Output(Node):
    """y_hat = W2 h; sends the error back, and the loss on each new sample."""

    def __init__(self) -> None:
        super().__init__(period=PERIOD, seed=2)
        self.w2 = torch.randn(N_OUT, N_HIDDEN, generator=self.generator) / N_HIDDEN
        self.clock = pc.LearnClock(
            period=PERIOD, learn_interval=LEARN, max_dt=2 * PERIOD
        )
        self.prev = Port(send=ERROR, recv=ACTIVITY)  # two-way
        self.loss = Port(send=TensorSpec(()))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act, t = self.prev.recv_with_time()
        if t is None:  # hidden has not sent yet
            return
        h = act["h"]
        e = h @ self.w2.T - act["y"]  # the prediction against its own target
        self.prev.send({"e": e, "time": act["time"]})
        tick = self.clock.tick(dt, act["time"].ns, self.timer.now().ns)
        if tick.new:  # the first update with this sample, before learning from it
            self.loss.send(0.5 * e.pow(2).sum(dim=1).mean())
        if tick.learn:
            self.w2 -= tick.elapsed / TAU * (e.T @ h) / BATCH


# --8<-- [end:output]


class FeedbackAlignment(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = Hidden()
        self.output = Output()
        self.hidden.next.connect(self.output.prev)
        self.prev, self.loss = self.hidden.prev, self.output.loss


# The task: a fixed random two-layer network of tanh, which the nodes learn.
g = torch.Generator().manual_seed(0)
t1, t2 = torch.randn(16, N_IN, generator=g), torch.randn(N_OUT, 16, generator=g)


def teacher(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x @ t1.T / N_IN**0.5) @ t2.T


torch.set_num_threads(1)
net = FeedbackAlignment()
losses: list[float] = []
with mujou.run(net) as run:
    for i in range(300):
        x = torch.randn(BATCH, N_IN, generator=g)
        run.write({net.prev: {"x": x, "y": teacher(x)}})
        run.timer.sleep(SHOWN)
        losses.append(run.read(net.loss).item())  # the loss on this batch
        if i == 0 or (i + 1) % 50 == 0:
            recent = losses[-50:]
            mean = sum(recent) / len(recent)
            print(
                f"batch {i + 1:3d}: loss {mean:.4f} (the mean of the last {len(recent)})"
            )
