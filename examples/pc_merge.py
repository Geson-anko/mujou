"""A node of PC-ALM of your own: ``Merge``, which has two inputs.

``pc.Layer`` reads one activity, so a node that joins two paths is written with
``pc.functional`` and ``pc.clock.LearnClock``. Two towers (``pc.Layer``) read one
``pc.Input``, and ``Merge`` predicts from both for a ``pc.Output``. The network
learns y = sin(2 x1) cos(x2) as ``pc_regression.py`` does: the loss on each
batch before learning from it falls below 0.01 in 10 s.
"""

from functools import partial
from typing import override

import torch
from torch import nn

import mujou
import mujou.pc as pc
from mujou import DictSpec, MultiPort, Network, Node, Port, TensorSpec

B = 32  # the batch
SHOWN = 0.032  # seconds of the run's timer each batch is shown
TAU = pc.TimeConstants(h=4e-3, lam=2e-3, r=4e-3)
OPTIMIZER = pc.optim.SGD(tau=SHOWN / 0.3)  # a rate of 0.3 per batch
LEARN = 0.75 * SHOWN  # the towers and the output learn once per batch


# --8<-- [start:merge]
class Merge(Node):
    """``mu = module(cat(h_a, h_b))``: a node of PC-ALM with two inputs."""

    def __init__(
        self,
        module: nn.Module,
        a: DictSpec[pc.msg.Activity],
        b: DictSpec[pc.msg.Activity],
    ) -> None:
        super().__init__(period=TAU.period)
        self.module = module
        self.optimizer = OPTIMIZER.bind(self.module.parameters())
        self.clock = pc.clock.LearnClock.of(
            TAU, period=self.period, learn_interval=None
        )
        h_a, h_b = a.field("h", TensorSpec), b.field("h", TensorSpec)
        self.sizes = [h_a.shape[-1], h_b.shape[-1]]
        h = TensorSpec((B, module(torch.zeros(B, sum(self.sizes))).shape[-1]))
        self.r, self.lam = h.zeros(), h.zeros()  # r = h - mu
        self.act_spec = pc.msg.activity_spec(h, a.field("y", TensorSpec), B)
        self.a = Port(send=h_a, recv=a)  # sends back g_in of h_a
        self.b = Port(send=h_b, recv=b)
        self.next = MultiPort(send=self.act_spec, recv=h)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        a, b = self.a.recv(), self.b.recv()
        t = self.clock.tick(dt, a["time"], self.timer.now())
        if t.new:  # the samples that start anew drop the credit of the ones before
            new = a["reset"].unsqueeze(-1)
            self.r.masked_fill_(new, 0.0)
            self.lam.masked_fill_(new, 0.0)
        labeled = a["mask"].unsqueeze(-1)  # a sample without a target sends no credit
        delta = torch.where(labeled, self.r + self.lam, 0)
        params = list(self.module.parameters())
        mu, g_in, g_theta = pc.functional.vjp(
            self.module, torch.cat([a["h"], b["h"]], dim=-1), lambda _: delta, params
        )
        g_a, g_b = g_in.split(self.sizes, dim=-1)
        self.a.send(g_a)
        self.b.send(g_b)
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        pc.functional.relax_(self.r, self.lam, g, t.dt, TAU)
        # The prediction goes on, with the y, mask, reset and time of a.
        self.next.send({**a, "h": mu})
        if a["mask"].any():  # nothing to learn from samples without a target
            self.optimizer.step_([g_p / B for g_p in g_theta], t.elapsed)


# --8<-- [end:merge]


# --8<-- [start:network]
class TwoTowers(Network, pc.Shown):
    def __init__(self) -> None:
        super().__init__()
        tower = partial(pc.Layer, tau=TAU, optimizer=OPTIMIZER, learn_interval=LEARN)
        self.inp = pc.Input(
            x=TensorSpec((B, 2)), y=TensorSpec((B, 1)), period=TAU.period, batch=B
        )
        self.a = tower(
            nn.Sequential(nn.Linear(2, 32), nn.Tanh()), act_spec=self.inp.act_spec
        )
        self.b = tower(
            nn.Sequential(nn.Linear(2, 32), nn.Tanh()), act_spec=self.inp.act_spec
        )
        self.merge = Merge(
            nn.Sequential(nn.Linear(64, 64), nn.Tanh()),
            self.a.act_spec,
            self.b.act_spec,
        )
        self.out = pc.Output(
            nn.Linear(64, 1),
            act_spec=self.merge.act_spec,
            loss=pc.loss.MSELoss(B),
            tau=TAU,
            optimizer=OPTIMIZER,
            learn_interval=LEARN,
        )
        self.inp.next.connect(self.a.prev)
        self.inp.next.connect(self.b.prev)
        self.a.next.connect(self.merge.a)
        self.b.next.connect(self.merge.b)
        self.merge.next.connect(self.out.prev)
        self.prev, self.next = self.inp.prev, self.out.next  # for pc.show


# --8<-- [end:network]


if __name__ == "__main__":
    torch.manual_seed(0)  # the initial weights, before the run
    net = TwoTowers()
    g = torch.Generator().manual_seed(1)
    mask = torch.ones(B, dtype=torch.bool)  # every sample has a target, and is new
    losses: list[float] = []
    with mujou.run(net) as run:
        for i in range(300):
            x = 2 * torch.rand(B, 2, generator=g) - 1
            y = torch.sin(2 * x[:, :1]) * torch.cos(x[:, 1:])
            sample = pc.msg.Sample(x=x, y=y, mask=mask, reset=mask)
            p = pc.show(run, net, sample, seconds=SHOWN)
            losses.append(nn.functional.mse_loss(p["y_hat"], y).item())
            if (i + 1) % 50 == 0:
                print(f"batch {i + 1}: loss {sum(losses[-50:]) / 50:.4f}")
