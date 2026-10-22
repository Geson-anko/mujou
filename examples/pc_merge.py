"""A node of PC-ALM of your own: ``Merge``, which has two inputs.

``pc.Layer`` reads one activity, so a node that joins two paths is written with
``pc.functional`` and ``pc.LearnClock``. Two towers (``pc.Layer``) read one
``pc.Input``, and ``Merge`` predicts from both for a ``pc.Output``. ``Merge``
learns in every update, so it needs none of the handling of the jumps of its
weights that a node learning at intervals does. The network learns
y = sin(2 x1) cos(x2) as ``pc_regression.py`` does: the loss on each batch before
learning from it falls to about 0.02 in 10 s.
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
TAU = pc.TimeConstants(h=4e-3, lam=8e-3, theta=SHOWN / 0.3)
LEARN = 0.75 * SHOWN  # the towers and the output learn once per batch


# --8<-- [start:merge]
class Merge(Node):
    """``mu = module(cat(h_a, h_b))``: a node of PC-ALM with two inputs."""

    def __init__(
        self, module: nn.Module, a: DictSpec[pc.Activity], b: DictSpec[pc.Activity]
    ) -> None:
        super().__init__(period=TAU.period)
        self.module = module
        self.clock = pc.LearnClock.of(TAU, period=self.period, learn_interval=None)
        h_a, h_b = a.field("h", TensorSpec), b.field("h", TensorSpec)
        self.sizes = [h_a.shape[-1], h_b.shape[-1]]
        h = TensorSpec((B, module(torch.zeros(B, sum(self.sizes))).shape[-1]))
        self.h, self.lam, self.mu = h.zeros(), h.zeros(), h.zeros()
        self.jumps, self.seen = 0, [0, 0]  # the jumps it sent, and read from a, b
        self.register_state("jumps")
        self.register_state("seen")
        self.act_spec = pc.activity_spec(h, a.field("y", TensorSpec), B)
        self.a = Port(send=h_a, recv=a)  # sends back g_in of h_a
        self.b = Port(send=h_b, recv=b)
        self.next = MultiPort(send=self.act_spec, recv=h)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        a, b = self.a.recv(), self.b.recv()
        t = self.clock.tick(dt, a["time"].ns, self.timer.now().ns)
        # A new sample or a jump of either input: h jumps with mu, keeping r.
        seen = [a["jumps"], b["jumps"]]
        jump, self.seen = t.new or seen != self.seen, seen
        r = self.h - self.mu
        labeled = a["mask"].unsqueeze(-1)  # a sample without a target sends no credit
        params = list(self.module.parameters())
        mu, g_in, g_theta = pc.functional.vjp(
            self.module,
            torch.cat([a["h"], b["h"]], dim=-1),
            lambda mu: torch.where(labeled, (r if jump else self.h - mu) + self.lam, 0),
            params,
        )
        g_a, g_b = g_in.split(self.sizes, dim=-1)
        self.a.send(g_a)
        self.b.send(g_b)
        if jump:
            self.h = mu + r
            self.jumps += 1
        g = sum(self.next.recv().values(), torch.zeros_like(self.h))
        pc.functional.relax_(self.h, self.lam, mu, g, t.dt, TAU)
        self.mu = mu
        # The y, mask and time of a go on with h.
        self.next.send({**a, "h": self.h, "jumps": self.jumps})
        if a["mask"].any():  # nothing to learn from samples without a target
            pc.functional.learn_(params, g_theta, t.elapsed, TAU, B)


# --8<-- [end:merge]


# --8<-- [start:network]
class TwoTowers(Network):
    def __init__(self) -> None:
        super().__init__()
        tower = partial(pc.Layer, tau=TAU, learn_interval=LEARN)
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
            loss=pc.MSELoss(B),
            tau=TAU,
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
    mask = torch.ones(B, dtype=torch.bool)  # every sample has a target
    losses: list[float] = []
    with mujou.run(net) as run:
        for i in range(300):
            x = 2 * torch.rand(B, 2, generator=g) - 1
            y = torch.sin(2 * x[:, :1]) * torch.cos(x[:, 1:])
            p = pc.show(run, net, pc.Sample(x=x, y=y, mask=mask), seconds=SHOWN)
            losses.append(nn.functional.mse_loss(p["y_hat"], y).item())
            if (i + 1) % 50 == 0:
                print(f"batch {i + 1}: loss {sum(losses[-50:]) / 50:.4f}")
