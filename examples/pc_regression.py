"""Regression with PC-ALM: ``pc.Chain`` and ``pc.show``.

A chain of ``mujou.pc`` nodes learns y = sin(2 x1) cos(x2) from a stream of
batches. ``pc.Chain`` makes one node per module and connects them; ``pc.show``
writes each batch, shows it for ``SHOWN`` seconds of the run's timer while the
nodes learn from it, and then returns the first prediction the chain made for it
(before learning from it). All that is chosen
here are the time constants and the time each batch is shown.

The loss on each batch before learning from it (prequential) falls from about
0.4 to below 0.01 in 10 s.
"""

import torch
from torch import nn

import mujou
import mujou.pc as pc
from mujou import TensorSpec

B = 32  # the batch: x and y start with it, and so does the loss of each sample

# --8<-- [start:chain]
SHOWN = 0.032  # seconds of the run's timer each batch is shown
# tau.h sets how fast the activities relax (and the period of the nodes, h / 4),
# and the weights learn once per batch, by SGD at the rate SHOWN / theta = 0.3.
tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=SHOWN / 0.3)
torch.manual_seed(0)  # the initial weights, before the run
net = pc.Chain(
    nn.Sequential(nn.Linear(2, 64), nn.Tanh()),
    nn.Sequential(nn.Linear(64, 64), nn.Tanh()),
    nn.Linear(64, 1),
    x=TensorSpec((B, 2)),
    y=TensorSpec((B, 1)),
    batch=B,
    loss=pc.MSELoss(B),  # the loss of each sample, (B,)
    tau=tau,
    learn_interval=0.75 * SHOWN,  # learn once, 3/4 of the way through each batch
)
# --8<-- [end:chain]

# --8<-- [start:run]
torch.set_num_threads(1)
g = torch.Generator().manual_seed(1)
every = torch.ones(B, dtype=torch.bool)  # every sample has a target
losses: list[float] = []
with mujou.run(net) as run:
    for i in range(300):
        x = 2 * torch.rand(B, 2, generator=g) - 1
        y = torch.sin(2 * x[:, :1]) * torch.cos(x[:, 1:])
        p = pc.show(run, net, pc.Sample(x=x, y=y, mask=every), seconds=SHOWN)
        losses.append(nn.functional.mse_loss(p["y_hat"], y).item())  # before learning
        if i == 0 or (i + 1) % 50 == 0:
            recent = losses[-50:]
            mean = sum(recent) / len(recent)
            print(
                f"batch {i + 1:3d}: loss {mean:.4f} (the mean of the last {len(recent)})"
            )
    # The state of a node, by its path: input, layer0, layer1, output.
    lam = run.state_dict()["nodes"]["layer0"]["state"]["lam"]
print(f"the mean |lam| of layer0 (its credit) at the end: {lam.abs().mean():.2e}")
# --8<-- [end:run]
