"""PC-ALM (augmented Lagrangian predictive coding) as nodes of mujou.

Each node holds a part of the model and learns from its neighbors' messages
only: ``Input`` takes the samples the host writes, each ``Layer`` wraps a module
as one constraint ``h = f(h_in)``, and ``Output`` holds the last module and the
loss. Inside a node, the module is differentiated by autograd; between nodes,
activities (``Activity``) flow to the next node and gradients of the energy to
the previous one. The target of each sample travels with its input in the
``Activity``, so the output node pairs every prediction with its own target. Each
node gives the spec of what it sends as ``act``, which the next node takes.
``functional`` holds the stateless computations the nodes are made of, for
writing nodes of your own.

Import it as a submodule::

    import mujou.pc as pc

    B = 64
    tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=1.28)
    inp = pc.Input(
        x=TensorSpec((B, 784)), y=TensorSpec((B,), torch.int64), period=tau.h / 4
    )
    hidden = pc.Layer(
        nn.Sequential(nn.Linear(784, 256), nn.ReLU()), act=inp.act, tau=tau
    )
    out = pc.Output(
        nn.Linear(256, 10), act=hidden.act, loss=pc.cross_entropy, tau=tau
    )
    inp.next.connect(hidden.prev)
    hidden.next.connect(out.prev)
"""

from mujou._core.pc.input import Input
from mujou._core.pc.layer import Layer
from mujou._core.pc.losses import cross_entropy, mse
from mujou._core.pc.messages import Activity, Prediction, Sample, activity_spec
from mujou._core.pc.output import Output
from mujou._core.pc.time_constants import TimeConstants
from mujou.pc import functional as functional

__all__ = [
    "Activity",
    "Input",
    "Layer",
    "Output",
    "Prediction",
    "Sample",
    "TimeConstants",
    "activity_spec",
    "cross_entropy",
    "functional",
    "mse",
]
