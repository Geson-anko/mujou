"""PC-ALM (augmented Lagrangian predictive coding) as nodes of mujou.

Each node holds a part of the model and learns from its neighbors' messages
only: ``Input`` takes the samples the host writes, each ``Layer`` wraps a module
as one constraint ``h = f(h_in)``, and ``Output`` holds the last module and the
loss. Inside a node, the module is differentiated by autograd; between nodes,
activities (``Activity``) flow to the next node and gradients of the energy to
the previous one. The target of each sample travels with its input in the
``Activity``, so the output node pairs every prediction with its own target.
``Chain`` connects the nodes for a sequence of modules, and ``show`` shows it a
sample. Each node gives the spec of what it sends as ``act_spec``, which the
next node takes, to build other shapes by hand. ``MSELoss`` and
``CrossEntropyLoss`` give the loss of each sample. ``SGD`` and ``AdamW`` set
how the parameters of a node move in time (``Optimizer``): each node binds the
setting to its own parameters (``BoundOptimizer``). ``LearnClock`` and
``functional`` hold the parts the nodes are made of, for writing nodes of your
own.

The samples are independent and the shapes start with the batch, ``batch``:
``()`` (the default) for one sample at a time, ``B`` for ``B`` samples, or a
shape of several dimensions.

Import it as a submodule::

    import mujou.pc as pc

    B = 64
    net = pc.Chain(
        nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
        nn.Linear(256, 10),
        x=TensorSpec((B, 784)),
        y=TensorSpec((B,), torch.int64),
        batch=B,
        loss=pc.CrossEntropyLoss(B),
        tau=pc.TimeConstants(h=4e-3, lam=8e-3),
        optimizer=pc.SGD(tau=1.28),
        learn_interval=0.15,
    )
"""

from mujou.pc import functional as functional
from mujou.pc.chain import Chain, Shown, show
from mujou.pc.clock import LearnClock, Tick
from mujou.pc.input import Input
from mujou.pc.layer import Layer
from mujou.pc.losses import CrossEntropyLoss, MSELoss
from mujou.pc.messages import Activity, Prediction, Sample, activity_spec
from mujou.pc.optim import SGD, AdamW, BoundOptimizer, Optimizer
from mujou.pc.output import Output
from mujou.pc.time_constants import TimeConstants

__all__ = [
    "Chain",
    "Shown",
    "show",
    "Input",
    "Layer",
    "Output",
    "TimeConstants",
    "Optimizer",
    "BoundOptimizer",
    "SGD",
    "AdamW",
    "LearnClock",
    "Tick",
    "Sample",
    "Activity",
    "Prediction",
    "activity_spec",
    "MSELoss",
    "CrossEntropyLoss",
    "functional",
]
