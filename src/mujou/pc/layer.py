"""The hidden node of ``mujou.pc``."""

import math
from typing import override

import torch
from torch import nn

from mujou import DictSpec, MultiPort, Node, Port, TensorSpec
from mujou.pc.clock import LearnClock
from mujou.pc.functional import relax_, vjp
from mujou.pc.msg import Activity, activity_spec, rows
from mujou.pc.optim import Optimizer
from mujou.pc.relax import Euler, Relaxer
from mujou.pc.time_constants import TimeConstants
from mujou.types import Device


class Layer(Node):
    r"""A node that wraps ``module`` as one constraint ``h = f(h_in; theta)`` of
    PC-ALM.

    It predicts ``mu = module(h_in)`` from the activity ``h_in`` of the previous
    node, and holds the deviation $r = h - \mu$ of its own activity and the
    dual ``lam`` of the constraint. It sends ``h = mu + r`` to the next nodes.
    With $\delta = r + \lambda$ and the sum $g$ of what the next nodes send
    back:

    $$
    \tau_h \dot r = -(r + \lambda + g) - \frac{\tau_h}{\tau_r}\,r,\qquad
    \tau_\lambda \dot\lambda = r,
    $$

    and ``theta`` moves by ``optimizer`` (such as ``pc.optim.SGD``,
    $\tau_\theta \dot\theta = -g_\theta / B$) with the mean gradient
    $g_\theta / B$, where $B$ is the number of samples in the batch (1 without
    one), $g_\theta = -J_\theta^\top \delta$ and the node sends
    $g_{\mathrm{in}} = -J_h^\top \delta$ to the previous node (both from one
    vector-Jacobian product of ``module``, ``functional.vjp``). At a fixed point,
    $r = 0$ (``h`` is the forward pass) and ``lam`` is the credit
    $-\partial\ell/\partial h$ of backpropagation. Inside the node, ``module``
    is differentiated by autograd; between nodes, only these messages flow.

    ``h`` follows ``mu`` in the update that computes it: whatever moves ``mu``
    (the relaxation of the nodes below, a new sample, an update of ``theta``
    here or below), ``r`` and ``lam`` stay and only relax. Without a target, ``r`` decays to 0 and ``h``
    is the forward pass. As ``h`` moves with every ``r`` below it, the credit
    loops back through the product of the Jacobians, and a deep chain diverges
    without the decay of ``r`` (``tau.r``; ``TimeConstants`` has the
    measurements).

    Each ``on_update(dt)``:

    1. reads the activity and ticks its ``clock``, which caps ``dt`` at
       ``max_dt``;
    2. computes ``mu``, ``g_in`` and, when it learns, ``g_theta`` with the
       ``delta`` before this update, zero for the samples whose ``mask`` is
       False, and sends ``g_in``;
    3. moves ``r`` and then ``lam`` by ``dt`` with the ``g`` the next nodes
       sent (``functional.relax_``);
    4. sends ``h = mu + r``, and the ``y``, ``mask`` and ``time`` that came
       with ``h_in``;
    5. when it learns, moves ``theta`` by the time since its last update
       (``optimizer.step_``).

    Its ``clock`` (a ``LearnClock``) tells when it learns: in its first update
    once ``learn_interval`` seconds have passed since the host wrote the input
    (its ``time``, on the run's time), and then every ``learn_interval``
    seconds while the input stays, but never in the update that first sees an
    input (its ``delta`` is still that of the input before); with
    ``learn_interval=None``, in every update; with ``math.inf``, never. The
    nodes read the one timer of the run, so the nodes on a path learn at about
    the same time, however deep they are, and each from the relaxed ``delta``
    of the same input. A ``learn_interval``
    longer than the round trip of the credit down the path and back
    (``2 * hops * period``, with ``hops`` the number of connections from the
    ``Input`` to the ``Output``) is recommended, so that the nodes learn from
    the relaxed ``delta``. The node does not learn when no sample of the input
    has a target (every ``mask`` is False): that time is dropped from the next
    update, which would otherwise learn from the ``delta`` left over from an
    earlier input. Nor does it keep the time of an input that changed before
    the node learned from it. Until the first sample reaches the node, the
    activity is zeros (``mask`` all False, ``time`` 0), so it does not learn
    then.

    ``dt`` is capped because the nodes are coupled explicitly: ``g`` was made
    from ``h`` as it was when the next node last updated. After a long stall
    (a loaded machine), the step ``dt / tau.h`` of every node grows at once and
    the coupling makes the network diverge, which no split of the step inside
    the node prevents. The time beyond ``max_dt`` is dropped, from the
    relaxation and from the weight of the next update of ``theta`` alike. The
    default ``tau.max_dt`` (``tau.h / 4`` with ``tau.r = tau.h``) keeps the step
    of ``r`` and ``lam`` stable in the node itself.

    ``relaxer`` sets how ``r`` moves along the force on it (``functional.relax_``):
    the plain step of ``pc.relax.Euler``, or ``pc.relax.HeavyBall``, whose velocity
    lets a force that lasts move ``r`` faster, with the same fixed point. The
    node calls the ``reset_`` of the bound relaxer with each new sample (where
    the forces it remembers are those of the sample before).

    ``module`` must treat the samples of the batch independently (no
    ``BatchNorm`` in training mode): the gradient of each sample must be its
    own. It must also be deterministic (no ``Dropout``): noise in ``mu`` goes
    straight into the ``h`` the node sends. Its parameters that require
    gradients learn; set ``requires_grad`` to False to freeze one. It is called
    once on zeros when the node is made, to find the shape and dtype of ``h``.

    The state (``Run.state_dict()["nodes"][path]["state"]``) holds
    ``"generator"`` (the state of the node's ``generator``), ``"module"`` (its
    ``state_dict``), ``"r"`` and ``"lam"`` (``(*batch, *out)``), when to learn
    (``"clock"``, the state of the ``LearnClock``), the state of ``optimizer``
    (``"optimizer"``, such as the moments of ``pc.optim.AdamW``) and that of
    ``relaxer`` (``"relaxer"``, such as the velocity of ``pc.relax.HeavyBall``).
    The ``h`` it sent last is in the state of its port ``next``. To watch the
    node while it runs, read them there: ``Run.state_dict()`` pauses the run,
    whereas reading ``module``, ``r`` or ``lam`` from the host's thread races
    with the node's updates in place. ``act_spec`` is the spec of what the node
    sends, for the next node.

    Example::

        B = 64
        tau = pc.TimeConstants(h=4e-3, lam=8e-3, r=4e-3)
        inp = pc.Input(
            x=TensorSpec((B, 784)),
            y=TensorSpec((B,), torch.int64),
            period=tau.period,
            batch=B,
        )
        hidden = pc.Layer(
            nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
            act_spec=inp.act_spec,
            tau=tau,
            optimizer=pc.optim.SGD(tau=1.28),
            learn_interval=24e-3,
        )
        inp.next.connect(hidden.prev)
        hidden.next.connect(out.prev)  # out: a pc.Output, or another pc.Layer

    Args:
        module: The function ``f``, ``(*batch, *in) -> (*batch, *out)``, on
            ``device``.
        act_spec: The spec of the activity ``h_in`` it receives: the
            ``act_spec`` of the node before, which also gives the batch.
        tau: The time constants of ``r`` and ``lam``.
        optimizer: How ``theta`` moves (``pc.optim.SGD``, ``pc.optim.AdamW`` or an
            ``Optimizer`` of your own). The node binds it to its parameters
            that require gradients and holds the bound optimizer as
            ``optimizer``.
        relaxer: How ``r`` moves along the force on it (``pc.relax.Euler``, the
            default, ``pc.relax.HeavyBall`` or a ``Relaxer`` of your own). The node
            binds it to its ``r`` and holds the bound relaxer as ``relaxer``.
        learn_interval: The time in seconds from the write of an input to the
            update of ``theta``, and between the updates while the input stays;
            longer than ``2 * hops * period`` is recommended. ``None`` updates
            in every ``on_update``, for a stream whose input changes at every
            frame (good values for it have not been measured yet);
            ``math.inf`` never updates.
        period: The period of the node in seconds; ``None`` is ``tau.period``
            (``tau.h / 4``). It must not be longer than ``max_dt``.
        max_dt: The longest ``dt`` an update counts, in seconds; ``None`` is
            ``tau.max_dt``.
        device: Where the node computes.

    Raises:
        ValueError: The output of ``module`` does not start with the batch of
            ``act_spec``, or ``LearnClock`` raises: ``learn_interval``
            is neither ``None`` nor > 0, ``max_dt`` is not > 0, or ``period``
            is longer than ``max_dt``. ``module`` raises as PyTorch does when it
            does not take the ``h`` of ``act_spec`` on ``device``.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        act_spec: DictSpec[Activity],
        tau: TimeConstants,
        optimizer: Optimizer,
        relaxer: Relaxer = Euler(),
        learn_interval: float | None,
        period: float | None = None,
        max_dt: float | None = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period if period is None else period)
        self._tau = tau
        batch = act_spec.field("mask", TensorSpec).shape
        self._samples = math.prod(batch)  # B
        self.clock = LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval, max_dt=max_dt
        )
        self.module = module
        # Frozen parameters do not learn.
        self._params = [p for p in module.parameters() if p.requires_grad]
        self.optimizer = optimizer.bind(self._params)
        h_in = act_spec.field("h", TensorSpec)
        with torch.no_grad():
            mu = module(h_in.zeros(device=self.device))
        self.r = torch.zeros_like(mu)
        self.relaxer = relaxer.bind(self.r)
        self.lam = torch.zeros_like(mu)
        out = TensorSpec(mu.shape, mu.dtype)
        self.act_spec = activity_spec(out, act_spec.field("y", TensorSpec), batch)
        self.prev = Port(send=h_in, recv=act_spec)
        self.next = MultiPort(send=self.act_spec, recv=out)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        # Nothing to learn from an input without a target: its time is dropped.
        learn = tick.learn and bool(act["mask"].any())
        mu, g_theta = self._predict(act, learn)
        if tick.new:
            # What the relaxer remembers are the forces of the sample before.
            self.relaxer.reset_()
        g = sum(self.next.recv().values(), torch.zeros_like(self.r))
        relax_(self.r, self.lam, g, tick.dt, self._tau, self.relaxer)
        self._send(act, mu)
        if learn:
            # theta moves by the mean gradient of the batch, g_theta / B.
            self.optimizer.step_([g / self._samples for g in g_theta], tick.elapsed)

    def _predict(
        self, act: Activity, learn: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute ``mu`` and, to ``learn``, ``g_theta`` from the ``delta`` before
        this update; send ``g_in``."""
        # A sample without a target sends back no credit and adds nothing to
        # g_theta: its delta may be left over from an earlier input.
        delta = torch.where(rows(act["mask"], self.r), self.r + self.lam, 0.0)
        mu, g_in, g_theta = vjp(
            self.module, act["h"], lambda _: delta, self._params if learn else ()
        )
        self.prev.send(g_in)
        return mu, g_theta

    def _send(self, act: Activity, mu: torch.Tensor) -> None:
        self.next.send(
            {
                "h": mu + self.r,
                "y": act["y"],
                "mask": act["mask"],
                "time": act["time"],
            }
        )
