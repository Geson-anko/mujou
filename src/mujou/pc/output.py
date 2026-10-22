"""The output node of ``mujou.pc``."""

from collections.abc import Callable
from typing import override

import torch
from torch import nn

from mujou import DictSpec, Node, Port, TensorSpec, TimestampSpec
from mujou.pc.clock import LearnClock
from mujou.pc.functional import learn_, loss_vjp
from mujou.pc.messages import Activity, Prediction
from mujou.pc.time_constants import TimeConstants
from mujou.types import Device


class Output(Node):
    r"""A node that predicts ``y_hat = module(h_in)`` and whose energy is the loss
    of the samples that have a target.

    It pairs ``h_in`` with the target ``y`` and ``mask`` of the same
    ``Activity``, so the prediction always meets the target of the very input it
    was computed from, however fast the host writes. With

    $$
    \ell = \sum_b \mathrm{mask}_b\, \ell_b(\hat y_b, y_b),\qquad
    \tau_\theta \dot\theta = -\frac{1}{B}\frac{\partial\ell}{\partial\theta},
    $$

    where $B$ is the number of samples in the batch (1 without one), it sends
    $g_{\mathrm{in}} = \partial\ell/\partial h_{\mathrm{in}}$ to the
    previous node (``functional.loss_vjp``): zero for a sample without a target
    while its ``y_hat`` is finite. Each ``on_update(dt)`` caps ``dt`` at
    ``max_dt`` as ``Layer`` does, reads the ``Activity``, sends ``g_in``, sends a
    ``Prediction`` (``y_hat`` with the ``y``, ``mask`` and ``time`` it was
    paired with) for the host to read, and updates ``theta`` when it learns. It
    does not read the ``jumps``: a jump of ``h_in`` needs no move here, as the
    node holds no activity of its own.

    It learns as ``Layer`` does, by its ``clock``: once ``learn_interval``
    seconds have passed since the host wrote the input, but not in the update
    that first sees it, or in every update with ``None``. When no sample has a
    target, it does not learn and drops the time since the last update; so too
    before the first sample arrives, while the activity is zeros (``mask`` all
    False), and when the input changes before it has learned from it. ``B`` is
    the number of samples in the batch, not the number of samples with a
    target: a sample without one adds zero.

    ``loss`` returns the loss of each sample, of the shape of the batch (that
    of ``mask``), such as ``MSELoss(batch)`` or ``CrossEntropyLoss(batch)``,
    and the node sums them. A loss that averages over the batch (such as
    ``nn.MSELoss()``) would scale the gradient that flows back by ``1 / B``, so
    the node refuses it when it is made: it calls ``module`` and ``loss`` once
    on zeros then (``functional.loss_vjp``), which also finds the shape of
    ``y_hat``. ``module`` must treat the samples independently, as in
    ``Layer``.

    The state holds ``"generator"``, ``"module"`` and when to learn
    (``"clock"``). Read it with ``Run.state_dict()``, as for ``Layer``.

    Example::

        out = pc.Output(
            nn.Linear(256, 10),
            act_spec=hidden.act_spec,  # hidden: a pc.Layer
            loss=pc.CrossEntropyLoss(B),
            tau=tau,
            learn_interval=24e-3,
        )
        hidden.next.connect(out.prev)
        # The host: prediction = run.read(out.next); prediction["y_hat"], ...

    Args:
        module: The function ``f``, ``(*batch, *in) -> (*batch, *out)``, on
            ``device``.
        act_spec: The spec of the activity ``h_in`` it receives: the
            ``act_spec`` of the node before, which also gives the batch.
        loss: The loss of each sample, ``(y_hat, y) -> batch``.
        tau: The time constants; ``tau.theta`` is used, and the defaults of
            ``period`` and ``max_dt``.
        learn_interval: As in ``Layer``.
        period: The period of the node in seconds; ``None`` is ``tau.period``
            (``tau.h / 4``). It must not be longer than ``max_dt``.
        max_dt: As in ``Layer``: the longest ``dt`` an update counts, here
            towards the weight of the update of ``theta``. The default
            ``tau.max_dt`` is that of ``Layer``, so that every node of a chain
            counts the same time.
        device: Where the node computes.

    Raises:
        ValueError: ``LearnClock`` raises, as in ``Layer``, or ``loss`` does
            not return the shape of the batch for the output of ``module`` and
            the target.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        act_spec: DictSpec[Activity],
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tau: TimeConstants,
        learn_interval: float | None,
        period: float | None = None,
        max_dt: float | None = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period if period is None else period)
        self._tau = tau
        self._loss = loss
        self._batch = act_spec.field("mask", TensorSpec).shape
        self.clock = LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval, max_dt=max_dt
        )
        self.module = module
        # Frozen parameters do not learn.
        self._params = [p for p in module.parameters() if p.requires_grad]
        h_in = act_spec.field("h", TensorSpec)
        y = act_spec.field("y", TensorSpec)
        mask = act_spec.field("mask", TensorSpec)
        y_hat, _, _ = loss_vjp(
            module,
            h_in.zeros(device=self.device),
            y.zeros(device=self.device),
            mask.zeros(device=self.device),
            loss,
        )
        self.prev = Port(send=h_in, recv=act_spec)
        self.next = Port(
            send=DictSpec(
                Prediction,
                y_hat=TensorSpec(y_hat.shape, y_hat.dtype),
                y=y,
                mask=mask,
                time=act_spec.field("time", TimestampSpec),
            )
        )

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, act["time"], self.timer.now())
        # Nothing to learn from an input without a target: its time is dropped.
        learn = tick.learn and bool(act["mask"].any())
        y_hat, g_in, g_theta = loss_vjp(
            self.module,
            act["h"],
            act["y"],
            act["mask"],
            self._loss,
            self._params if learn else (),
        )
        self.prev.send(g_in)
        self.next.send(
            {"y_hat": y_hat, "y": act["y"], "mask": act["mask"], "time": act["time"]}
        )
        if learn:
            learn_(self._params, g_theta, tick.elapsed, self._tau, self._batch)
