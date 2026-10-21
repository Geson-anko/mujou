"""The output node of ``mujou.pc``."""

from collections.abc import Callable
from typing import override

import torch
from torch import nn

from mujou._core.pc.functional import learn_, loss_vjp
from mujou._core.pc.learner import Learner
from mujou._core.pc.messages import Activity, Prediction
from mujou._core.pc.time_constants import TimeConstants
from mujou._core.port import NextPort, PrevPort
from mujou._core.spec import DictSpec, TensorSpec


class Output(Learner):
    r"""A node that predicts ``y_hat = module(h_in)`` and whose energy is the loss
    of the samples that have a target.

    It pairs ``h_in`` with the target ``y`` and ``mask`` of the same
    ``Activity``, so the prediction always meets the target of the very input it
    was computed from, however fast the host writes. With

    $$
    \ell = \sum_b \mathrm{mask}_b\, \ell_b(\hat y_b, y_b),\qquad
    \tau_\theta \dot\theta = -\frac{1}{B}\frac{\partial\ell}{\partial\theta},
    $$

    it sends $g_{\mathrm{in}} = \partial\ell/\partial h_{\mathrm{in}}$ to the
    previous node (``functional.loss_vjp``), zeros when no sample has a target.
    Each ``on_update(dt)`` caps ``dt`` at ``max_dt`` as ``Layer`` does, reads
    the ``Activity``, sends ``g_in``, sends a ``Prediction``
    (``y_hat`` with the ``y``, ``mask`` and ``time`` it was paired with) for
    the host to read, and updates ``theta`` when it learns. It does not read the
    ``epoch``: a jump of ``h_in`` needs no move here, as the node holds no
    activity of its own.

    It learns as ``Layer`` does: once ``learn_interval`` seconds have passed
    since the host wrote the input, but not in the update that first sees it,
    or in every update with ``None``. When no sample has a target, it does not
    learn and drops the time since the last update; so too before the first
    sample arrives, while the activity is zeros (``mask`` all False), and
    when the input changes before it has learned from it. ``B`` is the batch,
    not the number of samples with a target: a sample without one adds zero.

    ``loss`` returns the loss of each sample, ``(B,)``, such as ``mse`` or
    ``cross_entropy``, and the node sums them. A loss that averages (such as
    ``nn.MSELoss()``) would scale the gradient that flows back by ``1 / B``.
    ``module`` must treat the samples independently, and is called once on zeros
    when the node is made, as in ``Layer``.

    The state holds ``"module"`` and when to learn (``"_seen_time"``,
    ``"_learn_at"``, ``"_elapsed"``, ``"_learned_input"``). Read it with
    ``Run.state_dict()``, as for ``Layer``.

    Example::

        out = pc.Output(
            nn.Linear(256, 10),
            act=hidden.act,  # hidden: a pc.Layer
            loss=pc.cross_entropy,
            tau=tau,
            learn_interval=24e-3,
        )
        hidden.next.connect(out.prev)
        # The host: prediction = run.read(out.next); prediction["y_hat"], ...

    Args:
        module: The function ``f``, ``(B, *in) -> (B, *out)``, on ``device``.
        act: The spec of the activity ``h_in`` it receives: the ``act`` of the
            node before.
        loss: The loss of each sample, ``(y_hat, y) -> (B,)``.
        tau: The time constants; ``tau.theta`` is used, and ``tau.h`` gives the
            default period.
        learn_interval: As in ``Layer``.
        period: The period of the node in seconds; ``None`` is ``tau.h / 4``.
            It must not be longer than ``max_dt``.
        max_dt: As in ``Layer``: the longest ``dt`` an update counts, here
            towards the weight of the update of ``theta``.
        device: Where the node computes.

    Raises:
        TypeError: As ``Layer`` raises.
        ValueError: As ``Layer`` raises, or ``loss`` does
            not return ``(B,)`` for the output of ``module`` and the target.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        act: DictSpec[Activity],
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tau: TimeConstants,
        learn_interval: float | None = None,
        period: float | None = None,
        max_dt: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            module,
            act=act,
            tau=tau,
            learn_interval=learn_interval,
            period=period,
            max_dt=max_dt,
            device=device,
        )
        y_hat = self._probe()
        target = act.specs["y"]
        with torch.no_grad():
            ell = loss(y_hat, target.zeros(device=self.device))
        if ell.shape != (self._batch,):
            raise ValueError(
                f"loss must return the loss of each sample, of shape ({self._batch},), "
                f"got {tuple(ell.shape)}; a loss that averages, such as "
                "nn.MSELoss(), does not (use pc.mse or pc.cross_entropy)"
            )
        self._loss = loss
        self.prev = PrevPort(send=self._input, recv=act)
        self.next = NextPort(
            send=DictSpec(
                Prediction,
                y_hat=TensorSpec(y_hat.shape, y_hat.dtype),
                y=target,
                mask=act.specs["mask"],
                time=act.specs["time"],
            )
        )

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        dt = self._cap(dt)
        activity = self.prev.recv()
        h_in = activity["h"]  # (B, *in)
        _, learn = self._tick(dt, int(activity["time"]))
        labeled = bool(activity["mask"].any())
        g_theta: list[torch.Tensor] = []
        if labeled:
            y_hat, g_in, g_theta = loss_vjp(
                self.module,
                h_in,
                activity["y"],
                activity["mask"],
                self._loss,
                self._params if learn else (),
            )
            assert g_in is not None  # wrt_input is True
        else:
            with torch.no_grad():
                y_hat = self.module(h_in)
            g_in = torch.zeros_like(h_in)  # no target, no loss
        self.prev.send(g_in)
        self.next.send(
            {
                "y_hat": y_hat,
                "y": activity["y"],
                "mask": activity["mask"],
                "time": activity["time"],
            }
        )
        if learn:
            if labeled:
                learn_(self._params, g_theta, self._elapsed, self._tau, self._batch)
            self._learned()
