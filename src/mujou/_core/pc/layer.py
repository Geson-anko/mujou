"""The hidden node of ``mujou.pc``."""

import math
from collections.abc import Sequence
from typing import override

import torch
from torch import nn

from mujou._core.pc.carry import Carried
from mujou._core.pc.functional import learn_, relax_, vjp
from mujou._core.pc.learner import Learner
from mujou._core.pc.time_constants import TimeConstants
from mujou._core.port import NextPort, PrevPort
from mujou._core.spec import DictSpec, TensorSpec


class Layer(Learner):
    """A node that wraps ``module`` as one constraint ``h = f(h_in; theta)`` of
    PC-ALM.

    It predicts ``mu = module(h_in)`` from the activity ``h_in`` of the previous
    node and holds the deviation ``u = h - mu`` and the dual ``lam`` of its own
    activity ``h = mu + u``, which it sends to the next nodes. With
    ``delta = u + lam`` and the sum ``g`` of what the next nodes send back:

    .. math::

        \\tau_h \\dot u = -(u + \\lambda + g),\\qquad
        \\tau_\\lambda \\dot\\lambda = u,\\qquad
        \\tau_\\theta \\dot\\theta = -g_\\theta / B,

    where ``g_theta = -J_theta^T delta`` and the node sends
    ``g_in = -J_h^T delta`` to the previous node (both from one vector-Jacobian
    product of ``module``, ``functional.vjp``). At a fixed point, ``u = 0`` (``h``
    is the forward pass) and ``lam`` is the credit ``-d loss / d h`` of
    backpropagation. Inside the node, ``module`` is differentiated by autograd;
    between nodes, only these messages flow.

    Each ``on_update(dt)``:

    1. reads ``h_in`` with the data carried with it, and ``g``;
    2. restarts the learning phase if the carried ``time`` changed (a new input;
       ``u`` and ``lam`` are kept);
    3. computes ``mu``, ``g_in`` and, when it learns, ``g_theta`` with the
       ``delta`` before this update;
    4. sends ``g_in``;
    5. moves ``u`` and then ``lam`` by ``dt`` (``functional.relax_``, in Euler
       steps of at most ``max_step``);
    6. sends ``h = mu + u``, carrying on the data that came with ``h_in``;
    7. when it learns, moves ``theta`` by the time since its last update
       (``functional.learn_``).

    It learns once ``learn_interval`` seconds have passed since the input changed,
    and then every ``learn_interval`` seconds while it stays; with
    ``learn_interval=None``, in every update. It does not learn when no sample
    of the input has a target (every carried ``mask`` is False): that time is
    dropped from the next update, which would otherwise learn from the ``delta``
    left over from an earlier input. Until the first sample reaches the node,
    the carried data is zeros (``mask`` all False, ``time`` 0), so it does not
    learn then; a first sample written at time 0 does not restart the phase,
    which runs from the node's first update.

    ``module`` must treat the samples of the batch independently (no
    ``BatchNorm`` in training mode, no ``Dropout``): the gradient of each sample
    must be its own. Its parameters that require gradients learn; set
    ``requires_grad`` to False to freeze one. It is called once on zeros when the
    node is made, to find the shape and dtype of ``h``.

    The state (``Run.state_dict()["nodes"][path]``) holds ``"module"`` (its
    ``state_dict``), ``"u"`` and ``"lam"`` (``(B, *out)``), and the learning
    phase (``"_seen_time"``, ``"_phase"``, ``"_elapsed"``). Read ``module``,
    ``u`` and ``lam`` to watch the node; do not assign them from outside it.

    Example::

        B = 64
        tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=1.28)
        carry = pc.carry_spec(TensorSpec((B,), torch.int64), batch=B)
        hidden = pc.Layer(
            nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
            input_shape=(B, 784),
            carry=carry,
            tau=tau,
            learn_interval=24e-3,
        )
        inp.next.connect(hidden.prev)  # inp: a pc.Input
        hidden.next.connect(out.prev)  # out: a pc.Output, or another pc.Layer

    Args:
        module: The function ``f``, ``(B, *in) -> (B, *out)``, on ``device``.
        input_shape: The shape ``(B, *in)`` of ``h_in``.
        carry: The spec of the carried data, from ``carry_spec``.
        tau: The time constants.
        learn_interval: The time in seconds from a new input to the update of
            ``theta``, and between the updates while the input stays. ``None``
            updates in every ``on_update``.
        period: The period of the node in seconds; ``None`` is ``tau.h / 4``.
        max_step: The longest Euler step of ``u`` and ``lam`` in seconds;
            ``None`` is ``tau.h / 2``.
        dtype: The dtype of ``h_in``; ``None`` is ``torch.get_default_dtype()``.
        device: Where the node computes.

    Raises:
        TypeError: ``tau`` is not a ``TimeConstants``, ``carry`` is not a spec
            of ``carry_spec``, or ``module`` does not return a tensor.
        ValueError: ``input_shape[0]`` or the first dimension of the output of
            ``module`` is not the batch of ``carry``; a parameter or buffer of
            ``module`` is not on ``device``; ``learn_interval`` is neither
            ``None`` nor finite and positive; or ``max_step`` is not finite and
            positive. ``module`` raises as PyTorch does when it does not take
            ``input_shape`` and ``dtype``.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        input_shape: Sequence[int],
        carry: DictSpec[Carried],
        tau: TimeConstants,
        learn_interval: float | None = None,
        period: float | None = None,
        max_step: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(
            module,
            input_shape=input_shape,
            carry=carry,
            tau=tau,
            learn_interval=learn_interval,
            period=period,
            dtype=dtype,
            device=device,
        )
        if max_step is not None and not (math.isfinite(max_step) and max_step > 0):
            raise ValueError(
                f"max_step must be None or finite and > 0, got {max_step!r}"
            )
        self._max_step = max_step
        h = self._probe()
        self.u = torch.zeros_like(h)
        self.lam = torch.zeros_like(h)
        self.prev = PrevPort(send=self._input, recv=self._input, recv_carry=carry)
        out = TensorSpec(h.shape, h.dtype)
        self.next = NextPort(send=out, recv=out, send_carry=self.prev)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        h_in = self.prev.recv()  # (B, *in)
        g = torch.zeros_like(self.u)  # (B, *out): the sum over the next nodes
        for g_next in self.next.recv().values():
            g += g_next
        learn = self._tick(dt, int(self.prev.carried("time")))
        if learn and not self.prev.carried("mask").any():
            self._learned()  # nothing to learn from this input: drop the time
            learn = False
        mu, g_in, g_theta = vjp(
            self.module, h_in, self.u + self.lam, self._params if learn else ()
        )
        assert g_in is not None  # wrt_input is True
        self.prev.send(g_in)
        relax_(self.u, self.lam, g, dt, self._tau, max_step=self._max_step)
        self.next.send(mu + self.u)  # h = mu + u, carrying what came with h_in
        if learn:
            learn_(self._params, g_theta, self._elapsed, self._tau, self._batch)
            self._learned()
