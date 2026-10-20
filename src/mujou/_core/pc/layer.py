"""The hidden node of ``mujou.pc``."""

from collections.abc import Sequence
from typing import override

import torch
from torch import nn

from mujou._core.pc.carry import Carried
from mujou._core.pc.functional import learn_, relax_, rows, vjp
from mujou._core.pc.learner import Learner
from mujou._core.pc.time_constants import TimeConstants
from mujou._core.port import NextPort, PrevPort
from mujou._core.spec import DictSpec, TensorSpec


class Layer(Learner):
    r"""A node that wraps ``module`` as one constraint ``h = f(h_in; theta)`` of
    PC-ALM.

    It predicts ``mu = module(h_in)`` from the activity ``h_in`` of the previous
    node, and holds its own activity ``h``, which it sends to the next nodes, and
    the dual ``lam`` of the constraint. With $r = h - \mu$,
    $\delta = r + \lambda$ and the sum $g$ of what the next nodes send back:

    $$
    \tau_h \dot h = -(r + \lambda + g),\qquad
    \tau_\lambda \dot\lambda = r,\qquad
    \tau_\theta \dot\theta = -g_\theta / B,
    $$

    where $g_\theta = -J_\theta^\top \delta$ and the node sends
    $g_{\mathrm{in}} = -J_h^\top \delta$ to the previous node (both from one
    vector-Jacobian product of ``module``, ``functional.vjp``). At a fixed point,
    $r = 0$ (``h`` is the forward pass) and ``lam`` is the credit
    $-\partial\ell/\partial h$ of backpropagation. Inside the node, ``module``
    is differentiated by autograd; between nodes, only these messages flow.

    When the input changes, and in the update after the node has moved
    ``theta``, ``mu`` jumps: the node then moves ``h`` with it once,
    ``h <- mu + (h - mu_last)`` with the ``mu_last`` of its previous update, so
    that only the deviation from the prediction carries over to the new input.
    Without a target, ``h`` is thus the forward pass.

    Each ``on_update(dt)``:

    1. caps ``dt`` at ``max_dt``;
    2. reads ``h_in`` with the data carried with it, and ``g``;
    3. notes a new input if the carried ``time`` changed;
    4. computes ``mu``, ``g_in`` and, when it learns, ``g_theta`` with the
       ``delta`` before this update, zero for the samples whose carried
       ``mask`` is False; when the input is new or ``theta`` moved in the
       previous update, moves ``h`` with ``mu`` first;
    5. sends ``g_in``;
    6. moves ``h`` and then ``lam`` by ``dt`` (``functional.relax_``);
    7. sends ``h``, carrying on the data that came with ``h_in``;
    8. when it learns, moves ``theta`` by the time since its last update
       (``functional.learn_``).

    It learns in its first update once ``learn_interval`` seconds have passed
    since the host wrote the input (the carried ``time``, on the run's time),
    and then every ``learn_interval`` seconds while the input stays; with
    ``learn_interval=None``, in every update. Timed from the write, the nodes
    on a path learn at about the same time, however deep they are: a node
    that learned later than the one before it would learn from a ``delta``
    shaken by the jump of ``h_in`` that the update of the other's ``theta``
    makes. The node does not learn when no sample of the input has a target
    (every carried ``mask`` is False): that time is dropped from the next
    update, which would otherwise learn from the ``delta`` left over from an
    earlier input. Nor does it keep the time of an input that changed before
    the node learned from it. Until the first sample reaches the node, the
    carried data is zeros (``mask`` all False, ``time`` 0), so it does not
    learn then; a first sample written at time 0 is not told from those zeros
    by its time, so ``h`` does not move with ``mu`` for it (it follows by
    relaxing).

    ``dt`` is capped because the nodes are coupled explicitly: ``g`` was made
    from ``h`` as it was when the next node last updated. After a long stall
    (a loaded machine), the step ``dt / tau.h`` of every node grows at once and
    the coupling makes the network diverge, which no split of the step inside
    the node prevents. The time beyond ``max_dt`` is dropped, from the
    relaxation and from the weight of the next update of ``theta`` alike. The
    default ``min(tau.h / 2, 3 * tau.lam)`` keeps the step of ``h`` and ``lam``
    stable in the node itself.

    ``module`` must treat the samples of the batch independently (no
    ``BatchNorm`` in training mode, no ``Dropout``): the gradient of each sample
    must be its own. Its parameters that require gradients learn; set
    ``requires_grad`` to False to freeze one. It is called once on zeros when the
    node is made, to find the shape and dtype of ``h``.

    The state (``Run.state_dict()["nodes"][path]``) holds ``"module"`` (its
    ``state_dict``), ``"h"`` and ``"lam"`` (``(B, *out)``), the prediction of
    the last update (``"_mu"``), whether ``theta`` moved in it (``"_moved"``)
    and when to learn (``"_seen_time"``, ``"_learn_at"``, ``"_elapsed"``,
    ``"_learned_input"``). To watch the node while it runs, read them there:
    ``Run.state_dict()`` pauses the run, whereas reading ``module``, ``h`` or
    ``lam`` from the host's thread races with the node's updates in place.

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
        learn_interval: The time in seconds from the write of an input to the
            update of ``theta``, and between the updates while the input stays.
            ``None`` updates in every ``on_update``.
        period: The period of the node in seconds; ``None`` is ``tau.h / 4``.
        max_dt: The longest ``dt`` an update counts, in seconds; ``None`` is
            ``min(tau.h / 2, 3 * tau.lam)``.
        dtype: The dtype of ``h_in``; ``None`` is ``torch.get_default_dtype()``.
        device: Where the node computes.

    Raises:
        TypeError: ``tau`` is not a ``TimeConstants``, ``carry`` is not a spec
            of ``carry_spec``, ``module`` does not return a tensor, or
            ``learn_interval`` or ``max_dt`` is a ``bool``.
        ValueError: ``input_shape[0]`` or the first dimension of the output of
            ``module`` is not the batch of ``carry``; a parameter or buffer of
            ``module`` is not on ``device``; or ``learn_interval`` or
            ``max_dt`` is neither ``None`` nor finite and positive. ``module``
            raises as PyTorch does when it does not take ``input_shape`` and
            ``dtype``.
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
        max_dt: float | None = None,
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
            max_dt=max_dt,
            dtype=dtype,
            device=device,
        )
        mu = self._probe()
        self.h = torch.zeros_like(mu)
        self.lam = torch.zeros_like(mu)
        # The prediction of the last update, from which h moves with a new mu.
        # At zero with h, the first update moves h to the forward pass.
        self._mu = torch.zeros_like(mu)
        self._moved = False  # whether theta moved in the last update
        self.register_state("_moved")
        self.prev = PrevPort(send=self._input, recv=self._input, recv_carry=carry)
        out = TensorSpec(mu.shape, mu.dtype)
        self.next = NextPort(send=out, recv=out, send_carry=self.prev)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        dt = self._cap(dt)
        h_in = self.prev.recv()  # (B, *in)
        g = torch.zeros_like(self.h)  # (B, *out): the sum over the next nodes
        for g_next in self.next.recv().values():
            g += g_next
        mask = self.prev.carried("mask")  # (B,)
        new, learn = self._tick(dt, int(self.prev.carried("time")))
        if learn and not mask.any():
            self._learned()  # nothing to learn from this input: drop the time
            learn = False
        # A sample without a target sends back no credit and adds nothing to
        # g_theta: its delta may be left over from an earlier input.
        labeled = rows(mask, self.h)
        reanchor = new or self._moved
        h, lam, mu_last = self.h, self.lam, self._mu

        def delta(mu: torch.Tensor) -> torch.Tensor:
            # When h moves with mu (below), h - mu is h - mu_last.
            r = h - (mu_last if reanchor else mu)
            return torch.where(labeled, r + lam, 0.0)

        mu, g_in, g_theta = vjp(self.module, h_in, delta, self._params if learn else ())
        assert g_in is not None  # wrt_input is True
        self.prev.send(g_in)
        if reanchor:
            h.sub_(mu_last).add_(mu)  # h <- mu + (h - mu_last)
        relax_(h, lam, mu, g, dt, self._tau)
        mu_last.copy_(mu)
        self.next.send(h)  # carrying what came with h_in
        if learn:
            learn_(self._params, g_theta, self._elapsed, self._tau, self._batch)
            self._learned()
        self._moved = learn
