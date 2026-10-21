"""The hidden node of ``mujou.pc``."""

from typing import override

import torch
from torch import nn

from mujou import DictSpec, MultiPort, Node, Port, TensorSpec
from mujou.pc.clock import LearnClock
from mujou.pc.functional import learn_, relax_, vjp
from mujou.pc.messages import Activity, activity_spec, rows
from mujou.pc.time_constants import TimeConstants
from mujou.types import Device


class Layer(Node):
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

    where $B$ is the number of samples in the batch (1 without one),
    $g_\theta = -J_\theta^\top \delta$ and the node sends
    $g_{\mathrm{in}} = -J_h^\top \delta$ to the previous node (both from one
    vector-Jacobian product of ``module``, ``functional.vjp``). At a fixed point,
    $r = 0$ (``h`` is the forward pass) and ``lam`` is the credit
    $-\partial\ell/\partial h$ of backpropagation. Inside the node, ``module``
    is differentiated by autograd; between nodes, only these messages flow.

    ``h`` follows the gradient flow above, with the ``mu`` of the moment: when
    the nodes below relax, ``h_in`` and ``mu`` move continuously and ``h`` only
    relaxes towards them. Where ``mu`` jumps, ``h`` moves with it once, so that
    the deviation ``r = h - mu`` and ``lam`` are kept, and only they are:

    - With a new sample (the ``time`` of the activity changed), or when the
      ``h_in`` it reads jumped (its ``jumps`` changed), the node moves
      ``h <- mu + (h - mu_last)`` with the ``mu_last`` of its previous update.
      Without a target, ``h`` is thus the forward pass.
    - When it updates ``theta``, it computes once more, without gradients,
      ``mu' = f(h_in; theta_new)`` at the same ``h_in``, and moves
      ``h <- h + (mu' - mu)`` at the start of its next update, with
      ``mu_last <- mu'``. Only the jump that ``theta`` makes moves ``h``, not
      the drift of ``h_in``. The move waits one update so that the next nodes,
      which may read ``h`` before they learn, see it as a jump (it is sent
      with new ``jumps``) rather than learn from it.

    The node advances its ``jumps`` once in an update that moves ``h`` with a
    new sample or a jump of ``h_in``, or by a jump of ``theta`` while
    ``learn_interval`` is set; the next node then moves with it in turn, so a
    jump runs up the path with ``r`` unchanged at every node, whenever each
    node learns. With ``learn_interval=None``, ``theta`` moves by ``dt`` in
    every update, continuously: the node moves its own ``h`` but does not
    advance its ``jumps``, as moving every node above in every update would
    hold ``r`` rather than ``h`` as the state, which diverges in deep chains.

    Each ``on_update(dt)``:

    1. reads the activity and ticks its ``clock``, which caps ``dt`` at
       ``max_dt``;
    2. moves ``h`` by the jump of ``theta`` of the previous update, if any;
    3. computes ``mu``, ``g_in`` and, when it learns, ``g_theta`` with the
       ``delta`` before this update, zero for the samples whose ``mask`` is
       False, and sends ``g_in``;
    4. with a new input or new ``jumps`` of ``h_in``, moves ``h`` with ``mu``;
       then moves ``h`` and then ``lam`` by ``dt`` with the ``g`` the next nodes
       sent (``functional.relax_``);
    5. sends ``h`` with its ``jumps``, and the ``y``, ``mask`` and ``time``
       that came with ``h_in``;
    6. when it learns, moves ``theta`` by the time since its last update
       (``functional.learn_``) and computes the jump of ``mu`` it made.

    Its ``clock`` (a ``LearnClock``) tells when it learns: in its first update
    once ``learn_interval`` seconds have passed since the host wrote the input
    (its ``time``, on the run's time), and then every ``learn_interval``
    seconds while the input stays, but never in the update that first sees an
    input (its ``delta`` is still that of the input before); with
    ``learn_interval=None``, in every update; with ``tau.theta`` infinite,
    never (``LearnClock.of``). The nodes read the one timer of the run, so the
    nodes on a path learn at about the same time, however deep they are, and
    each from the relaxed ``delta`` of the same input. A ``learn_interval``
    longer than the round trip of the credit down the path and back
    (``2 * hops * period``, with ``hops`` the number of connections from the
    ``Input`` to the ``Output``) is recommended: with intervals of a few
    periods, every node learns, and so advances its ``jumps``, in almost every
    update, the nodes above move ``h`` with every change of ``mu``, and deep
    chains have been seen to diverge however slowly ``theta`` moves. The
    node does not learn when no sample of the input has a target (every
    ``mask`` is False): that time is dropped from the next update, which would
    otherwise learn from the ``delta`` left over from an earlier input. Nor
    does it keep the time of an input that changed before the node learned
    from it. Until the first sample reaches the node, the activity is zeros
    (``mask`` all False, ``time`` 0), so it does not learn then.

    ``dt`` is capped because the nodes are coupled explicitly: ``g`` was made
    from ``h`` as it was when the next node last updated. After a long stall
    (a loaded machine), the step ``dt / tau.h`` of every node grows at once and
    the coupling makes the network diverge, which no split of the step inside
    the node prevents. The time beyond ``max_dt`` is dropped, from the
    relaxation and from the weight of the next update of ``theta`` alike. The
    default ``tau.max_dt``, ``min(tau.h / 2, 3 * tau.lam)``, keeps the step of
    ``h`` and ``lam`` stable in the node itself.

    ``module`` must treat the samples of the batch independently (no
    ``BatchNorm`` in training mode): the gradient of each sample must be its
    own. It must also be deterministic and keep no state of its own (no
    ``Dropout``, no ``BatchNorm`` in training mode): an update that learns calls
    it once more on the same ``h_in`` to find the jump of ``theta``, which would
    otherwise add noise, or update running statistics a second time. Its
    parameters that require gradients learn; set ``requires_grad`` to False to
    freeze one. It is called once on zeros when the node is made, to find the
    shape and dtype of ``h``.

    The state (``Run.state_dict()["nodes"][path]["state"]``) holds
    ``"generator"`` (the state of the node's ``generator``), ``"module"`` (its
    ``state_dict``), ``"h"`` and ``"lam"`` (``(*batch, *out)``), the prediction of
    the last update (``"mu"``), the jump of ``theta`` still to add to ``h``
    (``"jump"``, and ``"jumped"`` whether there is one), the ``jumps`` it sends
    and the last ones it read (``"jumps"``, ``"seen_jumps"``) and when to
    learn (``"clock"``, the state of the ``LearnClock``). To watch the node
    while it runs, read them there: ``Run.state_dict()`` pauses the run,
    whereas reading ``module``, ``h`` or ``lam`` from the host's thread races
    with the node's updates in place. ``act_spec`` is the spec of what the node
    sends, for the next node.

    Example::

        B = 64
        tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=1.28)
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
            learn_interval=24e-3,
        )
        inp.next.connect(hidden.prev)
        hidden.next.connect(out.prev)  # out: a pc.Output, or another pc.Layer

    Args:
        module: The function ``f``, ``(*batch, *in) -> (*batch, *out)``, on
            ``device``.
        act_spec: The spec of the activity ``h_in`` it receives: the
            ``act_spec`` of the node before, which also gives the batch.
        tau: The time constants.
        learn_interval: The time in seconds from the write of an input to the
            update of ``theta``, and between the updates while the input stays;
            longer than ``2 * hops * period`` is recommended. ``None`` updates
            in every ``on_update``, for a stream whose input changes at every
            frame (good values for it have not been measured yet);
            ``math.inf`` never updates.
        period: The period of the node in seconds; ``None`` is ``tau.period``
            (``tau.h / 4``). It must not be longer than ``max_dt``.
        max_dt: The longest ``dt`` an update counts, in seconds; ``None`` is
            ``tau.max_dt`` (``min(tau.h / 2, 3 * tau.lam)``).
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
        learn_interval: float | None,
        period: float | None = None,
        max_dt: float | None = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=tau.period if period is None else period)
        self._tau = tau
        self._batch = act_spec.specs["mask"].shape
        self.clock = LearnClock.of(
            tau, period=self.period, learn_interval=learn_interval, max_dt=max_dt
        )
        self.module = module
        # Frozen parameters do not learn.
        self._params = [p for p in module.parameters() if p.requires_grad]
        h_in = act_spec.specs["h"]
        with torch.no_grad():
            mu = module(h_in.zeros(device=self.device))
        self.h = torch.zeros_like(mu)
        self.lam = torch.zeros_like(mu)
        # The prediction of the last update, from which h moves with a new mu.
        # At zero with h, the first update moves h to the forward pass.
        self.mu = torch.zeros_like(mu)
        # The jump of mu that the last update of theta made, for h to take in
        # the next update; zeros once taken.
        self.jump = torch.zeros_like(mu)
        self.jumped = False
        self.jumps = 0  # the jumps of the h this node sends
        self.seen_jumps = 0  # those of the h_in read last; the zeros before any
        for name in ("jumped", "jumps", "seen_jumps"):
            self.register_state(name)
        out = TensorSpec(mu.shape, mu.dtype)
        self.act_spec = activity_spec(out, act_spec.specs["y"], self._batch)
        self.prev = Port(send=h_in, recv=act_spec)
        self.next = MultiPort(send=self.act_spec, recv=out)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        act = self.prev.recv()
        tick = self.clock.tick(dt, int(act["time"]), self.timer.now().ns)
        # Nothing to learn from an input without a target: its time is dropped.
        learn = tick.learn and bool(act["mask"].any())
        jumped = self._take_jump()
        # A new sample, or h_in jumped: h moves with mu, once.
        seen_jumps = int(act["jumps"])
        reanchor = tick.new or seen_jumps != self.seen_jumps
        self.seen_jumps = seen_jumps
        mu, g_theta = self._predict(act, reanchor, learn)
        self._relax(mu, reanchor, tick.dt)
        # The jump of theta is discrete only when it learns at intervals.
        if reanchor or (jumped and self.clock.discrete):
            self.jumps += 1
        self._send(act)
        if learn:
            self._learn(act["h"], mu, g_theta, tick.elapsed)

    def _take_jump(self) -> bool:
        """Add the jump of ``theta`` of the last update to ``h``, and return
        whether there was one."""
        jumped, self.jumped = self.jumped, False
        self.h.add_(self.jump)
        self.jump.zero_()
        return jumped

    def _predict(
        self, act: Activity, reanchor: bool, learn: bool
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute ``mu`` and, to ``learn``, ``g_theta`` from the ``delta`` before
        this update; send ``g_in``."""
        h, lam, mu_last = self.h, self.lam, self.mu
        # A sample without a target sends back no credit and adds nothing to
        # g_theta: its delta may be left over from an earlier input.
        labeled = rows(act["mask"], h)

        def delta(mu: torch.Tensor) -> torch.Tensor:
            # When h moves with mu (in _relax), h - mu is h - mu_last.
            r = h - (mu_last if reanchor else mu)
            return torch.where(labeled, r + lam, 0.0)

        mu, g_in, g_theta = vjp(
            self.module, act["h"], delta, self._params if learn else ()
        )
        self.prev.send(g_in)
        return mu, g_theta

    def _relax(self, mu: torch.Tensor, reanchor: bool, dt: float) -> None:
        """Move ``h`` with ``mu`` if ``reanchor``, then ``h`` and ``lam`` by
        ``dt``."""
        if reanchor:
            self.h.sub_(self.mu).add_(mu)  # h <- mu + (h - mu_last)
        g = sum(self.next.recv().values(), torch.zeros_like(self.h))
        relax_(self.h, self.lam, mu, g, dt, self._tau)
        self.mu.copy_(mu)

    def _send(self, act: Activity) -> None:
        jumps = torch.tensor(self.jumps, dtype=torch.int64, device=self.device)
        self.next.send(
            {
                "h": self.h,
                "jumps": jumps,
                "y": act["y"],
                "mask": act["mask"],
                "time": act["time"],
            }
        )

    def _learn(
        self,
        h_in: torch.Tensor,
        mu: torch.Tensor,
        g_theta: list[torch.Tensor],
        elapsed: float,
    ) -> None:
        """Move ``theta``, and keep the jump of ``mu`` it made for ``h`` to take
        in the next update."""
        learn_(self._params, g_theta, elapsed, self._tau, self._batch)
        with torch.no_grad():
            moved = self.module(h_in)  # mu' = f(h_in; theta_new)
        torch.sub(moved, mu, out=self.jump)
        self.mu.copy_(moved)
        self.jumped = True
