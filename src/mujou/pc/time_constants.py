"""The time constants of PC-ALM in continuous time."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TimeConstants:
    r"""The time constants of the continuous-time PC-ALM, in seconds.

    A node holds the deviation $r = h - \mu$ of its activity ``h`` from its
    prediction ``mu``, and the dual ``lam``, and relaxes them with

    $$
    \tau_h \dot r = -(r + \lambda + g) - \frac{\tau_h}{\tau_r}\,r,\qquad
    \tau_\lambda \dot\lambda = r,
    $$

    and a step of ``dt`` seconds moves them by ``dt / tau`` of the right-hand
    side (the discrete step sizes are ``eta_h = dt / h`` and ``alpha = dt /
    lam``). The parameters ``theta`` move by the optimizer of the node
    (``pc.optim.SGD`` or ``pc.optim.AdamW``), which holds their time constant.

    ``lam=math.inf`` makes the dual stay at zero (``alpha = 0``); with
    ``r=math.inf`` too, the fixed point is that of plain predictive coding (the
    decay would shrink ``r`` by ``1 + h / r`` at each node). ``h`` must be
    finite.

    ``r`` is the time constant of the decay of the deviation. The fixed point
    stays (``lam`` integrates ``r``, so ``r = 0`` there, and ``lam`` is the
    credit of backpropagation), but without the decay a deep chain diverges:
    ``h = mu + r`` moves with every ``r`` below it, and the credit loops back
    through the product of the Jacobians. In a chain of 8 residual nodes,
    ``lam`` diverged with ``r=math.inf`` and with ``r = 4 * h`` it had not
    settled after 1000 ``h``; with ``r = 2 * h`` down to ``h / 3`` it reached
    the credit (``specs/pc.md``). ``r = h`` worked in the chains measured, and
    is the longest for which ``max_dt`` is still the default ``period``;
    ``math.inf`` turns the decay off, for shallow chains. No constant ``r``
    keeps every chain stable: one whose forward pass amplifies (the product of
    its Jacobians far above 1, such as a linear chain of 16 gains of 1.25)
    diverges slowly whatever ``r`` is. An analog integrator that holds ``r``
    and leaks is this decay.

    The fields are keyword-only and have no defaults: all are floats, and a swap
    by position could not be caught by the type checker.

    Example::

        tau = TimeConstants(h=4e-3, lam=8e-3, r=4e-3)
        pc_only = TimeConstants(h=4e-3, lam=math.inf, r=math.inf)

    Raises:
        ValueError: ``h`` is not finite and > 0, or ``lam`` or ``r`` is not > 0
            (NaN is not).
    """

    h: float
    lam: float
    r: float

    @property
    def period(self) -> float:
        """The default period of a node, ``h / 4``: a step of ``eta_h = 0.25``
        per update, about that of the official implementation of PC-ALM. It is
        ``max_dt`` with ``r = h``; with ``r`` shorter, ``max_dt`` is shorter
        than it, and a node needs a period of its own."""
        return self.h / 4

    @property
    def max_dt(self) -> float:
        """The default longest ``dt`` an update counts,
        ``min(h / (2 * (1 + h / r)), 3 * lam)`` (``h / 4`` with ``r = h``,
        ``min(h / 2, 3 * lam)`` without the decay): the explicit Euler step of
        ``r`` then ``lam`` is stable for ``dt / h <= 1/2`` and ``dt / lam <= 3``
        (``specs/pc.md``, "dt の打ち切り"), and the decay pulls ``r``
        ``1 + h / r`` times as hard. Every node of a chain caps at the same time,
        so that they weigh their updates of the parameters alike."""
        return min(self.h / (2 * (1 + self.h / self.r)), 3 * self.lam)

    def __post_init__(self) -> None:
        if not 0 < self.h < math.inf:
            raise ValueError(f"TimeConstants.h must be finite and > 0, got {self.h!r}")
        if not self.lam > 0:
            raise ValueError(f"TimeConstants.lam must be > 0, got {self.lam!r}")
        if not self.r > 0:
            raise ValueError(f"TimeConstants.r must be > 0, got {self.r!r}")
