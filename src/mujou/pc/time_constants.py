"""The time constants of PC-ALM in continuous time."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TimeConstants:
    r"""The time constants of the continuous-time PC-ALM, in seconds.

    A node relaxes its activity ``h`` and its dual ``lam`` with

    $$
    \tau_h \dot h = -(r + \lambda + g) - \frac{\tau_h}{\tau_r}\,r,\qquad
    \tau_\lambda \dot\lambda = r
    $$

    ($r = h - \mu$), and a step of ``dt`` seconds moves them by ``dt / tau`` of
    the right-hand side (the discrete step sizes are ``eta_h = dt / h`` and
    ``alpha = dt / lam``). The parameters ``theta`` move by the optimizer of the
    node (``pc.optim.SGD`` or ``pc.optim.AdamW``), which holds their time constant.

    ``lam=math.inf`` makes the dual stay at zero (``alpha = 0``, plain predictive
    coding). ``h`` must be finite.

    ``r`` makes the deviation ``r`` decay with its own time constant, an extra
    pull of ``h`` towards ``mu``. The fixed point stays (``lam`` integrates
    ``r``, so ``r = 0`` there, and ``lam`` is the credit of backpropagation
    still), and in a deep chain ``lam`` gets near that credit sooner: in a
    ResNet of 11 nodes with ``lam = h / 4``, the cosine of the ``lam`` of the
    first node with the credit after ``24 * h`` was 0.84 without the decay and
    0.94 with ``r = h / 3`` (``specs/pc.md``). The default ``math.inf`` adds nothing. An analog
    integrator that holds ``r`` (not ``h``) and leaks is this decay, which is
    why a leak there is harmless where one of ``h`` towards 0 is not.

    The fields are keyword-only: all are floats, and a swap by position could
    not be caught by the type checker.

    Example::

        tau = TimeConstants(h=4e-3, lam=8e-3)
        pc_only = TimeConstants(h=4e-3, lam=math.inf)
        damped = TimeConstants(h=4e-3, lam=1e-3, r=4e-3 / 3)

    Raises:
        ValueError: ``h`` is not finite and > 0, or ``lam`` or ``r`` is not > 0
            (NaN is not).
    """

    h: float
    lam: float
    r: float = math.inf

    @property
    def period(self) -> float:
        """The default period of a node, ``h / 4``: a step of ``h`` of
        ``eta_h = 0.25`` per update, about that of the official implementation
        of PC-ALM, and well within the measured stable range ``eta_h <= 0.5``
        (``specs/pc.md``)."""
        return self.h / 4

    @property
    def max_dt(self) -> float:
        """The default longest ``dt`` an update counts,
        ``min(h / (2 * (1 + h / r)), 3 * lam)`` (``min(h / 2, 3 * lam)`` without
        the decay of ``r``): the explicit Euler step of ``h`` then ``lam`` is
        stable for ``dt / h <= 1/2`` and ``dt / lam <= 3`` (``specs/pc.md``, "dt
        の打ち切り"), and the decay of ``r`` pulls ``h`` ``1 + h / r`` times as
        hard. Every node of a chain caps at the same time, so that they weigh
        their updates of the parameters alike."""
        return min(self.h / (2 * (1 + self.h / self.r)), 3 * self.lam)

    def __post_init__(self) -> None:
        if not 0 < self.h < math.inf:
            raise ValueError(f"TimeConstants.h must be finite and > 0, got {self.h!r}")
        if not self.lam > 0:
            raise ValueError(f"TimeConstants.lam must be > 0, got {self.lam!r}")
        if not self.r > 0:
            raise ValueError(f"TimeConstants.r must be > 0, got {self.r!r}")
