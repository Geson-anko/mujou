"""The time constants of PC-ALM in continuous time."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TimeConstants:
    r"""The time constants of the continuous-time PC-ALM, in seconds.

    A node sends its prediction ``mu`` and holds the deviation $r = h - \mu$ of
    its activity from it and the dual ``lam``, which it relaxes with

    $$
    \tau_h \dot r = -(r + \lambda + g) - \frac{\tau_h}{\tau_r}\,r,\qquad
    \tau_\lambda \dot\lambda = r,
    $$

    and a step of ``dt`` seconds moves them by ``dt / tau`` of the right-hand
    side (the discrete step sizes are ``eta_h = dt / h`` and ``alpha = dt /
    lam``). The parameters ``theta`` move by the optimizer of the node
    (``pc.optim.SGD`` or ``pc.optim.AdamW``), which holds their time constant.

    ``lam=math.inf`` makes the dual stay at zero (``alpha = 0``); with
    ``r=math.inf`` too, the nodes are those of predictive coding with the fixed
    prediction assumption, whose error ``r`` settles at the credit of
    backpropagation (the decay would shrink it by ``1 + h / r`` at each node).
    ``h`` must be finite.

    ``r`` is the time constant of the decay of the deviation. The fixed point
    stays (``lam`` integrates ``r``, so ``r = 0`` there, and ``lam`` is the
    credit of backpropagation). The nodes send ``mu``, so the credit flows down
    the path only, and the steps are stable in any chain once they are in each
    node. The defaults are chosen together: with ``r = h``, ``lam = h / 2`` and
    a period of ``h / 2``, one update moves ``delta = r + lam`` to ``-g``
    exactly, so the credit moves one node in each period and does not
    overshoot (a larger step overshoots, and the overshoot grows at every
    node). An analog integrator that holds ``r`` and leaks is this decay.

    The fields are keyword-only and have no defaults: all are floats, and a swap
    by position could not be caught by the type checker.

    Example::

        tau = TimeConstants(h=4e-3, lam=2e-3, r=4e-3)  # a period of 2e-3
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
        """The default period of a node, ``h / (1 + h / r)``: the step that
        moves ``r`` all the way along its own pull (``h / 2`` with ``r = h``,
        ``h`` without the decay). With ``lam = h * r / (h + r)`` (``h / 2``
        with ``r = h``), it moves ``delta`` to ``-g`` in one update."""
        return self.h / (1 + self.h / self.r)

    @property
    def max_dt(self) -> float:
        """The default longest ``dt`` an update counts,
        ``min(h / (1 + h / r), 2 * lam)`` (the default ``period`` unless the
        dual is faster): the explicit Euler step of ``r`` then ``lam`` is then
        stable (``specs/pc.md``, "dt の打ち切り"). Every node of a chain caps
        at the same time, so that they weigh their updates of the parameters
        alike."""
        return min(self.period, 2 * self.lam)

    def __post_init__(self) -> None:
        if not 0 < self.h < math.inf:
            raise ValueError(f"TimeConstants.h must be finite and > 0, got {self.h!r}")
        if not self.lam > 0:
            raise ValueError(f"TimeConstants.lam must be > 0, got {self.lam!r}")
        if not self.r > 0:
            raise ValueError(f"TimeConstants.r must be > 0, got {self.r!r}")
