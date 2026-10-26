"""The time constants of PC-ALM in continuous time."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TimeConstants:
    r"""The time constants of the continuous-time PC-ALM, in seconds.

    A node relaxes its activity ``h`` and its dual ``lam`` with

    $$
    \tau_h \dot h = -(r + \lambda + g),\qquad
    \tau_\lambda \dot\lambda = r
    $$

    ($r = h - \mu$), and a step of ``dt`` seconds moves them by ``dt / tau`` of
    the right-hand side (the discrete step sizes are ``eta_h = dt / h`` and
    ``alpha = dt / lam``). The parameters ``theta`` move by the optimizer of the
    node (``pc.SGD`` or ``pc.AdamW``), which holds their time constant.

    ``lam=math.inf`` makes the dual stay at zero (``alpha = 0``, plain predictive
    coding). ``h`` must be finite.

    ``momentum`` is the time constant of the inertia of ``h`` in seconds (0, the
    default, has none). With it, a node keeps a velocity of ``h`` that remembers
    the force of the updates before, and moves ``h`` along it (a heavy ball):

    $$
    v \leftarrow e^{-\Delta t / \tau_m} v + (r + \lambda + g),\qquad
    h \leftarrow h - \frac{\Delta t}{\tau_h} v.
    $$

    The fixed point is that of the plain relaxation ($v = 0$ there). A force that
    persists moves ``h`` up to $1 / (1 - e^{-\Delta t / \tau_m})$ times faster, so
    the front of the credit of a deep chain crosses it in fewer periods; the slow
    mode of a chain near the identity does not relax faster for it (its rate is
    no better). The force weighs one update, not ``dt``, so the gain depends on
    the period. With a period of ``tau.h / 4`` and ``lam = tau.h``, a
    ``momentum`` of ``tau.h / 3`` ($e^{-0.75} \approx 0.47$ per update) halved the
    periods that 11 nodes of a ResNet needed, and one run in a few of them
    collapsed; ``tau.h / 4`` is safer. On a linear chain of identities the
    explicit step is stable up to about $e^{-\Delta t/\tau_m} \approx 0.78$, below
    it with late updates, and lower for a larger ``alpha = dt / lam``
    (``examples/cifar10/`` diverged at 0.8).

    The fields are keyword-only: both are floats, and a swap by position could
    not be caught by the type checker.

    Example::

        tau = TimeConstants(h=4e-3, lam=8e-3)
        pc_only = TimeConstants(h=4e-3, lam=math.inf)

    Raises:
        ValueError: ``h`` is not finite and > 0, ``lam`` is not > 0, or ``momentum``
            is not finite and >= 0 (NaN is not).
    """

    h: float
    lam: float
    momentum: float = 0.0

    @property
    def period(self) -> float:
        """The default period of a node, ``h / 4``: a step of ``h`` of
        ``eta_h = 0.25`` per update, about that of the official implementation
        of PC-ALM, and well within the measured stable range ``eta_h <= 0.5``
        (``specs/pc.md``)."""
        return self.h / 4

    @property
    def max_dt(self) -> float:
        """The default longest ``dt`` an update counts, ``min(h / 2, 3 * lam)``:
        the explicit Euler step of ``h`` then ``lam`` is stable for
        ``dt / h <= 1/2`` and ``dt / lam <= 3`` (``specs/pc.md``, "dt の打ち切り").
        Every node of a chain caps at the same time, so that they weigh their
        updates of the parameters alike."""
        return min(self.h / 2, 3 * self.lam)

    def __post_init__(self) -> None:
        if not 0 < self.h < math.inf:
            raise ValueError(f"TimeConstants.h must be finite and > 0, got {self.h!r}")
        if not self.lam > 0:
            raise ValueError(f"TimeConstants.lam must be > 0, got {self.lam!r}")
        if not 0 <= self.momentum < math.inf:
            raise ValueError(
                f"TimeConstants.momentum must be finite and >= 0, got {self.momentum!r}"
            )
