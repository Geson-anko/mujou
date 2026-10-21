"""The time constants of PC-ALM in continuous time."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TimeConstants:
    r"""The time constants of the continuous-time PC-ALM, in seconds.

    A node relaxes its activity ``h``, its dual ``lam`` and its parameters
    ``theta`` with

    $$
    \tau_h \dot h = -(r + \lambda + g),\qquad
    \tau_\lambda \dot\lambda = r,\qquad
    \tau_\theta \dot\theta = -g_\theta / B
    $$

    ($r = h - \mu$), and a step of ``dt`` seconds moves them by ``dt / tau`` of
    the right-hand side (the discrete step sizes are ``eta_h = dt / h``,
    ``alpha = dt / lam`` and ``eta_theta = dt / theta``). One update of
    ``theta`` that weighs ``elapsed`` seconds is thus one step of SGD on the
    mean gradient of the batch with the learning rate ``elapsed / theta``.

    ``lam=math.inf`` makes the dual stay at zero (``alpha = 0``, plain predictive
    coding), and ``theta=math.inf`` makes the parameters stay where they are (no
    learning). ``h`` must be finite.

    The fields are keyword-only: all three are floats, and a swap by position
    could not be caught by the type checker.

    Example::

        tau = TimeConstants(h=4e-3, lam=8e-3, theta=1.28)
        pc_only = TimeConstants(h=4e-3, lam=math.inf, theta=1.28)

    Raises:
        ValueError: ``h`` is not finite and > 0, or ``lam`` or ``theta`` is not
            > 0 (NaN is not).
    """

    h: float
    lam: float
    theta: float

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
        updates of ``theta`` alike."""
        return min(self.h / 2, 3 * self.lam)

    def __post_init__(self) -> None:
        if not 0 < self.h < math.inf:
            raise ValueError(f"TimeConstants.h must be finite and > 0, got {self.h!r}")
        for name in ("lam", "theta"):
            value: float = getattr(self, name)
            if not value > 0:
                raise ValueError(f"TimeConstants.{name} must be > 0, got {value!r}")
