"""The time constants of PC-ALM in continuous time."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class TimeConstants:
    """The time constants of the continuous-time PC-ALM, in seconds.

    A node relaxes its deviation ``u``, its dual ``lam`` and its parameters
    ``theta`` with

    .. math::

        \\tau_h \\dot u = -(u + \\lambda + g),\\qquad
        \\tau_\\lambda \\dot\\lambda = u,\\qquad
        \\tau_\\theta \\dot\\theta = -g_\\theta / B

    and a step of ``dt`` seconds moves them by ``dt / tau`` of the right-hand side
    (the discrete step sizes are ``eta_h = dt / h``, ``alpha = dt / lam`` and
    ``eta_theta = dt / theta``).

    ``lam=math.inf`` makes the dual stay at zero (``alpha = 0``, plain predictive
    coding), and ``theta=math.inf`` makes the parameters stay where they are (no
    learning). ``h`` must be finite.

    The fields are keyword-only: all three are floats, and a swap by position
    could not be caught by the type checker.

    Example::

        tau = TimeConstants(h=4e-3, lam=8e-3, theta=1.28)
        pc_only = TimeConstants(h=4e-3, lam=math.inf, theta=1.28)

    Raises:
        TypeError: A field is not a number (``bool`` is not one).
        ValueError: ``h`` is not finite and positive, or ``lam`` or ``theta`` is
            not positive (NaN is not).
    """

    h: float
    lam: float
    theta: float

    def __post_init__(self) -> None:
        # User input: check the types at runtime.
        for name in ("h", "lam", "theta"):
            value: object = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"TimeConstants.{name} must be a number, got {value!r}")
            # NaN fails `value > 0`, so it is rejected here too.
            if not value > 0:
                raise ValueError(f"TimeConstants.{name} must be > 0, got {value!r}")
        if not math.isfinite(self.h):
            raise ValueError(f"TimeConstants.h must be finite, got {self.h!r}")
