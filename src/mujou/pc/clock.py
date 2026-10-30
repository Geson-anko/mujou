"""When a node that learns updates its parameters: ``LearnClock`` and the
``Tick`` it returns for each update."""

import math
from collections.abc import Mapping
from typing import Any, NamedTuple, Self, override

from mujou import Stateful
from mujou.pc.time_constants import TimeConstants
from mujou.time import Timestamp

__all__ = [
    "LearnClock",
    "Tick",
]


class Tick(NamedTuple):
    """What ``LearnClock.tick`` tells one update of a node."""

    dt: float
    """The ``dt`` of the update, capped at ``max_dt``: the time to move by."""
    new: bool
    """Whether the input is new: its ``time`` is not the one seen last."""
    learn: bool
    """Whether to update the parameters in this update."""
    elapsed: float
    """The sum of the capped ``dt`` since the parameters were last updated, this
    update included: the weight of the update when ``learn`` is True."""


class LearnClock(Stateful):
    """Decides, update by update, when a node updates its parameters, from the
    time the host wrote the node's input.

    The node calls ``tick(dt, time, now)`` once in each update, with the ``dt``
    of the update, the ``Timestamp`` the input it received was written at, and
    the time now on the run's timer. The clock learns in the first update at or
    after ``time + k * learn_interval`` (``k = 1, 2, ...``), but
    never in the update that first sees an input: its gradient is still that of
    the input before. The nodes on a path read the one timer of the run, so
    they learn at about the same time, however deep they are. With
    ``learn_interval=None``, it learns in every update, a new input or not;
    with ``math.inf``, never.

    A ``Tick`` with ``learn`` counts as learned: the next interval starts there,
    whether the node updates its parameters or skips them for want of a target.
    ``elapsed`` sums the capped ``dt`` since then; an input that changes before
    the clock learned from it drops that sum, as an input without a target does,
    or inputs that change faster than ``learn_interval`` would pile the time up
    into one large update. ``dt`` is capped at ``max_dt``, and the time beyond
    it is dropped from ``elapsed`` too.

    The clock starts as if it had seen no input (no time a host writes at): the
    first update sees a new input, also one written at time 0.

    The state (``state_dict()``) holds the time of the input seen last
    (``"seen_ns"``, ``None`` before any), the time from which the next update
    learns
    (``"learn_at_ns"``), ``"elapsed"``, and whether the clock has learned since
    the input changed (``"learned_input"``). An attribute of a node that holds a
    ``LearnClock`` is in the node's state by itself.

    It is not tied to PC-ALM: any node that learns from inputs the host writes
    can use it. ``LearnClock.of`` makes the clock of a node of PC-ALM from its
    time constants.

    Example::

        self.clock = pc.clock.LearnClock(period=self.period, learn_interval=24e-3,
                                   max_dt=2e-3)
        # In on_update(dt):
        tick = self.clock.tick(dt, activity["time"], self.timer.now())
        # ... move by tick.dt; if tick.learn, learn weighted by tick.elapsed.

    Args:
        period: The period of the node, in seconds.
        learn_interval: The time in seconds from the write of an input to the
            first update of the parameters, and between the updates while the
            input stays. ``None`` learns in every update, ``math.inf`` never.
        max_dt: The longest ``dt`` an update counts, in seconds.

    Raises:
        ValueError: ``learn_interval`` is neither ``None`` nor > 0, ``max_dt`` is
            not > 0, or ``period`` is longer than ``max_dt``.
    """

    def __init__(
        self, *, period: float, learn_interval: float | None, max_dt: float
    ) -> None:
        if not (learn_interval is None or learn_interval > 0):
            raise ValueError(
                f"learn_interval must be None or > 0 (math.inf never learns), got "
                f"{learn_interval!r}"
            )
        if not max_dt > 0:
            raise ValueError(f"max_dt must be > 0, got {max_dt!r}")
        if period > max_dt:
            raise ValueError(
                f"the period {period!r} s must not be longer than max_dt "
                f"{max_dt!r} s, or every update would drop the time beyond "
                "max_dt; give a shorter period or a longer max_dt"
            )
        # An infinite interval puts the points to learn at infinity.
        self._interval_ns = (
            learn_interval
            if learn_interval is None or math.isinf(learn_interval)
            else max(1, round(learn_interval * 1e9))
        )
        self._max_dt = max_dt
        # None before the first input.
        self._seen_ns: int | None = None
        self._learn_at_ns: float = 0
        self._elapsed = 0.0
        self._learned_input = False

    @classmethod
    def of(
        cls,
        tau: TimeConstants,
        *,
        period: float,
        learn_interval: float | None,
        max_dt: float | None = None,
    ) -> Self:
        """Return the clock of a node of PC-ALM with the time constants ``tau``,
        as ``Layer`` and ``Output`` make theirs: ``max_dt`` defaults to
        ``tau.max_dt``.

        Raises:
            ValueError: As ``LearnClock`` raises.
        """
        return cls(
            period=period,
            learn_interval=learn_interval,
            max_dt=tau.max_dt if max_dt is None else max_dt,
        )

    def tick(self, dt: float, time: Timestamp, now: Timestamp) -> Tick:
        """Count an update of ``dt`` seconds that received an input written at
        ``time``, at ``now`` on the run's timer, and return what the update
        does. With ``learn``, the clock has learned at ``now``."""
        time_ns, now_ns = time.ns, now.ns
        dt = min(dt, self._max_dt)
        new = time_ns != self._seen_ns
        if new:
            self._see(time_ns)
        self._elapsed += dt
        elapsed = self._elapsed
        learn = self._interval_ns is None or (not new and now_ns >= self._learn_at_ns)
        if learn:
            self._learned(now_ns)
        return Tick(dt, new, learn, elapsed)

    def _see(self, time_ns: int) -> None:
        self._seen_ns = time_ns
        if self._interval_ns is not None:
            self._learn_at_ns = time_ns + self._interval_ns
        if not self._learned_input:
            self._elapsed = 0.0
        self._learned_input = False

    def _learned(self, now_ns: int) -> None:
        """Start the next interval: the next update learns from the first
        ``time_ns + k * learn_interval`` after ``now_ns``."""
        self._elapsed = 0.0
        self._learned_input = True
        if self._interval_ns is not None:
            behind = now_ns - self._learn_at_ns  # >= 0
            self._learn_at_ns += (behind // self._interval_ns + 1) * self._interval_ns

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return the state: plain numbers, which ``torch.load`` reads with
        ``weights_only=True``."""
        return {
            "seen_ns": self._seen_ns,
            "learn_at_ns": self._learn_at_ns,
            "elapsed": self._elapsed,
            "learned_input": self._learned_input,
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the state that ``state_dict`` returned."""
        self._seen_ns = state["seen_ns"]
        self._learn_at_ns = state["learn_at_ns"]
        self._elapsed = state["elapsed"]
        self._learned_input = state["learned_input"]
