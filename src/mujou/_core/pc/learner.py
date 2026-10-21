"""The base of the nodes of ``mujou.pc`` that learn: the module ``f`` and when to
update its parameters."""

import math
from collections.abc import Sequence

import torch
from torch import nn

from mujou._core.node import Node
from mujou._core.pc.carry import Carried, batch_of, check_batch
from mujou._core.pc.time_constants import TimeConstants
from mujou._core.spec import DictSpec, TensorSpec


class Learner(Node):
    """A node that holds ``module`` (the ``f`` of ``specs/pc-module.md``) and
    updates its parameters ``theta`` once ``learn_interval`` has passed since the
    host wrote the input.

    The input changes when the ``time`` carried with the activity it receives
    changes (the host wrote a new input at that time). The node learns in its
    first update at or after ``time + k * learn_interval`` (``k = 1, 2, ...``)
    on the run's time, but not in the update that first sees the input, so the
    nodes on a path learn at about the same time, however deep they are. An
    update counts at most ``max_dt`` seconds (``_cap``). The state holds,
    besides ``module``, the last such ``time`` seen (``"_seen_time"``), the time
    (ns) from which the next update learns (``"_learn_at"``), the time since the
    last update of ``theta`` (``"_elapsed"``), which weighs the update, and
    whether the node has learned (or skipped learning for want of a target)
    since the input changed (``"_learned_input"``).

    Raises:
        TypeError: ``tau`` is not a ``TimeConstants``, ``carry`` is not a spec
            of ``carry_spec``, ``module`` does not return a tensor, or
            ``learn_interval`` or ``max_dt`` is a ``bool``.
        ValueError: ``learn_interval`` or ``max_dt`` is neither ``None`` nor
            finite and positive; ``period`` is longer than ``max_dt``;
            ``input_shape[0]`` or the first dimension of the output of
            ``module`` is not the batch of ``carry``; or a parameter or buffer
            of ``module`` is not on ``device``.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        input_shape: Sequence[int],
        carry: DictSpec[Carried],
        tau: TimeConstants,
        learn_interval: float | None,
        period: float | None,
        max_dt: float | None,
        dtype: torch.dtype | None,
        device: torch.device | str,
    ) -> None:
        # User input: check the types at runtime.
        if not isinstance(tau, TimeConstants):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"tau must be a TimeConstants, got {tau!r}")
        for name, value in (("learn_interval", learn_interval), ("max_dt", max_dt)):
            if isinstance(value, bool):
                raise TypeError(f"{name} must be None or a number, got {value!r}")
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{name} must be None or finite and > 0, got {value!r}"
                )
        super().__init__(device=device, period=tau.h / 4 if period is None else period)
        self._batch = batch_of(carry)
        check_batch(input_shape, self._batch, "input_shape")
        for name, tensor in [*module.named_parameters(), *module.named_buffers()]:
            if tensor.device != self.device:
                raise ValueError(
                    f"module's {name!r} is on {tensor.device}, but the node computes "
                    f"on {self.device}"
                )
        self._input = TensorSpec(input_shape, dtype)
        self._tau = tau
        self._interval_ns = (
            None if learn_interval is None else max(1, round(learn_interval * 1e9))
        )
        # The explicit Euler step of h then lam is stable for dt / tau.h <= 1/2
        # and dt / tau.lam <= 3 (specs/pc-module.md, "dt が伸びたとき").
        self._max_dt = min(tau.h / 2, 3 * tau.lam) if max_dt is None else max_dt
        if self.period > self._max_dt:
            raise ValueError(
                f"the period {self.period!r} s must not be longer than max_dt "
                f"{self._max_dt!r} s, or every update would drop the time beyond "
                "max_dt; give a shorter period or a longer max_dt"
            )
        # No time a host writes at (>= 0), so the first update sees a new input.
        # The carried data before the first send is zeros, whose time 0 is also
        # that of a first write at time 0, so a new input is not told by its
        # time alone there; whether to learn rests on the carried mask, all
        # False until a real sample arrives.
        self._seen_time = -1
        self._learn_at = 0
        self._elapsed = 0.0
        self._learned_input = False
        for name in ("_seen_time", "_learn_at", "_elapsed", "_learned_input"):
            self.register_state(name)
        self.module = module
        # Frozen parameters do not learn.
        self._params = [p for p in module.parameters() if p.requires_grad]

    def _probe(self) -> torch.Tensor:
        """Return the output of ``module`` for zeros of the input spec, to give
        the specs of what the node sends.

        Raises:
            TypeError: ``module`` does not return a tensor.
            ValueError: Its first dimension is not the batch.
        """
        with torch.no_grad():
            out: object = self.module(self._input.zeros(device=self.device))
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"module must return a tensor, got {type(out).__name__} for the "
                f"input {self._input!r}"
            )
        check_batch(out.shape, self._batch, "the output of module")
        return out

    def _cap(self, dt: float) -> float:
        """Return the time an update of ``dt`` seconds counts: at most
        ``max_dt``. The time beyond it is dropped (``specs/pc-module.md``, "dt
        が伸びたとき")."""
        return min(dt, self._max_dt)

    def _tick(self, dt: float, time: int) -> tuple[bool, bool]:
        """Count ``dt`` seconds (already capped) of an update that received an
        input written at ``time`` (ns), and return whether the input is new and
        whether to update ``theta`` in this update.

        With ``learn_interval``, the update that first sees an input does not
        learn: its ``delta`` is still that of the input before, which a node
        deep on a path (or one that was stalled) may see only after
        ``time + learn_interval``. It learns in the next update. With ``None``,
        every update learns, a new input or not.

        An input that changes before the node has learned from it drops the
        time counted for it, as an input without a target does: otherwise an
        input that changes faster than ``learn_interval`` would pile the time up
        into one large update.
        """
        new = time != self._seen_time
        if new:
            self._seen_time = time
            if self._interval_ns is not None:
                self._learn_at = time + self._interval_ns
            if not self._learned_input:
                self._elapsed = 0.0
            self._learned_input = False
        self._elapsed += dt
        learn = math.isfinite(self._tau.theta) and (
            self._interval_ns is None
            or (not new and self.timer.now().ns >= self._learn_at)
        )
        return new, learn

    def _learned(self) -> None:
        """Start the next interval: ``theta`` was updated, or the update was
        skipped for want of a target. The next update learns from the first
        ``time + k * learn_interval`` after now."""
        self._elapsed = 0.0
        self._learned_input = True
        if self._interval_ns is not None:
            behind = self.timer.now().ns - self._learn_at  # >= 0
            self._learn_at += (behind // self._interval_ns + 1) * self._interval_ns
