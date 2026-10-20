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

# The timer counts whole nanoseconds, so every dt is a whole number of them; the
# phase, a sum of such dt as floats, may fall short of learn_interval by a rounding
# error at the step that reaches it.
_PHASE_TOLERANCE = 5e-10


class Learner(Node):
    """A node that holds ``module`` (the ``f`` of ``specs/pc-module.md``) and
    updates its parameters ``theta`` once ``learn_interval`` has passed since the
    input changed.

    The input changes when the ``time`` carried with the activity it receives
    changes (the host wrote a new input). The state holds, besides ``module``, the
    last such ``time`` seen (``"_seen_time"``), the time since then or since the
    last update of ``theta`` (``"_phase"``) and the time since the last update of
    ``theta`` (``"_elapsed"``), which weighs the update.

    Raises:
        TypeError: ``tau`` is not a ``TimeConstants``, ``carry`` is not a spec
            of ``carry_spec``, or ``module`` does not return a tensor.
        ValueError: ``learn_interval`` is neither ``None`` nor finite and
            positive; ``input_shape[0]`` or the first dimension of the output of
            ``module`` is not the batch of ``carry``; or a parameter or buffer of
            ``module`` is not on ``device``.
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
        dtype: torch.dtype | None,
        device: torch.device | str,
    ) -> None:
        # User input: check the types at runtime.
        if not isinstance(tau, TimeConstants):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"tau must be a TimeConstants, got {tau!r}")
        if learn_interval is not None and not (
            math.isfinite(learn_interval) and learn_interval > 0
        ):
            raise ValueError(
                f"learn_interval must be None or finite and > 0, got {learn_interval!r}"
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
        self._learn_interval = learn_interval
        # No time a host writes at (>= 0): the first update starts the phase. The
        # carried data before the first send is zeros, whose time 0 is also that
        # of a first write at time 0, so a new input is not told by its time
        # alone there; whether to learn rests on the carried mask, all False
        # until a real sample arrives.
        self._seen_time = -1
        self._phase = 0.0
        self._elapsed = 0.0
        for name in ("_seen_time", "_phase", "_elapsed"):
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

    def _tick(self, dt: float, time: int) -> bool:
        """Count ``dt`` seconds of an update that received an input written at
        ``time`` (ns), and return whether to update ``theta`` in it."""
        self._elapsed += dt
        if time != self._seen_time:
            self._seen_time = time
            self._phase = 0.0
        else:
            self._phase += dt
        return math.isfinite(self._tau.theta) and (
            self._learn_interval is None
            or self._phase >= self._learn_interval - _PHASE_TOLERANCE
        )

    def _learned(self) -> None:
        """Start the next interval: ``theta`` was updated, or the update was
        skipped for want of a target."""
        self._elapsed = 0.0
        self._phase = 0.0
