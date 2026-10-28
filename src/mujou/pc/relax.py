r"""The relaxers that move the deviation ``r = h - mu`` of a node in time:
``Euler`` and ``HeavyBall``.

A relaxer is a setting (``Relaxer``), as an optimizer is: a frozen value that
holds no state and may be shared by every node. A node binds it to its own
deviation with ``bind(r)``, which returns the ``BoundRelaxer`` that holds the
state (the velocity of ``HeavyBall``) and moves ``r`` by ``step_(r, force, dt,
tau_h)``. It is the counterpart of ``pc.optim.SGD`` and ``pc.optim.AdamW`` for
``r``: those move the parameters along their gradient, a relaxer moves the
deviation along the force on it, $F = (1 + \tau_h / \tau_r)\,r + \lambda + g$
(``functional.relax_``). The dual ``lam`` is not a relaxer's business: it
follows the moved ``r``.

A relaxer changes how ``r`` gets to its fixed point, not where it is: at the
fixed point the force is zero and the state of the relaxer is at rest (the
velocity of ``HeavyBall`` is zero), and ``r`` stays there.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, override

import torch

from mujou import Stateful

__all__ = [
    "Euler",
    "HeavyBall",
    "Relaxer",
    "BoundRelaxer",
]


class BoundRelaxer(Stateful):
    """A relaxer bound to the deviation ``r`` of one node, which holds its
    state.

    The node that bound it owns it and calls it from its own hooks only. It is a
    ``Stateful``, so an attribute of a node that holds it is in the node's state
    by itself: ``state_dict`` returns the state with tensors that share memory
    with the relaxer's, and ``load_state_dict`` copies the tensors of the state
    into those. Only ``step_`` moves ``r``.
    """

    @abstractmethod
    def step_(
        self, r: torch.Tensor, force: torch.Tensor, dt: float, tau_h: float
    ) -> None:
        r"""Move ``r`` in place by ``dt`` seconds of the flow along ``force``,
        without recording gradients.

        Args:
            r: The deviation ``h - mu`` of the node, ``(*batch, *out)``.
            force: The force $F = (1 + \tau_h / \tau_r)\,r + \lambda + g$ on
                it, of the shape of ``r``; the plain flow is
                $\tau_h \dot r = -F$.
            dt: The time to move by, in seconds (0 changes nothing).
            tau_h: The time constant $\tau_h$ of ``r``, in seconds
                (``TimeConstants.h``).
        """

    @abstractmethod
    def reset_(self) -> None:
        """Forget the forces before: the node calls it with a new sample, where
        what the relaxer remembers is the forces of the sample before. A relaxer
        without memory does nothing."""


class Relaxer(ABC):
    r"""The setting of how the deviation ``r = h - mu`` of a node moves in time.

    It holds values only, no state, so the nodes may share it. ``bind(r)`` makes
    the ``BoundRelaxer`` of the deviation of one node, which holds the state.

    A relaxer of your own inherits this class, and its bound relaxer
    ``BoundRelaxer``. For example, a step that is capped in size::

        @dataclass(frozen=True)
        class Capped(pc.relax.Relaxer):
            limit: float

            def bind(self, r):
                return _BoundCapped(self.limit)

        class _BoundCapped(pc.relax.BoundRelaxer):
            def __init__(self, limit):
                self._limit = limit

            def step_(self, r, force, dt, tau_h):
                step = (dt / tau_h) * force
                r.sub_(step.clamp(-self._limit, self._limit))

            def reset_(self):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, state):
                pass

    Its fixed point must be a zero of the force (with the state of the relaxer
    at rest), or the node no longer solves PC-ALM: ``lam`` is the credit of
    backpropagation only where ``r`` stops at a zero of the force.
    """

    @abstractmethod
    def bind(self, r: torch.Tensor) -> BoundRelaxer:
        """Return the relaxer bound to a deviation of the shape, dtype and
        device of ``r``, with a fresh state."""


@dataclass(frozen=True, kw_only=True)
class Euler(Relaxer):
    r"""The plain gradient flow of ``r``, $\tau_h \dot r = -F$, by one explicit
    Euler step of ``dt``: ``r -= (dt / tau_h) * force``.

    It is the default. The bound relaxer has no state (an empty ``state_dict``).

    Example::

        hidden = pc.Layer(module, act_spec=..., tau=tau, optimizer=...,
                          relaxer=pc.relax.Euler(), learn_interval=24e-3)
    """

    @override
    def bind(self, r: torch.Tensor) -> BoundRelaxer:
        """Return the relaxer bound to ``r``."""
        return _BoundEuler()


class _BoundEuler(BoundRelaxer):
    @override
    def step_(
        self, r: torch.Tensor, force: torch.Tensor, dt: float, tau_h: float
    ) -> None:
        with torch.no_grad():
            r.sub_((dt / tau_h) * force)  # tau_h dr/dt = -F

    @override
    def reset_(self) -> None:
        pass

    @override
    def state_dict(self) -> dict[str, Any]:
        return {}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class HeavyBall(Relaxer):
    r"""``r`` as a heavy ball: a velocity $v$ remembers the forces of the updates
    before, and ``r`` moves along it.

    $$
    v \leftarrow e^{-\Delta t / \tau} v + F,\qquad
    r \leftarrow r - \frac{\Delta t}{\tau_h} v.
    $$

    The fixed point is that of ``Euler`` ($v = 0$ there). A force that lasts
    moves ``r`` up to $1 / (1 - e^{-\Delta t / \tau})$ times faster. The force
    weighs one update, not ``dt``, so the gain depends on the period. A step of
    ``dt = 0`` changes nothing, the velocity included. The node resets the
    velocity with each new sample (``reset_``): the forces it remembers are
    those of the sample before.

    With ``r`` as the state of a node, it has not been seen to help: in a chain
    of 4 residual nodes (``tau.r = tau.h``, a period of ``tau.h / 4``), the
    credit crossed the chain no sooner, as each period carries it one node
    further whatever the step, and a ``tau`` of ``0.75 * tau.h`` diverged where
    ``tau.h / 2`` did not (``specs/pc.md``).

    The state (``state_dict()``) holds the velocity ``"velocity"``, a tensor of
    the shape, dtype and device of ``r``.

    Example::

        hidden = pc.Layer(module, act_spec=..., tau=tau, optimizer=...,
                          relaxer=pc.relax.HeavyBall(tau=tau.h / 4),
                          learn_interval=24e-3)

    Args:
        tau: The time constant $\tau$ of the velocity, in seconds.

    Raises:
        ValueError: ``tau`` is not finite and > 0.
    """

    tau: float

    def __post_init__(self) -> None:
        if not 0 < self.tau < math.inf:
            raise ValueError(f"HeavyBall.tau must be finite and > 0, got {self.tau!r}")

    @override
    def bind(self, r: torch.Tensor) -> BoundRelaxer:
        """Return the relaxer bound to ``r``, with the velocity at zero."""
        return _BoundHeavyBall(torch.zeros_like(r), self.tau)


class _BoundHeavyBall(BoundRelaxer):
    def __init__(self, velocity: torch.Tensor, tau: float) -> None:
        self._velocity = velocity
        self._tau = tau

    @override
    def step_(
        self, r: torch.Tensor, force: torch.Tensor, dt: float, tau_h: float
    ) -> None:
        if dt == 0:
            return  # an update of no time must not store a force
        with torch.no_grad():
            self._velocity.mul_(math.exp(-dt / self._tau)).add_(force)
            r.sub_((dt / tau_h) * self._velocity)

    @override
    def reset_(self) -> None:
        self._velocity.zero_()

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"velocity": self._velocity}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with torch.no_grad():
            self._velocity.copy_(state["velocity"])
