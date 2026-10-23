r"""The optimizers that move the parameters of a node in time: ``SGD`` and
``AdamW``.

An optimizer here is a setting (``Optimizer``): a frozen value that holds no
state and may be shared by every node. A node binds it to its own parameters
with ``bind(params)``, which returns the ``BoundOptimizer`` that holds the state
(the moments of ``AdamW``) and moves the parameters by ``step_(grads,
elapsed)``. The time constants are in seconds, so the flow does not depend on
how often the node learns: ``elapsed`` is the time the step weighs.

They do not use ``torch.optim``: its optimizers count steps rather than time,
and making one brings the GIL back into a free-threaded Python.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import torch

from mujou import Stateful


class BoundOptimizer(Stateful):
    """An optimizer bound to the parameters of one node, which holds its state.

    It holds the parameters by reference. The node that bound it owns it and
    calls it from its own hooks only. It is a ``Stateful``, so an attribute of a
    node that holds it is in the node's state by itself: ``state_dict`` returns
    the state with tensors that share memory with the optimizer's, and
    ``load_state_dict`` copies the tensors of the state into those (it keeps
    none of the state it is given).
    """

    @abstractmethod
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        """Move the parameters in place by ``elapsed`` seconds of the flow,
        without recording gradients.

        Args:
            grads: The gradients of the parameters, in the order they were
                bound and of their shapes, dtypes and devices, averaged over the
                batch (the node applies the ``1 / B``; the optimizer does not
                know the batch).
            elapsed: The time the step weighs, in seconds (``Tick.elapsed``:
                the time since the parameters last moved, capped, without the
                time the clock dropped).

        Raises:
            ValueError: ``grads`` does not have one gradient per parameter.
        """


class Optimizer(ABC):
    """The setting of how the parameters of a node move in time.

    It holds values only, no state, so the nodes may share it. ``bind(params)``
    makes the ``BoundOptimizer`` of the parameters of one node, which holds the
    state. Give it only the parameters that learn (``pc.Layer`` and
    ``pc.Output`` bind those that require gradients).

    An optimizer of your own inherits this class, and its bound optimizer
    ``BoundOptimizer``. For example, SGD with a learning rate per step rather
    than per second::

        @dataclass(frozen=True)
        class StepSGD(pc.Optimizer):
            lr: float

            def bind(self, params):
                return _BoundStepSGD(list(params), self.lr)

        class _BoundStepSGD(pc.BoundOptimizer):
            def __init__(self, params, lr):
                self._params, self._lr = params, lr

            def step_(self, grads, elapsed):
                with torch.no_grad():
                    for p, g in zip(self._params, grads, strict=True):
                        p.sub_(g, alpha=self._lr)

            def state_dict(self):
                return {}

            def load_state_dict(self, state):
                pass
    """

    @abstractmethod
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        """Return the optimizer bound to ``params``, with a fresh state."""


def _check_positive(owner: str, name: str, value: float) -> None:
    if not 0 < value < math.inf:
        raise ValueError(f"{owner}.{name} must be finite and > 0, got {value!r}")


@dataclass(frozen=True, kw_only=True)
class SGD(Optimizer):
    r"""Gradient descent in time, $\tau_\theta \dot\theta = -g$.

    A step of ``elapsed`` seconds is one step of SGD with the learning rate
    ``elapsed / tau``: ``p -= (elapsed / tau) * g``. With $\Delta t$ seconds
    between two updates of the parameters (the time each input is shown, for a
    node that learns once from each), the step is of the learning rate
    ``Δt / tau``. The bound optimizer has no state (an empty ``state_dict``).

    Example::

        hidden = pc.Layer(module, act_spec=..., tau=tau,
                          optimizer=pc.SGD(tau=0.3), learn_interval=24e-3)

    Args:
        tau: The time constant $\tau_\theta$ of the parameters, in seconds.

    Raises:
        ValueError: ``tau`` is not finite and > 0. Never to learn, give the node
            ``learn_interval=math.inf``.
    """

    tau: float

    def __post_init__(self) -> None:
        _check_positive("SGD", "tau", self.tau)

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        """Return the optimizer bound to ``params``."""
        return _BoundSGD(list(params), self.tau)


class _BoundSGD(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], tau: float) -> None:
        self._params = params
        self._tau = tau

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        with torch.no_grad():
            for p, g in zip(self._params, grads, strict=True):
                p.sub_(g, alpha=elapsed / self._tau)  # tau dtheta/dt = -g

    @override
    def state_dict(self) -> dict[str, Any]:
        return {}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class AdamW(Optimizer):
    r"""AdamW in time: moments that decay with time constants, and decoupled
    weight decay.

    With the gradient $g$ (averaged over the batch):

    $$
    \tau_m \dot m = g - m,\qquad
    \tau_v \dot v = g^2 - v,\qquad
    \tau_\theta \dot\theta
        = -\left(\frac{\hat m}{\sqrt{\hat v} + \varepsilon} + w\,\theta\right),
    $$

    where $\hat m = m / c_m$ and $\hat v = v / c_v$ correct the bias of the
    moments that start at zero, with the masses
    $\tau_m \dot c_m = 1 - c_m$ and $\tau_v \dot c_v = 1 - c_v$ from
    $c = 0$: $c$ is the share of the moment made of gradients so far (the rest
    is the zero it started from), $1 - e^{-t/\tau}$ for gradients seen over
    the time $t$.

    A step of ``elapsed`` $= \Delta t$ seconds moves the moments and masses
    exactly as the flow would with $g$ held over $\Delta t$
    ($m \leftarrow m + a_m (g - m)$ and $c_m \leftarrow c_m + a_m (1 - c_m)$
    with $a_m = 1 - e^{-\Delta t/\tau_m}$, the same for $v$), so that two steps
    of $\Delta t$ and one of $2\Delta t$ give the same moments for the same
    $g$. $\theta$ then steps with $\eta = \Delta t / \tau_\theta$, in the
    order of ``torch.optim.AdamW``: the decay first, on the $\theta$ before the
    step, and the Adam term with the moments and masses just moved:

    $$
    \theta \leftarrow \theta (1 - \eta w)
        - \eta\,\frac{\hat m}{\sqrt{\hat v} + \varepsilon}.
    $$

    A step of ``elapsed == 0`` moves nothing.

    With a constant $\Delta t$, $1 - a = e^{-\Delta t/\tau}$ is the $\beta$ of
    ``torch.optim.AdamW`` and $c = 1 - \beta^t$ its bias correction after $t$
    steps: the steps are those of ``torch.optim.AdamW(lr=Δt / tau,
    betas=(exp(-Δt / tau_m), exp(-Δt / tau_v)), eps=eps,
    weight_decay=weight_decay)``, up to rounding. From the $\beta$ of a node
    that learns every $\Delta t$, $\tau_m = -\Delta t / \ln\beta_1$: learning
    every 32 ms, $\beta_1 = 0.9$ is about 0.30 s and $\beta_2 = 0.999$ about
    32 s.

    ``weight_decay`` $w$ has no unit, and decays $\theta$ with the time
    constant $\tau_\theta / w$, as ``weight_decay`` does with ``lr`` in
    ``torch.optim.AdamW``.

    The state (``state_dict()``) holds the moments ``"m"`` and ``"v"`` (a list
    of tensors each, of the dtype and device of the parameters) and the masses
    ``"c_m"`` and ``"c_v"`` (Python floats in ``[0, 1]``). There is no count
    of steps.

    A step moves each element by about $\eta$ whatever the size of its
    gradient, so ``tau`` is ``Δt / lr`` for the ``lr`` of Adam, with $\Delta t$
    the time between two updates of the parameters.

    Example (``lr = 1e-3``, learning once from each input, shown for 32 ms)::

        hidden = pc.Layer(module, act_spec=..., tau=tau,
                          optimizer=pc.AdamW(tau=32e-3 / 1e-3, tau_m=0.3,
                                             tau_v=30.0),
                          learn_interval=24e-3)

    Args:
        tau: The time constant $\tau_\theta$ of the parameters, in seconds.
        tau_m: The time constant $\tau_m$ of the first moment, in seconds.
        tau_v: The time constant $\tau_v$ of the second moment, in seconds.
        eps: The $\varepsilon$ added to $\sqrt{\hat v}$. It is > 0, so that an
            element whose gradient has been 0 does not become ``0 / 0``.
        weight_decay: The decoupled weight decay $w$.

    Raises:
        ValueError: ``tau``, ``tau_m``, ``tau_v`` or ``eps`` is not finite and
            > 0, or ``weight_decay`` is not finite and >= 0.
    """

    tau: float
    tau_m: float
    tau_v: float
    eps: float = 1e-8
    weight_decay: float = 1e-2

    def __post_init__(self) -> None:
        for name in ("tau", "tau_m", "tau_v", "eps"):
            _check_positive("AdamW", name, getattr(self, name))
        if not 0 <= self.weight_decay < math.inf:
            raise ValueError(
                f"AdamW.weight_decay must be finite and >= 0, got {self.weight_decay!r}"
            )

    @override
    def bind(self, params: Iterable[torch.Tensor]) -> BoundOptimizer:
        """Return the optimizer bound to ``params``, with the moments and the
        masses at zero."""
        return _BoundAdamW(list(params), self)


class _BoundAdamW(BoundOptimizer):
    def __init__(self, params: list[torch.Tensor], config: AdamW) -> None:
        self._params = params
        self._config = config
        self._m = [torch.zeros_like(p) for p in params]
        self._v = [torch.zeros_like(p) for p in params]
        self._c_m = 0.0
        self._c_v = 0.0

    @override
    def step_(self, grads: Sequence[torch.Tensor], elapsed: float) -> None:
        if elapsed == 0:
            return  # the flow of no time; before the first step, c = 0
        config = self._config
        a_m = -math.expm1(-elapsed / config.tau_m)  # 1 - exp(-dt / tau_m)
        a_v = -math.expm1(-elapsed / config.tau_v)
        self._c_m += a_m * (1 - self._c_m)  # tau_m dc_m/dt = 1 - c_m
        self._c_v += a_v * (1 - self._c_v)
        eta = elapsed / config.tau
        sqrt_c_v = math.sqrt(self._c_v)
        with torch.no_grad():
            for p, g, m, v in zip(self._params, grads, self._m, self._v, strict=True):
                p.mul_(1 - eta * config.weight_decay)  # the decay, first
                m.lerp_(g, a_m)  # tau_m dm/dt = g - m
                v.lerp_(g * g, a_v)  # tau_v dv/dt = g^2 - v
                # theta -= eta * m_hat / (sqrt(v_hat) + eps)
                denom = (v.sqrt() / sqrt_c_v).add_(config.eps)
                p.addcdiv_(m, denom, value=-eta / self._c_m)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "m": list(self._m),
            "v": list(self._v),
            "c_m": self._c_m,
            "c_v": self._c_v,
        }

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        with torch.no_grad():
            for mine, saved in zip(self._m, state["m"], strict=True):
                mine.copy_(saved)
            for mine, saved in zip(self._v, state["v"], strict=True):
                mine.copy_(saved)
        self._c_m = float(state["c_m"])
        self._c_v = float(state["c_v"])
