r"""The stateless computations of PC-ALM that the nodes of ``mujou.pc`` are made
of, for writing nodes of your own (such as one with several inputs).

A node ``i`` predicts ``mu = f(h_in; theta)`` from the activity ``h_in`` of the
previous node, and keeps its activity ``h`` and the dual ``lam`` of the constraint
``h = mu``. Its energy is ``E = 0.5 * ||h - mu + lam||^2`` (``r = h - mu``,
``delta = r + lam``).

- ``vjp`` gives the gradients of ``E`` with respect to ``h_in`` and ``theta`` (one
  vector-Jacobian product inside the node). The node sends the one of ``h_in`` to
  the previous node, which adds the values it receives into its ``g``.
- ``loss_vjp`` does the same for an output node, whose energy is the loss.
- ``relax_`` moves ``h`` and ``lam`` by ``dt`` seconds; ``learn_`` moves ``theta``.

Every ``g`` here is a gradient of an energy; the update subtracts it. Gradients are
summed over the batch, and the ``1 / B`` is applied to ``theta`` only (in
``learn_``), so ``lam`` holds the credit of each sample.
"""

import math
from collections.abc import Callable, Sequence

import torch

from mujou.pc._messages import rows
from mujou.pc._time_constants import TimeConstants

__all__ = ["learn_", "loss_vjp", "relax_", "vjp"]


def vjp(
    module: Callable[[torch.Tensor], torch.Tensor],
    h_in: torch.Tensor,
    delta: Callable[[torch.Tensor], torch.Tensor],
    params: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    r"""Predict ``mu = module(h_in)`` and the gradients of the node's energy.

    With the activity $h$ held fixed, the energy
    $E = \tfrac12\|h - \mu + \lambda\|^2$ has the gradients

    $$
    g_{\mathrm{in}} = \frac{\partial E}{\partial h_{\mathrm{in}}}
        = -J_h^\top \delta,\qquad
    g_\theta = \frac{\partial E}{\partial \theta} = -J_\theta^\top \delta,
    $$

    where $\delta = h - \mu + \lambda$ and $J$ are the Jacobians of ``module``
    at ``h_in``. Both are summed over the batch. ``h_in`` is detached first, so
    no graph reaches the node that sent it.

    Args:
        module: The function ``f`` of the node, ``(B, *in) -> (B, *out)``. It must
            treat the samples independently.
        h_in: The activity of the previous node, ``(B, *in)``.
        delta: Makes $\delta$ of the node, ``(B, *out)``, from ``mu`` (detached,
            without gradients), such as ``lambda mu: h - mu + lam``.
        params: The parameters to get the gradients of, usually those of
            ``module`` that require gradients.

    Returns:
        ``(mu, g_in, g_theta)``: ``mu`` (``(B, *out)``, detached), ``g_in``
        (``(B, *in)``) and ``g_theta`` (one tensor per parameter, in the order of
        ``params``; a parameter that ``module`` does not use gets zeros).

    Raises:
        ValueError: ``delta`` does not make a tensor of the shape of ``mu``.
    """

    def energy(mu: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            d = delta(mu.detach())
        # A broadcast would silently give gradients of another shape.
        if d.shape != mu.shape:
            raise ValueError(
                f"delta must be of the shape of mu = module(h_in), "
                f"{tuple(mu.shape)}, got {tuple(d.shape)}"
            )
        # E with h held fixed, to the first order in mu: its gradient is -delta.
        return (mu * -d).sum()

    return _grad(module, h_in, energy, params)


def loss_vjp(
    module: Callable[[torch.Tensor], torch.Tensor],
    h_in: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    params: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    r"""Predict ``y_hat = module(h_in)`` and the gradients of the masked loss.

    The energy of an output node is the loss of the samples that have a target,

    $$
    \ell = \sum_b \mathrm{mask}_b\, \ell_b(\hat y_b, y_b),\qquad
    g_{\mathrm{in}} = \frac{\partial \ell}{\partial h_{\mathrm{in}}},\qquad
    g_\theta = \frac{\partial \ell}{\partial \theta}.
    $$

    A sample whose ``mask`` is False adds nothing, whatever its ``y`` is (NaN
    too): its ``y`` is replaced by zeros before ``loss`` sees it, and its
    ``g_in`` is $J^\top 0$, zero as long as its ``y_hat`` is finite (NaN where
    it is not). With every ``mask`` False, ``g_in`` is thus zeros. ``h_in`` is
    detached first.

    Args:
        module: The function ``f`` of the node, ``(B, *in) -> (B, *out)``.
        h_in: The activity of the previous node, ``(B, *in)``.
        y: The target, ``(B, *target)``.
        mask: Whether each sample has a target, ``(B,)`` bool.
        loss: The loss of each sample, ``(y_hat, y) -> (B,)``, such as
            ``mse`` or ``cross_entropy``.
        params: The parameters to get the gradients of.

    Returns:
        ``(y_hat, g_in, g_theta)`` as ``vjp`` returns ``(mu, g_in, g_theta)``.

    Raises:
        ValueError: ``loss`` does not return ``(B,)``, as a loss that averages
            (``nn.MSELoss()``) does.
    """
    # A masked-out y may be anything; a NaN in it would reach the gradients
    # through 0 * NaN, so it never reaches the loss. Its mask is False, so the
    # value put in its place does not matter.
    y = torch.where(rows(mask, y), y, torch.zeros_like(y))

    def energy(y_hat: torch.Tensor) -> torch.Tensor:
        ell_b = loss(y_hat, y)
        if ell_b.shape != mask.shape:
            raise ValueError(
                f"loss must return the loss of each sample, of shape "
                f"{tuple(mask.shape)}, got {tuple(ell_b.shape)}; a loss that "
                "averages, such as nn.MSELoss(), does not (use pc.mse or "
                "pc.cross_entropy)"
            )
        return torch.where(mask, ell_b, 0.0).sum()

    return _grad(module, h_in, energy, params)


def _grad(
    module: Callable[[torch.Tensor], torch.Tensor],
    h_in: torch.Tensor,
    energy: Callable[[torch.Tensor], torch.Tensor],
    params: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """``module(h_in)`` and the gradients of ``energy`` of it with respect to
    ``h_in`` and ``params``."""
    with torch.enable_grad():
        h_in = h_in.detach().requires_grad_()
        out = module(h_in)
        # The gradient of this node's own energy: h_in is a detached leaf, so
        # it stays inside the node.
        g_in, *g_theta = torch.autograd.grad(  # mujou: local-autograd
            energy(out), [h_in, *params], materialize_grads=True
        )
    return out.detach(), g_in, g_theta


def relax_(
    h: torch.Tensor,
    lam: torch.Tensor,
    mu: torch.Tensor,
    g: torch.Tensor,
    dt: float,
    tau: TimeConstants,
) -> None:
    r"""Move ``h`` and ``lam`` in place by one explicit Euler step of ``dt``
    seconds of

    $$
    \tau_h \dot h = -(r + \lambda + g),\qquad
    \tau_\lambda \dot\lambda = r,\qquad r = h - \mu.
    $$

    The step moves ``h`` first and then ``lam`` with the ``r`` of the moved
    ``h``. ``mu`` and ``g`` are held fixed. The step is not split: ``g`` comes
    from the next nodes and stays as it was over ``dt``, so the coupling between
    the nodes is explicit however finely the step is split. Keep ``dt`` short
    instead (``Layer`` caps it at ``max_dt``). ``tau.lam = inf`` leaves ``lam``
    exactly as it is.

    Args:
        h: The activity of the node, ``(B, *out)``.
        lam: The dual of the node's activity, ``(B, *out)``.
        mu: The prediction ``f(h_in)`` of the node, ``(B, *out)``.
        g: The sum of the gradients the next nodes sent, ``(B, *out)``.
        dt: The time to move by, in seconds.
        tau: The time constants; ``tau.h`` and ``tau.lam`` are used.
    """
    with torch.no_grad():
        h.sub_((dt / tau.h) * (h - mu + lam + g))  # tau_h dh/dt = -(r + lam + g)
        if math.isfinite(tau.lam):  # alpha = 0 (PC): lam stays exactly as it is
            lam.add_(h - mu, alpha=dt / tau.lam)  # tau_lam dlam/dt = r


def learn_(
    params: Sequence[torch.Tensor],
    g_theta: Sequence[torch.Tensor],
    elapsed: float,
    tau: TimeConstants,
    batch: int,
) -> None:
    """Move ``params`` in place by ``elapsed`` seconds of
    ``tau_theta * dtheta/dt = -g_theta / B``.

    That is one gradient step on the mean loss of the batch with the learning
    rate ``elapsed / tau.theta``: ``p -= (elapsed / tau.theta) * g / batch``.
    ``tau.theta = inf`` makes the rate 0, which leaves ``params`` as they are
    while the gradients are finite; the nodes of ``mujou.pc`` do not call it
    then (``LearnClock.of`` never learns).

    Args:
        params: The parameters of the node.
        g_theta: Their gradients summed over the batch, in the same order, as
            ``vjp`` or ``loss_vjp`` returns them.
        elapsed: The time since the last update of ``params``, in seconds.
        tau: The time constants; ``tau.theta`` is used.
        batch: The batch size ``B``.
    """
    rate = elapsed / tau.theta / batch
    with torch.no_grad():
        for p, g in zip(params, g_theta, strict=True):
            p.sub_(g, alpha=rate)  # tau_theta dtheta/dt = -g_theta / B
