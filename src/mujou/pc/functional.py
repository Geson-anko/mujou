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
- ``relax_`` moves ``h`` and ``lam`` by ``dt`` seconds (``h`` as a relaxer of
  ``mujou.pc.relax`` says: ``pc.Euler``, ``pc.HeavyBall``); an optimizer of
  ``mujou.pc.optim`` (``pc.SGD``, ``pc.AdamW``) moves ``theta``.
- ``mse`` and ``cross_entropy`` are losses of each sample, for ``loss_vjp``
  (``MSELoss`` and ``CrossEntropyLoss`` hold their ``batch``).

Every ``g`` here is a gradient of an energy; the update subtracts it. Gradients are
summed over the batch, and the ``1 / B`` (``B`` the number of samples) is applied
to ``theta`` only (the node divides ``g_theta`` by ``B`` before it gives it to the
optimizer), so ``lam`` holds the credit of each sample.
The shapes start with the batch, ``batch`` (see ``activity_spec``).
"""

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from mujou.pc.messages import batch_shape, rows
from mujou.pc.relax import BoundRelaxer
from mujou.pc.time_constants import TimeConstants
from mujou.types import Size

__all__ = ["cross_entropy", "loss_vjp", "mse", "relax_", "vjp"]


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
        module: The function ``f`` of the node, ``(*batch, *in) ->
            (*batch, *out)``. It must treat the samples independently.
        h_in: The activity of the previous node, ``(*batch, *in)``.
        delta: Makes $\delta$ of the node, ``(*batch, *out)``, from ``mu`` (detached,
            without gradients), such as ``lambda mu: h - mu + lam``.
        params: The parameters to get the gradients of, usually those of
            ``module`` that require gradients.

    Returns:
        ``(mu, g_in, g_theta)``: ``mu`` (``(*batch, *out)``, detached), ``g_in``
        (``(*batch, *in)``) and ``g_theta`` (one tensor per parameter, in the order of
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
        module: The function ``f`` of the node, ``(*batch, *in) ->
            (*batch, *out)``.
        h_in: The activity of the previous node, ``(*batch, *in)``.
        y: The target, ``(*batch, *target)``.
        mask: Whether each sample has a target, ``batch`` bool.
        loss: The loss of each sample, ``(y_hat, y) -> batch``, such as
            ``MSELoss(batch)`` or ``CrossEntropyLoss(batch)``.
        params: The parameters to get the gradients of.

    Returns:
        ``(y_hat, g_in, g_theta)`` as ``vjp`` returns ``(mu, g_in, g_theta)``.

    Raises:
        ValueError: ``loss`` does not return the shape of ``mask``, as a loss
            that averages over the batch (``nn.MSELoss()``) does.
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
                "averages, such as nn.MSELoss(), does not (use pc.MSELoss or "
                "pc.CrossEntropyLoss with the batch)"
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
    relaxer: BoundRelaxer | None = None,
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

    With a ``relaxer`` (``pc.Euler().bind(h)``, ``pc.HeavyBall(...).bind(h)`` or
    one of your own), ``h`` moves as it says along the force
    $F = r + \lambda + g$ instead of by the plain step: ``relaxer.step_(h, F, dt,
    tau.h)``. $\lambda$ follows the moved ``h`` as before. The relaxer is the
    node's: it holds the state, and the node calls its ``reset_`` where the force
    jumps.

    Args:
        h: The activity of the node, ``(*batch, *out)``.
        lam: The dual of the node's activity, ``(*batch, *out)``.
        mu: The prediction ``f(h_in)`` of the node, ``(*batch, *out)``.
        g: The sum of the gradients the next nodes sent, ``(*batch, *out)``.
        dt: The time to move by, in seconds.
        tau: The time constants; ``tau.h`` and ``tau.lam`` are used.
        relaxer: How ``h`` moves, bound to ``h``; ``None`` is the plain step.
    """
    with torch.no_grad():
        force = h - mu + lam + g  # tau_h dh/dt = -(r + lam + g)
        if relaxer is None:
            h.sub_((dt / tau.h) * force)
        else:
            relaxer.step_(h, force, dt, tau.h)
        if math.isfinite(tau.lam):  # alpha = 0 (PC): lam stays exactly as it is
            lam.add_(h - mu, alpha=dt / tau.lam)  # tau_lam dlam/dt = r


def mse(y_hat: torch.Tensor, y: torch.Tensor, batch: Size = ()) -> torch.Tensor:
    """Return the mean squared error of each sample of ``batch``.

    The squared error is averaged over the dimensions of a sample, as
    ``nn.MSELoss`` averages it, but not over the batch: the mean of the result
    is ``nn.MSELoss()(y_hat, y)``, and with the ``1 / B`` the node applies before
    its optimizer, ``theta`` follows its gradient.

    With ``d`` values in a sample, its gradient with respect to ``y_hat`` is
    ``(2 / d) * (y_hat - y)``: that of ``0.5 * ||y - y_hat||^2`` for ``d = 2``,
    twice it for ``d = 1``. The larger it is, the more strongly the output
    pulls on the activities, and the shorter the period that keeps their
    explicit steps stable; for a scalar target, a loss of your own such as
    ``0.5 * mse(y_hat, y, batch)`` keeps the pull of the half.

    Args:
        y_hat: The prediction, ``(*batch, *out)``.
        y: The target, of the same shape as ``y_hat``.
        batch: The batch that the shapes start with; ``()`` for one sample.

    Returns:
        The loss of each sample, ``batch``.

    Raises:
        ValueError: The shapes differ (a broadcast would silently give
            another shape), or they do not start with ``batch``.
    """
    if y_hat.shape != y.shape:
        raise ValueError(
            f"mse needs y_hat and y of the same shape, got {tuple(y_hat.shape)} "
            f"and {tuple(y.shape)}"
        )
    batch = _starting(batch, y_hat, y)
    return (y - y_hat).square().reshape(*batch, -1).mean(dim=-1)


def cross_entropy(
    y_hat: torch.Tensor, y: torch.Tensor, batch: Size = ()
) -> torch.Tensor:
    """Return the cross entropy of each sample of ``batch``.

    The classes are the dimension right after the batch, as in
    ``F.cross_entropy`` (``(N, C)`` and ``(C,)`` alike). A sample with more
    dimensions, such as one class for each pixel (``(*batch, C, H, W)``), has
    its cross entropy averaged over them, as ``nn.CrossEntropyLoss`` does, but
    not over the batch: the mean of the result is
    ``nn.CrossEntropyLoss()(y_hat, y)``. A class of ``-100`` (the
    ``ignore_index`` of torch) adds zero but still counts in that mean; leave a
    sample out with ``mask`` instead.

    Args:
        y_hat: The logits, ``(*batch, C, *rest)``.
        y: The class of each sample, ``(*batch, *rest)`` int64, or the
            probability of each class, of the shape of ``y_hat``.
        batch: The batch that the shapes start with; ``()`` for one sample.

    Returns:
        The loss of each sample, ``batch``.

    Raises:
        ValueError: ``y_hat`` or ``y`` does not start with ``batch``.
    """
    batch = _starting(batch, y_hat, y)
    n, k = math.prod(batch), len(batch)
    # F.cross_entropy takes the batch as one dimension, before the classes.
    ell = F.cross_entropy(
        y_hat.reshape(n, *y_hat.shape[k:]),
        y.reshape(n, *y.shape[k:]),
        reduction="none",
    )
    return ell.reshape(*batch, -1).mean(dim=-1)


def _starting(batch: Size, *tensors: torch.Tensor) -> tuple[int, ...]:
    """``batch`` as a shape, which each of ``tensors`` must start with: a
    reshape would otherwise mix the samples whenever the sizes divide."""
    batch = batch_shape(batch)
    for t in tensors:
        if t.shape[: len(batch)] != batch:
            raise ValueError(
                f"the tensors must start with the batch {batch}, got the shape "
                f"{tuple(t.shape)}"
            )
    return batch
