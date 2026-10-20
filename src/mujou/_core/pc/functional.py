"""The stateless computations of PC-ALM that the nodes of ``mujou.pc`` are made of.

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
``learn_``), so ``lam`` holds the credit of each sample (``specs/pc-module.md``).
"""

import math
from collections.abc import Callable, Sequence

import torch

from mujou._core.pc.time_constants import TimeConstants


def vjp(
    module: Callable[[torch.Tensor], torch.Tensor],
    h_in: torch.Tensor,
    delta: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
    params: Sequence[torch.Tensor] = (),
    *,
    wrt_input: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
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
        delta: $\delta$ of the node, ``(B, *out)``, or a function that makes it
            from ``mu`` (detached), such as ``lambda mu: h - mu + lam``, for a
            $\delta$ that depends on the prediction of this call.
        params: The parameters to get the gradients of, usually those of
            ``module`` that require gradients.
        wrt_input: Whether to get the gradient of ``h_in``.

    Returns:
        ``(mu, g_in, g_theta)``: ``mu`` (``(B, *out)``, detached), ``g_in``
        (``(B, *in)``, or ``None`` when ``wrt_input`` is False) and ``g_theta``
        (one tensor per parameter, in the order of ``params``; a parameter that
        ``module`` does not use gets zeros).

    Raises:
        ValueError: ``delta`` is not of the shape of ``mu``.
    """
    params = list(params)
    if not wrt_input and not params:
        with torch.no_grad():
            mu = module(h_in)
            _check_delta(mu, delta if isinstance(delta, torch.Tensor) else delta(mu))
        return mu, None, []
    with torch.enable_grad():
        h_in = h_in.detach().requires_grad_(wrt_input)
        mu = module(h_in)
        if not isinstance(delta, torch.Tensor):
            with torch.no_grad():
                delta = delta(mu.detach())
        _check_delta(mu, delta)
        inputs = [h_in, *params] if wrt_input else params
        # The VJP of this node's f only: h_in is a detached leaf, so the gradients
        # stay inside the node.
        grads = torch.autograd.grad(  # mujou: local-autograd
            mu, inputs, grad_outputs=-delta, materialize_grads=True
        )
    g_in = grads[0] if wrt_input else None
    g_theta = list(grads[1:] if wrt_input else grads)
    return mu.detach(), g_in, g_theta


def _check_delta(mu: torch.Tensor, delta: torch.Tensor) -> None:
    # A broadcast would silently give gradients of another shape.
    if delta.shape != mu.shape:
        raise ValueError(
            f"delta must be of the shape of mu = module(h_in), {tuple(mu.shape)}, "
            f"got {tuple(delta.shape)}"
        )


def loss_vjp(
    module: Callable[[torch.Tensor], torch.Tensor],
    h_in: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    params: Sequence[torch.Tensor] = (),
    *,
    wrt_input: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
    r"""Predict ``y_hat = module(h_in)`` and the gradients of the masked loss.

    The energy of an output node is the loss of the samples that have a target,

    $$
    \ell = \sum_b \mathrm{mask}_b\, \ell_b(\hat y_b, y_b),\qquad
    g_{\mathrm{in}} = \frac{\partial \ell}{\partial h_{\mathrm{in}}},\qquad
    g_\theta = \frac{\partial \ell}{\partial \theta}.
    $$

    A sample whose ``mask`` is False adds nothing, whatever its ``y`` is (NaN
    too): its ``y`` is replaced by zeros before ``loss`` sees it, and its
    ``g_in`` is zero. ``h_in`` is detached first.

    Args:
        module: The function ``f`` of the node, ``(B, *in) -> (B, *out)``.
        h_in: The activity of the previous node, ``(B, *in)``.
        y: The target, ``(B, *target)``.
        mask: Whether each sample has a target, ``(B,)`` bool.
        loss: The loss of each sample, ``(y_hat, y) -> (B,)``, such as
            ``mse`` or ``cross_entropy``.
        params: The parameters to get the gradients of.
        wrt_input: Whether to get the gradient of ``h_in``.

    Returns:
        ``(y_hat, g_in, g_theta)`` as ``vjp`` returns ``(mu, g_in, g_theta)``.

    Raises:
        ValueError: ``loss`` does not return ``(B,)``.
    """
    params = list(params)
    # A masked-out y may be anything; a NaN in it would reach the gradients
    # through 0 * NaN, so it never reaches the loss. Its mask is False, so the
    # value put in its place does not matter.
    y = torch.where(rows(mask, y), y, torch.zeros_like(y))
    if not wrt_input and not params:
        with torch.no_grad():
            y_hat = module(h_in)
            _check_loss(loss(y_hat, y), mask)
        return y_hat, None, []
    with torch.enable_grad():
        h_in = h_in.detach().requires_grad_(wrt_input)
        y_hat = module(h_in)
        ell_b = loss(y_hat, y)  # (B,)
        _check_loss(ell_b, mask)
        ell = torch.where(mask, ell_b, 0.0).sum()
        inputs = [h_in, *params] if wrt_input else params
        # The gradient of this node's loss with respect to its own input and
        # parameters: h_in is a detached leaf, so it stays inside the node.
        grads = torch.autograd.grad(  # mujou: local-autograd
            ell, inputs, materialize_grads=True
        )
    g_in = grads[0] if wrt_input else None
    g_theta = list(grads[1:] if wrt_input else grads)
    return y_hat.detach(), g_in, g_theta


def _check_loss(ell_b: torch.Tensor, mask: torch.Tensor) -> None:
    if ell_b.shape != mask.shape:
        raise ValueError(
            f"loss must return the loss of each sample, of shape "
            f"{tuple(mask.shape)}, got {tuple(ell_b.shape)}"
        )


def rows(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Return the ``(B,)`` ``mask`` shaped to select the rows of ``like``
    (``(B, *shape)``) in ``torch.where``."""
    return mask.reshape(-1, *([1] * (like.dim() - 1)))


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
        if math.isfinite(tau.lam):  # 0 * inf would make NaN of an infinite r
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
    ``tau.theta = inf`` leaves ``params`` exactly as they are.

    Args:
        params: The parameters of the node.
        g_theta: Their gradients summed over the batch, in the same order, as
            ``vjp`` or ``loss_vjp`` returns them.
        elapsed: The time since the last update of ``params``, in seconds.
        tau: The time constants; ``tau.theta`` is used.
        batch: The batch size ``B``.

    Raises:
        ValueError: ``params`` and ``g_theta`` differ in length (nothing is moved).
    """
    if len(params) != len(g_theta):
        raise ValueError(
            f"learn_ needs one gradient per parameter, got {len(params)} "
            f"parameters and {len(g_theta)} gradients"
        )
    rate = elapsed / tau.theta / batch
    with torch.no_grad():
        for p, g in zip(params, g_theta, strict=True):
            p.sub_(g, alpha=rate)  # tau_theta dtheta/dt = -g_theta / B
