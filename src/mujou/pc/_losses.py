"""Per-sample losses for an output node.

A loss takes the prediction ``y_hat`` and the target ``y`` and returns the loss of
each sample, ``(B,)``. The node sums them; it never averages, so that the gradient
it sends to the previous node is the credit of each sample (``specs/pc-module.md``).
"""

import torch
import torch.nn.functional as F


def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ``0.5 * ||y_b - y_hat_b||^2`` of each sample ``b``.

    The squared error is summed over every dimension but the first. Unlike
    ``nn.MSELoss``, there is no mean and the half is there, so the gradient with
    respect to ``y_hat`` is ``y_hat - y``.

    Args:
        y_hat: The prediction, ``(B, *out)``.
        y: The target, of the same shape as ``y_hat``.

    Returns:
        The loss of each sample, ``(B,)``.

    Raises:
        ValueError: The shapes differ (a broadcast would silently give ``(B, B)``).
    """
    if y_hat.shape != y.shape:
        raise ValueError(
            f"mse needs y_hat and y of the same shape, got {tuple(y_hat.shape)} "
            f"and {tuple(y.shape)}"
        )
    return 0.5 * (y - y_hat).square().reshape(y.shape[0], -1).sum(dim=1)


def cross_entropy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return the cross entropy of each sample, as
    ``F.cross_entropy(y_hat, y, reduction="none")``.

    Args:
        y_hat: The logits, ``(B, C)``.
        y: The class of each sample, ``(B,)`` int64, or the probability of each
            class, ``(B, C)``.

    Returns:
        The loss of each sample, ``(B,)``.
    """
    return F.cross_entropy(y_hat, y, reduction="none")
