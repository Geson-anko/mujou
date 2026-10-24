"""What the programs measure, the same for PC-ALM and BP."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Epoch:
    """What one epoch measured: the prequential accuracy (each batch predicted
    before it is learned from), the test accuracy after the epoch, and the real
    seconds the training took."""

    prequential: float
    test: float
    seconds: float


def correct(y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> int:
    """Count the samples in ``mask`` whose class ``y_hat`` predicts right."""
    return int(((y_hat.argmax(dim=1) == y) & mask).sum())


def evaluate(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    """Return the accuracy of ``model`` on ``device`` on the batches
    ``(x, y, mask)``."""
    right = total = 0
    with torch.no_grad():
        for x, y, mask in batches:
            y_hat = model(x.to(device)).cpu()
            right += correct(y_hat, y, mask)
            total += int(mask.sum())
    return right / total
