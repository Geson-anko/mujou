"""The losses of each sample for an output node.

A loss takes the prediction ``y_hat`` and the target ``y`` and returns the loss of
each sample, of the shape of the batch. The node sums them; it never averages over
the batch, so that the gradient it sends to the previous node is the credit of
each sample (``specs/pc.md``). Each class holds its ``batch`` for the functions of
``functional`` of the same name.
"""

import torch

from mujou.pc.functional import cross_entropy, mse
from mujou.types import Size

__all__ = [
    "MSELoss",
    "CrossEntropyLoss",
]


class MSELoss:
    """The mean squared error of each sample: ``functional.mse`` with ``batch``.

    Example::

        loss = pc.losses.MSELoss()  # one sample: y_hat (out,) -> ()
        loss = pc.losses.MSELoss(64)  # y_hat (64, out) -> (64,)

    Args:
        batch: The batch that the shapes start with; ``()`` for one sample.
    """

    def __init__(self, batch: Size = ()) -> None:
        self._batch = batch

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mse(y_hat, y, self._batch)


class CrossEntropyLoss:
    """The cross entropy of each sample: ``functional.cross_entropy`` with
    ``batch``.

    Example::

        loss = pc.losses.CrossEntropyLoss()  # one sample: y_hat (C,), y () -> ()
        loss = pc.losses.CrossEntropyLoss(64)  # y_hat (64, C), y (64,) -> (64,)

    Args:
        batch: The batch that the shapes start with; ``()`` for one sample.
    """

    def __init__(self, batch: Size = ()) -> None:
        self._batch = batch

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cross_entropy(y_hat, y, self._batch)
