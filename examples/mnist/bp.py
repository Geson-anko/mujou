"""The baseline: the same network trained by backpropagation (BP) with plain
PyTorch SGD, on the same batches in the same order."""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from data import MNIST, batches
from torch import nn

import mujou.pc as pc


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


def evaluate(model: nn.Module, samples: Iterable[pc.Sample]) -> float:
    """Return the accuracy of ``model`` on ``samples``."""
    right = total = 0
    with torch.no_grad():
        for sample in samples:
            right += correct(model(sample["x"]), sample["y"], sample["mask"])
            total += int(sample["mask"].sum())
    return right / total


def train(
    model: nn.Module,
    data: MNIST,
    *,
    epochs: int,
    batch: int,
    lr: float,
    generator: Callable[[int], torch.Generator],
) -> list[Epoch]:
    """Train ``model`` with SGD at the learning rate ``lr`` per batch on the mean
    cross entropy, the batches of epoch ``e`` shuffled by ``generator(e)``.

    The step is written out instead of using ``torch.optim.SGD``: on the
    free-threaded Python, the step of a ``torch.optim`` optimizer loads a module
    that turns the GIL on for the rest of the process.

    The loss sums over the samples with a label and divides by ``batch``, as the
    PC-ALM nodes do (``sum_dt / tau.theta * g_theta / B``): for a full batch it is
    the mean.
    """
    params = list(model.parameters())
    results: list[Epoch] = []
    for epoch in range(epochs):
        right = total = 0
        start = time.perf_counter()
        for sample in batches(data.train_x, data.train_y, batch, generator(epoch)):
            x, y, mask = sample["x"], sample["y"], sample["mask"]
            y_hat = model(x)
            right += correct(y_hat.detach(), y, mask)  # before learning from it
            total += int(mask.sum())
            loss = torch.where(mask, pc.cross_entropy(y_hat, y), 0.0).sum() / batch
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, g in zip(params, grads, strict=True):
                    p -= lr * g
        seconds = time.perf_counter() - start
        test = evaluate(model, batches(data.test_x, data.test_y, batch))
        results.append(Epoch(right / total, test, seconds))
    return results
