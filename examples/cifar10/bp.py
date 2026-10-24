"""The baseline: the same ResNet trained by backpropagation (BP) with plain
PyTorch on the GPU, on the same batches in the same order, with the same
optimizer."""

import time
from collections.abc import Callable

import torch
from data import CIFAR10, batches
from metrics import Epoch, correct, evaluate
from torch import nn

import mujou.pc as pc


def train(
    model: nn.Module,
    data: CIFAR10,
    *,
    epochs: int,
    batch: int,
    optimizer: pc.Optimizer,
    elapsed: float,
    generator: Callable[[int], torch.Generator],
    device: torch.device,
) -> list[Epoch]:
    """Train ``model`` on ``device`` on the mean cross entropy, the batches of
    epoch ``e`` shuffled and augmented by ``generator(e)``.

    ``optimizer`` is the one the PC-ALM nodes use, bound to the parameters of
    the whole model, and steps by ``elapsed`` seconds per batch: the time the
    nodes learn for, once per batch. ``torch.optim`` is not used: on the
    free-threaded Python, its step loads a module that turns the GIL on for
    the rest of the process.

    The loss sums over the samples with a label and divides by ``batch``, as the
    PC-ALM nodes do: for a full batch it is the mean.
    """
    params = list(model.parameters())
    bound = optimizer.bind(params)
    results: list[Epoch] = []
    for epoch in range(epochs):
        right = total = 0
        start = time.perf_counter()
        for x, y, mask in batches(data.train_x, data.train_y, batch, generator(epoch)):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            y_hat = model(x)
            right += correct(y_hat.detach(), y, mask)  # before learning from it
            total += int(mask.sum())
            ce = nn.functional.cross_entropy(y_hat, y, reduction="none")
            loss = torch.where(mask, ce, 0.0).sum() / batch
            bound.step_(torch.autograd.grad(loss, params), elapsed)
        seconds = time.perf_counter() - start
        test = evaluate(model, batches(data.test_x, data.test_y, batch), device)
        results.append(Epoch(right / total, test, seconds))
        print(
            f"BP epoch {epoch + 1}: prequential {right / total:.2%}, "
            f"test {test:.2%}, {seconds:.1f} s"
        )
    return results
