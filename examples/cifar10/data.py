"""CIFAR-10: download the binary version once, read it with numpy, and cut
augmented batches.

Nothing here depends on mujou: the batches are plain tensors.
"""

import tarfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
TRAIN = [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)]
TEST = ["cifar-10-batches-bin/test_batch.bin"]
RECORD = 1 + 3 * 32 * 32  # a label byte, then the red, green and blue planes
PAD = 4  # the padding of the random crops


@dataclass(frozen=True)
class CIFAR10:
    """Images ``(N, 3, 32, 32)`` float32, standardized per channel with the mean
    and standard deviation of the training pixels, and labels ``(N,)`` int64."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def _fetch(root: Path) -> Path:
    """Return the path of the archive in ``root``, downloading it first if it is
    not there."""
    path = root / URL.rsplit("/", 1)[1]
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, partial)
        partial.rename(path)  # only a complete download gets the name
    return path


def _read(archive: tarfile.TarFile, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read the records of the files ``names`` of ``archive``: images as
    ``(N, 3, 32, 32)`` uint8 and labels as ``(N,)`` uint8."""
    parts: list[np.ndarray] = []
    for name in names:
        member = archive.extractfile(name)  # read in memory, nothing extracted
        if member is None:
            raise ValueError(f"{name} is not a file of the archive")
        parts.append(np.frombuffer(member.read(), dtype=np.uint8).reshape(-1, RECORD))
    records = np.concatenate(parts)
    return records[:, 1:].reshape(-1, 3, 32, 32), records[:, 0]


def load(root: Path) -> CIFAR10:
    """Load CIFAR-10 from ``root``, downloading the archive (about 160 MB) if
    needed."""
    with tarfile.open(_fetch(root)) as archive:
        train_x, train_y = _read(archive, TRAIN)
        test_x, test_y = _read(archive, TEST)
    train = torch.from_numpy(train_x.astype(np.float32))
    test = torch.from_numpy(test_x.astype(np.float32))
    mean = train.mean(dim=(0, 2, 3), keepdim=True)
    std = train.std(dim=(0, 2, 3), keepdim=True)

    def labels(y: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(y.astype(np.int64))

    return CIFAR10(
        (train - mean) / std, labels(train_y), (test - mean) / std, labels(test_y)
    )


def augment(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Flip each image of ``x`` ``(B, 3, 32, 32)`` horizontally with probability
    1/2, and crop it at random from the image padded with ``PAD`` zeros."""
    b, _, height, width = x.shape
    flip = torch.rand(b, generator=generator) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    padded = torch.nn.functional.pad(x, (PAD, PAD, PAD, PAD))
    top, left = torch.randint(0, 2 * PAD + 1, (2, b), generator=generator)
    rows = (top[:, None] + torch.arange(height))[:, None, :, None]  # (B, 1, H, 1)
    cols = (left[:, None] + torch.arange(width))[:, None, None, :]  # (B, 1, 1, W)
    samples = torch.arange(b)[:, None, None, None]
    channels = torch.arange(x.shape[1])[None, :, None, None]
    return padded[samples, channels, rows, cols]


def batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch: int,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Cut the samples into batches ``(x, y, mask)`` of ``batch``. With
    ``generator``, the samples are shuffled and augmented by it; without one,
    they come in order as they are. The last batch is padded with zeros whose
    ``mask`` is False."""
    n = len(labels)
    order = (
        torch.arange(n) if generator is None else torch.randperm(n, generator=generator)
    )
    for start in range(0, n, batch):
        index = order[start : start + batch]
        pad = batch - len(index)
        x, y = images[index], labels[index]
        if generator is not None:
            x = augment(x, generator)
        mask = torch.ones(batch, dtype=torch.bool)
        if pad:
            x = torch.cat([x, x.new_zeros(pad, *x.shape[1:])])
            y = torch.cat([y, y.new_zeros(pad)])
            mask[-pad:] = False
        yield x, y, mask
