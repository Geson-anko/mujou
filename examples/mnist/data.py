"""MNIST: download the idx files once, read them with numpy, and cut batches."""

import gzip
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import mujou.pc as pc

URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
IMAGES, LABELS = 2051, 2049  # the magic numbers of the idx files


@dataclass(frozen=True)
class MNIST:
    """Images ``(N, 784)`` float32, standardized with the mean and standard
    deviation of the training pixels, and labels ``(N,)`` int64."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor


def _fetch(root: Path, name: str) -> Path:
    """Return the path of the file ``name`` in ``root``, downloading it first if it
    is not there."""
    path = root / name
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL + name}")
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(URL + name, partial)
        partial.rename(path)  # only a complete download gets the name
    return path


def _read(path: Path, magic: int) -> np.ndarray:
    """Read an idx file: images as ``(N, rows * cols)``, labels as ``(N,)``."""
    data = gzip.decompress(path.read_bytes())
    found = int.from_bytes(data[:4], "big")
    if found != magic:
        raise ValueError(f"{path} is not an idx file of magic {magic}, got {found}")
    if magic == LABELS:
        return np.frombuffer(data, dtype=np.uint8, offset=8)
    n, rows, cols = (int.from_bytes(data[i : i + 4], "big") for i in (4, 8, 12))
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows * cols)


def load(root: Path) -> MNIST:
    """Load MNIST from ``root``, downloading the 4 files (about 11 MB) if needed."""
    arrays = {
        key: _read(_fetch(root, name), LABELS if key.endswith("_y") else IMAGES)
        for key, name in FILES.items()
    }
    train = torch.from_numpy(arrays["train_x"].astype(np.float32))
    mean, std = train.mean(), train.std()

    def images(key: str) -> torch.Tensor:
        return (torch.from_numpy(arrays[key].astype(np.float32)) - mean) / std

    def labels(key: str) -> torch.Tensor:
        return torch.from_numpy(arrays[key].astype(np.int64))

    return MNIST(
        images("train_x"), labels("train_y"), images("test_x"), labels("test_y")
    )


def batches(
    images: torch.Tensor,
    labels: torch.Tensor,
    batch: int,
    generator: torch.Generator | None = None,
) -> Iterator[pc.Sample]:
    """Cut the samples into batches of ``batch``, shuffled by ``generator`` (in
    order without one). The last batch is padded with zeros whose ``mask`` is
    False."""
    n = len(labels)
    order = (
        torch.arange(n) if generator is None else torch.randperm(n, generator=generator)
    )
    for start in range(0, n, batch):
        index = order[start : start + batch]
        pad = batch - len(index)
        x, y = images[index], labels[index]
        mask = torch.ones(batch, dtype=torch.bool)
        if pad:
            x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
            y = torch.cat([y, y.new_zeros(pad)])
            mask[-pad:] = False
        yield {"x": x, "y": y, "mask": mask}
