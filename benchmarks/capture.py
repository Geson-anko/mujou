"""Benchmark of ``Run.state_dict``: the time to capture the state of a running node.

Not a test: pytest does not collect it and no ``just`` recipe runs it. Run it by
hand on an otherwise idle machine and read the Markdown table it prints::

    uv run python benchmarks/capture.py                 # on the CPU
    uv run python benchmarks/capture.py --device cuda   # a node on the GPU

A node holds ``--tensors`` tensors of each size on ``--device`` and updates every
10 ms; the host captures the state ``--repeats`` times, and the table gives the
median and the minimum of the ms per capture.
"""

import argparse
import platform
import statistics
import sys
import time
from typing import override

import torch

import mujou


class Holder(mujou.Node):
    """Holds ``count`` tensors of ``numel`` float32 elements on ``device``."""

    def __init__(self, count: int, numel: int, device: str) -> None:
        super().__init__(period=0.01, device=device)
        for i in range(count):
            setattr(self, f"w{i}", torch.zeros(numel, device=device))

    @override
    def on_update(self, dt: float) -> None:
        pass


def measure(count: int, numel: int, device: str, repeats: int) -> list[float]:
    """ms per capture, for each of ``repeats`` captures."""
    per: list[float] = []
    with mujou.run(Holder(count, numel, device)) as run:
        run.state_dict()  # warm up
        for _ in range(repeats):
            t0 = time.perf_counter()
            run.state_dict()
            per.append((time.perf_counter() - t0) * 1e3)
    return per


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark of Run.state_dict.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tensors", type=int, default=4, help="tensors per size")
    parser.add_argument(
        "--numels", type=int, nargs="+", default=[1, 256 * 256, 784 * 256, 4096**2]
    )
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    gil = "GIL" if sys._is_gil_enabled() else "free-threaded"  # pyright: ignore[reportPrivateUsage]
    print("# Capture benchmark\n")
    print(f"- Python {platform.python_version()} ({gil}), torch {torch.__version__}")
    print(f"- {platform.platform()}, {platform.machine()}, device {args.device}")
    print(f"- {args.tensors} float32 tensors per node, {args.repeats} captures\n")
    print(
        "| elements per tensor | MB in all | ms/capture (median) | ms/capture (min) |"
    )
    print("| --- | --- | --- | --- |")
    for numel in args.numels:
        per = measure(args.tensors, numel, args.device, args.repeats)
        mb = args.tensors * numel * 4 / 1e6
        print(
            f"| {numel:,} | {mb:.1f} | {statistics.median(per):.3f} | {min(per):.3f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
