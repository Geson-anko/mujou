"""Benchmark of ``Port.send`` and ``Port.recv``: the cost of one call in a hook.

Not a test: pytest does not collect it and no ``just`` recipe runs it. Run it by
hand on an otherwise idle machine and read the Markdown table it prints::

    uv run python benchmarks/ports.py
    uv run python benchmarks/ports.py --calls 20000 --repeats 7

With the GIL, in an environment of its own::

    UV_PROJECT_ENVIRONMENT=.venv-gil uv run --python 3.14 python benchmarks/ports.py

A source node sends to a sink node in a manual run, one after the other; each
times ``--calls`` calls of one method in a single ``on_update``, ``--repeats``
times, and the table gives the median and the minimum of the µs per call. The messages: a scalar, a tensor of a hidden
layer of MNIST, a tensor of its input, and a dict like a PC activity.
"""

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, TypedDict, override

import torch

import mujou
import mujou.testing
from mujou import DictSpec, TensorSpec


class Activity(TypedDict):
    h: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor


B = 64
SPECS: dict[str, Any] = {
    "scalar ()": TensorSpec(()),
    f"hidden ({B}, 256)": TensorSpec((B, 256)),
    f"input ({B}, 784)": TensorSpec((B, 784)),
    "activity {h, y, mask}": DictSpec(
        Activity,
        h=TensorSpec((B, 256)),
        y=TensorSpec((B,), torch.int64),
        mask=TensorSpec((B,), torch.bool),
    ),
}


def timed(call: Callable[[], object], calls: int, repeats: int) -> list[float]:
    """µs per call of ``call``, for each of ``repeats`` runs of ``calls`` calls."""
    for _ in range(calls // 10):
        call()
    per: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(calls):
            call()
        per.append((time.perf_counter() - t0) / calls * 1e6)
    return per


class Timing(mujou.Node):
    """Times each of its ``measures`` in ``on_update``."""

    def __init__(self, calls: int, repeats: int) -> None:
        super().__init__()
        self.calls = calls
        self.repeats = repeats
        self.times: dict[str, list[float]] = {}

    def measures(self) -> dict[str, Callable[[], object]]:
        raise NotImplementedError

    @override
    def on_update(self, dt: float) -> None:
        for name, call in self.measures().items():
            self.times[name] = timed(call, self.calls, self.repeats)


class Source(Timing):
    def __init__(self, spec: Any, calls: int, repeats: int) -> None:
        super().__init__(calls, repeats)
        self.next = mujou.Port(send=spec)
        self.value = spec.zeros()

    @override
    def measures(self) -> dict[str, Callable[[], object]]:
        return {"send": lambda: self.next.send(self.value)}


class Sink(Timing):
    def __init__(self, spec: Any, calls: int, repeats: int) -> None:
        super().__init__(calls, repeats)
        self.prev = mujou.Port(recv=spec)

    @override
    def measures(self) -> dict[str, Callable[[], object]]:
        return {"recv": self.prev.recv, "recv_with_time": self.prev.recv_with_time}


def measure(spec: Any, calls: int, repeats: int) -> dict[str, list[float]]:
    source, sink = Source(spec, calls, repeats), Sink(spec, calls, repeats)
    source.next.connect(sink.prev)
    with mujou.testing.run(mujou.Group(source=source, sink=sink)) as run:
        run.step(1e-3, source)
        run.step(0.0, sink)
    return source.times | sink.times


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark of Port.send / recv.")
    parser.add_argument("--calls", type=int, default=10000, help="calls per run")
    parser.add_argument("--repeats", type=int, default=5, help="runs per method")
    args = parser.parse_args()

    torch.set_num_threads(1)
    gil = "GIL" if sys._is_gil_enabled() else "free-threaded"  # pyright: ignore[reportPrivateUsage]
    print("# Port benchmark\n")
    print(f"- Python {platform.python_version()} ({gil}), torch {torch.__version__}")
    print(f"- {platform.platform()}, {platform.machine()}")
    print(f"- {args.calls} calls per run, {args.repeats} runs\n")
    print("| message | method | µs/call (median) | µs/call (min) |")
    print("| --- | --- | --- | --- |")
    for label, spec in SPECS.items():
        for method, per in measure(spec, args.calls, args.repeats).items():
            print(
                f"| {label} | {method} | {statistics.median(per):.2f} | "
                f"{min(per):.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
