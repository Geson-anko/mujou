"""Benchmark of a PC-ALM chain whose target is as large as an activity (R12).

Not a test: pytest does not collect it and no ``just`` recipe runs it. Run it by
hand on an otherwise idle machine and read the Markdown table it prints::

    uv run python benchmarks/pc_dense.py
    uv run python benchmarks/pc_dense.py --steps 4000 --seconds 5
    uv run python benchmarks/pc_dense.py --device cuda   # the nodes on the GPU

A chain ``Input`` -> 784-256 ``Layer`` -> 256-256 ``Layer`` -> ``Output`` of
``mujou.pc`` (B = 64) learns one fixed sample, with a dense target (a
regression: ``y`` of ``(B, 256)``, as large as the ``h`` of a hidden layer) and
with class labels (``y`` of ``(B,)``, as MNIST). The target travels with the
activity from ``Input`` to ``Output``, so what it costs grows with every hop.

- manual: a ``mujou.testing.run`` whose steps update the four nodes once each,
  in order, on one thread. The table gives the median and the minimum over
  ``--repeats`` runs of ``--steps`` steps of the µs per step.
- free: a ``mujou.run`` on ``--workers`` workers with every period 0, so each
  node updates as fast as it can for ``--seconds`` seconds of real time. The
  table gives the updates per second of each node, the median over
  ``--repeats`` runs.
"""

import argparse
import platform
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any, override

import torch
from torch import nn

import mujou
import mujou.pc as pc
import mujou.testing
from mujou import TensorSpec

B = 64
TAU = pc.TimeConstants(h=4e-3, lam=8e-3)
OPTIMIZER = pc.optim.SGD(tau=1.28)
PERIOD = TAU.h / 4
PATHS = ("inp", "l1", "l2", "out")

# The target: its spec, the width of the output and the loss.
TARGETS: dict[str, tuple[TensorSpec, int, Callable[..., torch.Tensor]]] = {
    "dense y (B, 256)": (TensorSpec((B, 256)), 256, pc.loss.MSELoss(B)),
    "labels y (B,)": (TensorSpec((B,), torch.int64), 10, pc.loss.CrossEntropyLoss(B)),
}


class Counted(mujou.Node):
    """Counts its updates in ``updates``, which the host reads in the state."""

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.updates += 1


class CountedInput(Counted, pc.Input):
    pass


class CountedLayer(Counted, pc.Layer):
    pass


class CountedOutput(Counted, pc.Output):
    pass


def build(
    target: str, period: float, device: str
) -> tuple[mujou.Group, dict[Any, pc.msg.Sample]]:
    """The chain for ``target``, with nodes of ``period`` on ``device``, and its
    input."""
    y, width, loss = TARGETS[target]
    torch.manual_seed(0)
    common: dict[str, Any] = {
        "tau": TAU,
        "optimizer": OPTIMIZER,
        "learn_interval": 8 * PERIOD,
        "period": period,
        "device": device,
    }
    inp = CountedInput(
        x=TensorSpec((B, 784)), y=y, period=period, batch=B, device=device
    )
    l1 = CountedLayer(
        nn.Sequential(nn.Linear(784, 256), nn.Tanh()).to(device),
        act_spec=inp.act_spec,
        **common,
    )
    l2 = CountedLayer(
        nn.Sequential(nn.Linear(256, 256), nn.Tanh()).to(device),
        act_spec=l1.act_spec,
        **common,
    )
    out = CountedOutput(
        nn.Linear(256, width).to(device), act_spec=l2.act_spec, loss=loss, **common
    )
    inp.next.connect(l1.prev)
    l1.next.connect(l2.prev)
    l2.next.connect(out.prev)
    for node in (inp, l1, l2, out):
        node.updates = 0
        node.register_state("updates")
    sample: pc.msg.Sample = {
        "x": torch.randn(B, 784),
        "y": y.rand() if y.dtype.is_floating_point else torch.randint(10, (B,)),
        "mask": torch.ones(B, dtype=torch.bool),
    }
    net = mujou.Group(inp=inp, l1=l1, l2=l2, out=out)
    return net, {inp.prev: sample}


def manual(target: str, steps: int, device: str) -> float:
    """µs per step of the four nodes in a manual run."""
    net, inputs = build(target, PERIOD, device)
    with mujou.testing.run(net, inputs=inputs) as run:
        for _ in range(steps // 10):
            run.step(PERIOD)
        t0 = time.perf_counter()
        for _ in range(steps):
            run.step(PERIOD)
        return (time.perf_counter() - t0) / steps * 1e6


def free(target: str, seconds: float, workers: int, device: str) -> list[float]:
    """Updates per second of each node in a free run."""
    net, inputs = build(target, 0.0, device)
    with mujou.run(net, inputs=inputs, workers=workers) as run:
        time.sleep(0.2)
        before = run.state_dict()["nodes"]
        t0 = time.perf_counter()
        time.sleep(seconds)
        after = run.state_dict()["nodes"]
        elapsed = time.perf_counter() - t0
    return [
        (after[p]["state"]["updates"] - before[p]["state"]["updates"]) / elapsed
        for p in PATHS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="PC-ALM with a dense target.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)  # one torch thread per node, as examples/mnist

    gil = sys._is_gil_enabled()  # pyright: ignore[reportPrivateUsage]
    print(
        f"- Python {platform.python_version()} (GIL {'on' if gil else 'off'}), "
        f"torch {torch.__version__}"
    )
    print(f"- {platform.platform()}, {platform.machine()}, device {args.device}")
    print(
        f"- manual: {args.steps} steps per run; free: {args.seconds:g} s per run, "
        f"{args.workers} workers; {args.repeats} runs"
    )
    print()
    print("| target | manual µs/step (median) | manual µs/step (min) | ", end="")
    print(" | ".join(f"free {p} updates/s" for p in PATHS) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in PATHS) + " |")
    for target in TARGETS:
        steps = [manual(target, args.steps, args.device) for _ in range(args.repeats)]
        rates = [
            free(target, args.seconds, args.workers, args.device)
            for _ in range(args.repeats)
        ]
        median = [statistics.median(r[i] for r in rates) for i in range(len(PATHS))]
        print(
            f"| {target} | {statistics.median(steps):.1f} | {min(steps):.1f} | "
            + " | ".join(f"{r:,.0f}" for r in median)
            + " |"
        )


if __name__ == "__main__":
    main()
