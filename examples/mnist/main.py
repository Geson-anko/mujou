"""MNIST with PC-ALM nodes running asynchronously, compared with BP.

A 784-256-256-10 relu network learns MNIST as four nodes of ``mujou.pc``, each
updating on its own, with no barrier between them: ``Input`` takes the batch the
host writes, two ``Layer`` nodes and ``Output`` (cross entropy) relax towards
the target and update their weights from their neighbors' messages only. The
same network, with the same initial weights and the same batches in the same
order, is then trained by BP with plain PyTorch SGD.

The host shows each batch for ``S`` periods of the nodes. It measures:

- the prequential accuracy: the first prediction ``Output`` makes for a new batch,
  before any node has learned from it;
- the test accuracy after each epoch: the test batches go through the same
  running network with ``mask`` False, so nothing learns from them. There is no
  inference mode to switch to;
- the real time, the time scale, the workers, and the ``dt`` the nodes saw.

The run's timer goes ``--time-scale`` times faster than real time so that the
nodes update about as fast as they can compute, instead of waiting for their
period. Too fast, and an update comes later than its period: ``dt`` grows. The
nodes count at most ``max_dt`` of it (``tau.h / 2``, 2 periods), which keeps the
relaxation stable but drops the time beyond. The program prints the
distribution of ``dt`` and warns when its median exceeds 1.5 periods or when
more than 1% of the updates come later than 2 periods. The default 1 keeps
``dt`` at 1 to 1.25 periods on an idle 32-core machine, where the first layer
(784 inputs) is the slowest node; there, 3 epochs reach a test accuracy of
97.82% (seed 0) and 97.59% (seed 1), against 97.54% and 97.45% for BP. At 2,
that node takes most of its period to compute and 1% of its updates come later
than 2 periods (a warning), though one epoch ran through (96.47% and 96.23%).
To choose the time scale on another machine, run one epoch and take the highest
scale that gives no warning and does not diverge.

Run it on an otherwise idle machine. When other processes take the cores, a node
may wait for one for 100 ms while its neighbors keep learning, and the network
may diverge (the program stops when a prediction is not finite) or degrade;
lower ``--time-scale`` if it does. On 32 cores, one epoch at 1 ran through with
16 and 26 busy processes (test 96.62%, and 96.52% and 96.23% in two runs) and
diverged with 32; at 0.5 with 32, it ran through to 93.52%.

Needs the network once, to download MNIST (about 11 MB) into ``--data``. Runs on
the CPU; the defaults (3 epochs) take about 2 minutes. Run it from the repository::

    uv run python examples/mnist/main.py
    uv run python examples/mnist/main.py --epochs 1
"""

import argparse
import copy
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, override

import bp
import torch
from bp import Epoch, correct
from data import MNIST, batches, load
from torch import nn

import mujou
import mujou.pc as pc
from mujou import TensorSpec

BINS = 8 * 4  # the dt histogram: quarter periods up to 8 periods, then the rest


class Watch(mujou.Node):
    """Counts the ``dt`` of the node's updates in a histogram of quarter periods,
    kept in the node's state (``"dt_histogram"``) for the host to read."""

    @override
    def on_start(self) -> None:
        super().on_start()
        self.dt_histogram = torch.zeros(BINS + 2, dtype=torch.int64)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        # Bin i holds (i - 1) / 4 < dt / period <= i / 4; the tolerance keeps a dt
        # of exactly one period (in whole ns) in the bin of 1.
        quarters = math.ceil(4 * dt / self.period - 1e-6)
        self.dt_histogram[min(quarters, BINS + 1)] += 1


class WatchedLayer(Watch, pc.Layer):
    pass


class WatchedOutput(Watch, pc.Output):
    pass


def generator(seed: int) -> Callable[[int], torch.Generator]:
    """Return the generator that shuffles epoch ``e``, the same for PC and BP."""
    return lambda epoch: torch.Generator().manual_seed(seed + epoch)


def quantile(histogram: torch.Tensor, q: float) -> float:
    """Return the upper edge, in periods, of the bin of the ``q`` quantile."""
    counts = histogram.tolist()
    rank = q * sum(counts)
    seen = 0
    for i, count in enumerate(counts):
        seen += count
        if seen >= rank:
            return i / 4 if i <= BINS else math.inf
    return math.inf


def dt_report(nodes: dict[str, Any]) -> str:
    """Describe the ``dt`` of the watched nodes from their states."""
    lines: list[str] = []
    for path, state in nodes.items():
        if "dt_histogram" in state:
            h = state["dt_histogram"]
            lines.append(
                f"  {path:>3}: {int(h.sum()):7d} updates, dt <= "
                + ", ".join(f"{quantile(h, q):g}" for q in (0.5, 0.9, 0.99))
                + f" periods (50/90/99%), > {BINS // 4} periods {int(h[BINS + 1])}"
            )
    return "\n".join(lines)


def check_dt(nodes: dict[str, Any]) -> None:
    """Warn when the ``dt`` of a watched node is too often longer than its period:
    a median over 1.5 periods, or a 99th percentile over 2 (``max_dt``, beyond
    which the node drops the time)."""
    for path, state in nodes.items():
        if "dt_histogram" not in state:
            continue
        histogram = state["dt_histogram"]
        if quantile(histogram, 0.5) > 1.5 or quantile(histogram, 0.99) > 2:
            print(
                f"warning: the updates of {path} come too late (see its dt below): "
                "the nodes do not keep up with the time scale; lower --time-scale"
            )


# --8<-- [start:build]
def build(
    modules: Sequence[nn.Module], batch: int, tau: pc.TimeConstants, learn: float
) -> tuple[pc.Input, WatchedOutput, mujou.Group]:
    """Wrap the three modules in nodes and connect them."""
    x, y = TensorSpec((batch, 784)), TensorSpec((batch,), torch.int64)
    inp = pc.Input(x=x, y=y, period=tau.h / 4)
    l1 = WatchedLayer(modules[0], act=inp.act, tau=tau, learn_interval=learn)
    l2 = WatchedLayer(modules[1], act=l1.act, tau=tau, learn_interval=learn)
    out = WatchedOutput(
        modules[2], act=l2.act, loss=pc.cross_entropy, tau=tau, learn_interval=learn
    )
    inp.next.connect(l1.prev)
    l1.next.connect(l2.prev)
    l2.next.connect(out.prev)
    return inp, out, mujou.Group(inp=inp, l1=l1, l2=l2, out=out)


# --8<-- [end:build]


def train_pc(
    modules: Sequence[nn.Module],
    data: MNIST,
    shuffle: Callable[[int], torch.Generator],
    args: argparse.Namespace,
) -> list[Epoch]:
    """Train the modules as PC-ALM nodes in a run, and measure each epoch."""
    # --8<-- [start:tau]
    period = args.tau_h / 4
    shown = args.steps * period  # how long the host shows each batch
    # The learning rate of one batch is sum_dt / tau.theta = S * period / tau.theta.
    tau = pc.TimeConstants(h=args.tau_h, lam=2 * args.tau_h, theta=shown / args.lr)
    learn = 0.75 * shown  # learn once, 3/4 of the way through each batch
    inp, out, net = build(modules, args.batch, tau, learn)
    timer = mujou.time.Timer()
    timer.set_time_scale(args.time_scale)
    # --8<-- [end:tau]

    # --8<-- [start:show]
    def show(run: mujou.Run, sample: pc.Sample) -> pc.Prediction:
        """Write ``sample``, return the first prediction made from it, and return
        once it has been shown for ``shown`` seconds of the run's time."""
        t = run.write({inp.prev: sample})
        first, _ = run.wait_for(out.next, lambda p: int(p["time"]) == t.ns, timeout=10)
        if not torch.isfinite(first["y_hat"]).all():
            raise SystemExit("PC-ALM diverged: lower --time-scale (see the docstring)")
        timer.sleep(max(0.0, shown - (float(timer.now()) - float(t))))
        return first

    # --8<-- [end:show]

    results: list[Epoch] = []
    # --8<-- [start:run]
    with mujou.run(net, timer=timer, workers=args.workers) as run:
        for epoch in range(args.epochs):
            right = total = 0
            start = time.perf_counter()
            train = batches(data.train_x, data.train_y, args.batch, shuffle(epoch))
            for i, sample in enumerate(train):
                first = show(run, sample)
                right += correct(first["y_hat"], first["y"], first["mask"])
                total += int(sample["mask"].sum())
                if epoch == 0 and i == 100:  # a pause to look, not to synchronize
                    check_dt(run.state_dict()["nodes"])
            seconds = time.perf_counter() - start
            # Test: the same nodes, with no label to learn from.
            test_right = test_total = 0
            for sample in batches(data.test_x, data.test_y, args.batch):
                unlabeled: pc.Sample = {
                    **sample,
                    "mask": torch.zeros_like(sample["mask"]),
                }
                show(run, unlabeled)
                last = run.read(out.next)  # the latest prediction, after S periods
                test_right += correct(last["y_hat"], sample["y"], sample["mask"])
                test_total += int(sample["mask"].sum())
            results.append(Epoch(right / total, test_right / test_total, seconds))
            print(
                f"PC-ALM epoch {epoch + 1}: prequential {right / total:.2%}, "
                f"test {test_right / test_total:.2%}, {seconds:.1f} s"
            )
            # --8<-- [end:run]
    nodes = run.state_dict()["nodes"]
    print(f"dt of the nodes (period {period * 1e3:g} ms of the run's time):")
    print(dt_report(nodes))
    check_dt(nodes)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST with PC-ALM nodes and BP.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32, help="periods per batch (S)")
    parser.add_argument("--tau-h", type=float, default=4e-3, help="seconds")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate per batch")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path.home() / ".cache/mujou/mnist")
    args = parser.parse_args()
    shuffle = generator(args.seed)

    # The nodes compute on the run's workers; one torch thread each (BP too).
    torch.set_num_threads(1)
    data = load(args.data)
    torch.manual_seed(args.seed)  # the initial weights, shared by PC and BP
    modules = [
        nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
        nn.Sequential(nn.Linear(256, 256), nn.ReLU()),
        nn.Linear(256, 10),
    ]
    model = nn.Sequential(*copy.deepcopy(modules))  # BP starts from the same weights

    gil = sys._is_gil_enabled()  # pyright: ignore[reportPrivateUsage]
    print(
        f"PC-ALM: B = {args.batch}, S = {args.steps}, tau.h = {args.tau_h * 1e3:g} ms, "
        f"time scale {args.time_scale:g}, {args.workers} workers, "
        f"GIL {'on' if gil else 'off'}"
    )
    pc_results = train_pc(modules, data, shuffle, args)
    bp_results = bp.train(
        model,
        data,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        generator=shuffle,
    )

    print(f"\n{'':>6} {'prequential':>23} {'test':>23} {'train time (s)':>23}")
    print(f"{'epoch':>6}" + f" {'PC-ALM':>11} {'BP':>11}" * 3)
    for epoch, (p, b) in enumerate(zip(pc_results, bp_results, strict=True)):
        print(
            f"{epoch + 1:>6} {p.prequential:>11.2%} {b.prequential:>11.2%} "
            f"{p.test:>11.2%} {b.test:>11.2%} {p.seconds:>11.1f} {b.seconds:>11.1f}"
        )
    n = len(pc_results) * math.ceil(len(data.train_y) / args.batch)  # batches
    pc_ms = sum(p.seconds for p in pc_results) / n * 1e3
    bp_ms = sum(b.seconds for b in bp_results) / n * 1e3
    print(
        f"per batch: PC-ALM {pc_ms:.2f} ms ({args.steps} forward and VJP in each "
        f"node), BP {bp_ms:.2f} ms (one forward and backward)"
    )


if __name__ == "__main__":
    main()
