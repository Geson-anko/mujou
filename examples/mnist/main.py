"""MNIST with PC-ALM nodes running asynchronously, compared with BP.

A 784-256-256-10 relu network learns MNIST as a ``pc.Chain`` of four nodes,
each updating on its own, with no barrier between them: ``input`` takes the
batch the host writes, ``layer0`` and ``layer1`` and ``output`` (cross entropy)
relax towards the target and update their weights from their neighbors'
messages only. The same network, with the same initial weights and the same
batches in the same order, is then trained by BP with plain PyTorch SGD.

The host shows each batch for ``S`` periods of the nodes with ``pc.show``. It
measures:

- the prequential accuracy: the first prediction the chain makes for a new
  batch, before any node has learned from it;
- the test accuracy after each epoch: the test batches go through the same
  running network with ``mask`` False, so nothing learns from them. There is no
  inference mode to switch to;
- the real time, the time scale and the workers.

The run's timer goes ``--time-scale`` times faster than real time so that the
nodes update about as fast as they can compute, instead of waiting for their
period. Too fast, and an update comes later than its period: ``dt`` grows. The
nodes count at most ``max_dt`` of it (``tau.h / 2``, 2 periods), which keeps the
relaxation stable but drops the time beyond. ``--watch`` counts the ``dt`` of
the nodes (``watch.py``), prints its distribution and warns when its median
exceeds 1.5 periods or when more than 1% of the updates come later than 2
periods. The default 1 keeps ``dt`` at 1 to 1.25 periods on an idle 32-core
machine, where the first layer (784 inputs) is the slowest node; there, 3
epochs reach a test accuracy of 97.60% (seed 0), against 97.54% for BP. At 2,
that node takes most of its period to compute and 1% of its updates come later
than 2 periods (a warning). To choose the time scale on another machine, run
one epoch with ``--watch`` and take the highest scale that gives no warning and
does not diverge.

Run it on an otherwise idle machine. When other processes take the cores, a node
may wait for one for 100 ms while its neighbors keep learning, and the network
may diverge (the program stops when a prediction is not finite) or degrade;
lower ``--time-scale`` if it does.

Needs the network once, to download MNIST (about 11 MB) into ``--data``. Runs on
the CPU; the defaults (3 epochs) take about 2 minutes. Run it from the repository::

    uv run python examples/mnist/main.py
    uv run python examples/mnist/main.py --epochs 1 --watch
"""

import argparse
import copy
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import bp
import torch
import watch
from data import MNIST, batches, load
from metrics import Epoch, correct
from torch import nn

import mujou
import mujou.pc as pc
import mujou.time
from mujou import TensorSpec


def generator(seed: int) -> Callable[[int], torch.Generator]:
    """Return the generator that shuffles epoch ``e``, the same for PC and BP."""
    return lambda epoch: torch.Generator().manual_seed(seed + epoch)


def train_pc(
    modules: Sequence[nn.Module],
    data: MNIST,
    shuffle: Callable[[int], torch.Generator],
    args: argparse.Namespace,
) -> list[Epoch]:
    """Train the modules as PC-ALM nodes in a run, and measure each epoch."""
    # --8<-- [start:chain]
    period = args.tau_h / 4  # the period of every node of pc.Chain
    shown = args.steps * period  # how long the host shows each batch
    # The learning rate of one batch is shown / tau.theta: the nodes learn once
    # per batch, for the time since they last learned.
    tau = pc.TimeConstants(h=args.tau_h, lam=2 * args.tau_h, theta=shown / args.lr)
    net = pc.Chain(
        *modules,
        x=TensorSpec((args.batch, 784)),
        y=TensorSpec((args.batch,), torch.int64),
        loss=pc.cross_entropy,
        tau=tau,
        learn_interval=0.75 * shown,  # learn once, 3/4 of the way through each batch
    )
    # --8<-- [end:chain]
    if args.watch:
        for node in vars(net).values():
            if isinstance(node, mujou.Node):  # input, layer0, layer1, output
                watch.watch(node)
    timer = mujou.time.Timer(time_scale=args.time_scale)

    results: list[Epoch] = []
    # --8<-- [start:run]
    with mujou.run(net, timer=timer, workers=args.workers) as run:
        for epoch in range(args.epochs):
            right = total = 0
            start = time.perf_counter()
            for i, (x, y, mask) in enumerate(
                batches(data.train_x, data.train_y, args.batch, shuffle(epoch))
            ):
                p = pc.show(run, net, pc.Sample(x=x, y=y, mask=mask), seconds=shown)
                if not torch.isfinite(p["y_hat"]).all():
                    raise SystemExit("PC-ALM diverged: lower --time-scale")
                right += correct(p["y_hat"], y, mask)  # before learning from it
                total += int(mask.sum())
                if args.watch and epoch == 0 and i == 100:  # a pause to look
                    watch.check(run.state_dict()["nodes"])
            seconds = time.perf_counter() - start
            # Test: the same nodes, with no label to learn from.
            test_right = test_total = 0
            for x, y, mask in batches(data.test_x, data.test_y, args.batch):
                unlabeled = torch.zeros_like(mask)  # keep mask to score with
                p = pc.show(
                    run, net, pc.Sample(x=x, y=y, mask=unlabeled), seconds=shown
                )
                test_right += correct(p["y_hat"], y, mask)
                test_total += int(mask.sum())
            results.append(Epoch(right / total, test_right / test_total, seconds))
            print(
                f"PC-ALM epoch {epoch + 1}: prequential {right / total:.2%}, "
                f"test {test_right / test_total:.2%}, {seconds:.1f} s"
            )
    # --8<-- [end:run]
    if args.watch:
        nodes = run.state_dict()["nodes"]
        print(f"dt of the nodes (period {period * 1e3:g} ms of the run's time):")
        print(watch.report(nodes))
        watch.check(nodes)
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
    parser.add_argument(
        "--watch", action="store_true", help="count the dt of the nodes"
    )
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
        model, data, epochs=args.epochs, batch=args.batch, lr=args.lr, generator=shuffle
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
