"""CIFAR-10 with a ResNet-20 of PC-ALM nodes on the GPU, compared with BP.

A ResNet-20 learns CIFAR-10 as a ``pc.Chain`` of five nodes on a CUDA device:
``input`` takes the batch the host writes, ``layer0`` to ``layer2`` (the three
stages: the stem with the first stage, then one node for each of the other
two) and ``output`` (the head and the cross entropy) relax towards the target
and update their weights from each other's messages only. Inside each node,
autograd differentiates its module, as in plain PyTorch; ``resnet.py`` says how
the network is cut. The same network, with the same initial weights, the same
batches in the same order and the same ``pc.optim.AdamW``, is then trained by BP
on the same GPU.

The host shows each batch, shuffled and augmented (random crops and flips), for
``S`` periods of the nodes with ``pc.show``, with ``reset`` set: no sample
continues the one before it. It measures the prequential accuracy (the first
prediction for each batch, before any node has learned from it), the test
accuracy after each epoch (the test batches with ``mask`` False, so that
nothing learns from them) and the real time.

The messages between the nodes stay on the GPU: a node on CUDA sends to a node
on CUDA from device to device. With the defaults of ``pc.TimeConstants``
(``tau.lam = tau.h / 2``, ``tau.r = tau.h``) and a period of ``tau.h / 2``, the
credit moves one node down in each period, so ``S = 12`` periods are enough
for the 4 connections down the chain and back. ``tau.h`` is 10 ms, so the
period is 5 ms of the timer, and the timer goes twice as fast as real time
(``--time-scale`` 2): 2.5 ms of real time, in which a node updates in about
1.7 ms on an RTX 4090 (launching the kernels from the CPU is the limit, so a
larger batch costs little more; BP gains more from it, and the batch is 32).

With the defaults (seed 0), in the 3rd epoch the prequential accuracy was
57.11% against 55.91% for BP, and the test accuracy 59.47% against 61.76%; in
the 10th, 73.05% against 73.20% and 73.90% against 74.35%. An epoch took 48 s
against 8 to 9 s for BP. The test accuracy moves by a few points from epoch to
epoch (the learning rate does not decay), for both, and the GPU does not
compute bit for bit the same in every run.

Needs a CUDA GPU, and the network once to download CIFAR-10 (about 160 MB)
into ``--data``. The defaults (3 epochs) take about 4 minutes, 10 epochs about
13 minutes. Run it from the repository::

    uv run python examples/cifar10/main.py
    uv run python examples/cifar10/main.py --epochs 10
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
from data import CIFAR10, batches, load
from metrics import Epoch, correct
from resnet import resnet
from torch import nn

import mujou
import mujou.pc as pc
import mujou.time
from mujou import TensorSpec


def generator(seed: int) -> Callable[[int], torch.Generator]:
    """Return the generator that shuffles and augments epoch ``e``, the same for
    PC and BP."""
    return lambda epoch: torch.Generator().manual_seed(seed + epoch)


def adamw(shown: float, lr: float) -> pc.optim.AdamW:
    """The ``torch.optim.AdamW(lr=lr)`` (``betas`` 0.9 and 0.999) of nodes that
    learn once per batch shown for ``shown`` seconds."""
    return pc.optim.AdamW(
        tau=shown / lr,
        tau_m=-shown / math.log(0.9),
        tau_v=-shown / math.log(0.999),
    )


def train_pc(
    modules: Sequence[nn.Module],
    data: CIFAR10,
    shuffle: Callable[[int], torch.Generator],
    args: argparse.Namespace,
) -> list[Epoch]:
    """Train the modules as PC-ALM nodes in a run, and measure each epoch."""
    # --8<-- [start:chain]
    # With these, delta follows -g in one update: the credit moves one node
    # down in each period of tau.h / 2, the default period of pc.Chain.
    tau = pc.TimeConstants(h=args.tau_h, lam=args.tau_h / 2, r=args.tau_h)
    shown = args.steps * tau.period  # how long the host shows each batch
    net = pc.Chain(
        *modules,  # already on the device
        x=TensorSpec((args.batch, 3, 32, 32)),
        y=TensorSpec((args.batch,), torch.int64),
        batch=args.batch,
        loss=pc.loss.CrossEntropyLoss(args.batch),
        tau=tau,
        optimizer=adamw(shown, args.lr),  # learn once per batch
        learn_interval=0.9 * shown,  # late in the showing, when lam has settled
        device=args.device,  # every node computes on the GPU
    )
    # --8<-- [end:chain]
    timer = mujou.time.Timer(time_scale=args.time_scale)
    test_seconds = 32 * tau.period  # enough to reach the output; nothing learns

    results: list[Epoch] = []
    with mujou.run(net, timer=timer, workers=args.workers) as run:
        for epoch in range(args.epochs):
            right = total = 0
            start = time.perf_counter()
            for x, y, mask in batches(
                data.train_x, data.train_y, args.batch, shuffle(epoch)
            ):
                # Shuffled batches: no sample continues the one before it.
                new = torch.ones_like(mask)
                sample = pc.msg.Sample(x=x, y=y, mask=mask, reset=new)
                p = pc.show(run, net, sample, seconds=shown)
                if not torch.isfinite(p["y_hat"]).all():
                    raise SystemExit("PC-ALM diverged: lower --time-scale")
                right += correct(p["y_hat"], y, mask)  # before learning from it
                total += int(mask.sum())
            seconds = time.perf_counter() - start
            # Test: the same nodes, with no label to learn from.
            test_right = test_total = 0
            for x, y, mask in batches(data.test_x, data.test_y, args.batch):
                unlabeled = torch.zeros_like(mask)  # keep mask to score with
                new = torch.ones_like(mask)
                p = pc.show(
                    run,
                    net,
                    pc.msg.Sample(x=x, y=y, mask=unlabeled, reset=new),
                    seconds=test_seconds,
                )
                test_right += correct(p["y_hat"], y, mask)
                test_total += int(mask.sum())
            results.append(Epoch(right / total, test_right / test_total, seconds))
            print(
                f"PC-ALM epoch {epoch + 1}: prequential {right / total:.2%}, "
                f"test {test_right / test_total:.2%}, {seconds:.1f} s"
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 with PC-ALM nodes and BP.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12, help="periods per batch (S)")
    parser.add_argument("--tau-h", type=float, default=10e-3, help="seconds")
    parser.add_argument("--lr", type=float, default=8e-3, help="the lr of AdamW")
    parser.add_argument("--time-scale", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--device", default="cuda", help="a CUDA device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data", type=Path, default=Path.home() / ".cache/mujou/cifar10"
    )
    args = parser.parse_args()
    shuffle = generator(args.seed)
    device = torch.device(args.device)

    # The nodes compute on the run's workers; one torch thread each (BP too).
    torch.set_num_threads(1)
    data = load(args.data)
    torch.manual_seed(args.seed)  # the initial weights, shared by PC and BP
    modules = [m.to(device) for m in resnet()]
    model = nn.Sequential(*copy.deepcopy(modules))  # BP starts from the same weights

    gil = sys._is_gil_enabled()  # pyright: ignore[reportPrivateUsage]
    print(
        f"PC-ALM: ResNet-20 in {len(modules) + 1} nodes on {device}, B = {args.batch}, "
        f"S = {args.steps}, tau.h = {args.tau_h * 1e3:g} ms, "
        f"time scale {args.time_scale:g}, {args.workers} workers, "
        f"GIL {'on' if gil else 'off'}"
    )
    pc_results = train_pc(modules, data, shuffle, args)
    shown = args.steps * args.tau_h / 2  # the period is tau.h / 2
    bp_results = bp.train(
        model,
        data,
        epochs=args.epochs,
        batch=args.batch,
        optimizer=adamw(shown, args.lr),
        elapsed=shown,
        generator=shuffle,
        device=device,
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
