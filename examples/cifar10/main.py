"""CIFAR-10 with a ResNet-20 of PC-ALM nodes on the GPU, compared with BP.

A ResNet-20 learns CIFAR-10 as a ``pc.Chain`` of eleven nodes on a CUDA device:
``input`` takes the batch the host writes, ``layer0`` to ``layer8`` (the stem
with the first block, then one node for each of the other 8 residual blocks)
and ``output`` (the head and the cross entropy) relax towards the target and
update their weights from each other's messages only. Inside each node,
autograd differentiates its module, as in plain PyTorch; ``resnet.py`` says how
the network is cut. The same network, with the same initial weights, the same
batches in the same order and the same ``pc.optim.AdamW``, is then trained by BP on
the same GPU.

The host shows each batch, shuffled and augmented (random crops and flips), for
``S`` periods of the nodes with ``pc.show``, and measures the prequential
accuracy (the first prediction for each batch, before any node has learned from
it), the test accuracy after each epoch (the test batches with ``mask`` False,
so that nothing learns from them) and the real time.

The messages between the nodes stay on the GPU: a node on CUDA sends to a node
on CUDA from device to device. An update of a node takes about 1.2 ms on an
RTX 4090, whatever the batch (the CPU launching the kernels is the limit, so
the batch is 128). The run's time is the real time (``--time-scale`` 1): with
``tau.h`` 20 ms the period is 5 ms, and 99% of the updates of every node came
within 2 periods of the one before (``max_dt`` is 2, beyond which a node drops
the time) with 11 workers. On a machine with fewer cores or a slower GPU, raise
``--tau-h`` or lower ``--time-scale``.

A batch is shown for ``S = 64`` periods, long for 11 nodes: the sample takes
about one period per node to reach the output and the credit about one per node
to come back. ``pc.relax.HeavyBall`` (``--momentum`` is its ``tau`` in ``tau.h``; 0 is the
plain ``pc.relax.Euler``) gives ``h`` the inertia of a heavy ball, which carries the
credit across the chain in fewer periods: without it, 64 periods left the 3rd epoch about 10 points below BP and
128 were needed (``--steps 128 --momentum 0``), twice the real time. With the
defaults, in the 3rd epoch the prequential accuracy was 49.48% (seed 0) against
54.45% for BP, and the test accuracy 48.71% against 59.02%; in the 10th, 62.77%
against 73.27% and 62.89% against 72.61%. Without the momentum and with 128
periods, the 10th epoch was 68.73% and 68.95%, so the momentum trades some
accuracy in a long run for half the time. A momentum of 0.8 per update diverged
and with 32 periods the nodes did not learn. The blocks make the relaxation stiff,
so it finds the credit less exactly than BP does (``resnet.py``). The test
accuracy moves by a few points from epoch to epoch (the learning rate does not
decay), for both, and the GPU does not compute bit for bit the same in every
run.

Needs a CUDA GPU, and the network once to download CIFAR-10 (about 160 MB)
into ``--data``. The defaults (3 epochs) take about 8 minutes, 10 epochs about
25. Run it from the repository::

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
    period = args.tau_h / 4  # the period of every node of pc.Chain
    shown = args.steps * period  # how long the host shows each batch
    # The inertia of h carries the credit across the 11 nodes in fewer periods.
    relaxer = (
        pc.relax.HeavyBall(tau=args.momentum * args.tau_h)
        if args.momentum > 0
        else pc.relax.Euler()
    )
    net = pc.Chain(
        *modules,  # already on the device
        x=TensorSpec((args.batch, 3, 32, 32)),
        y=TensorSpec((args.batch,), torch.int64),
        batch=args.batch,
        loss=pc.losses.CrossEntropyLoss(args.batch),
        # The dual as fast as h: a slower one needs even more periods to relax.
        tau=pc.TimeConstants(h=args.tau_h, lam=args.tau_h),
        optimizer=adamw(shown, args.lr),  # learn once per batch
        relaxer=relaxer,
        learn_interval=0.75 * shown,
        device=args.device,  # every node computes on the GPU
    )
    # --8<-- [end:chain]
    timer = mujou.time.Timer(time_scale=args.time_scale)

    results: list[Epoch] = []
    with mujou.run(net, timer=timer, workers=args.workers) as run:
        for epoch in range(args.epochs):
            right = total = 0
            start = time.perf_counter()
            for x, y, mask in batches(
                data.train_x, data.train_y, args.batch, shuffle(epoch)
            ):
                p = pc.show(
                    run, net, pc.messages.Sample(x=x, y=y, mask=mask), seconds=shown
                )
                if not torch.isfinite(p["y_hat"]).all():
                    raise SystemExit("PC-ALM diverged: lower --time-scale")
                right += correct(p["y_hat"], y, mask)  # before learning from it
                total += int(mask.sum())
            seconds = time.perf_counter() - start
            # Test: the same nodes, with no label to learn from.
            test_right = test_total = 0
            for x, y, mask in batches(data.test_x, data.test_y, args.batch):
                unlabeled = torch.zeros_like(mask)  # keep mask to score with
                p = pc.show(
                    run,
                    net,
                    pc.messages.Sample(x=x, y=y, mask=unlabeled),
                    seconds=shown,
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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=64, help="periods per batch (S)")
    parser.add_argument("--tau-h", type=float, default=20e-3, help="seconds")
    parser.add_argument("--lr", type=float, default=1.6e-2, help="the lr of AdamW")
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.25,
        help="the tau of pc.relax.HeavyBall in tau.h; 0 is pc.relax.Euler",
    )
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=11)
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
    shown = args.steps * args.tau_h / 4
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
