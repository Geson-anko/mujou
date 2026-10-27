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
the batch is 128), and with 11 workers the nodes keep up with a period of 5 ms
of real time.

The period is ``tau.h / 16``, a quarter of that of ``pc.Chain``, so the timer
goes at a quarter of the real time (``--time-scale`` 0.25): with ``tau.h`` 20
ms the period is 1.25 ms of the timer and 5 ms of real time. The blocks make
the relaxation stiff (``resnet.py``), and with ``tau.h / 4`` its explicit
steps did not find the credit of BP. With ``tau.h / 16`` they found it as with
``tau.h / 64``, and what remains is how long the credit takes to settle: the
dual is fast (``tau.lam = tau.h / 4``), the deviation ``r = h - mu`` decays
(``tau.r = tau.h / 3``), each batch is shown for ``24 * tau.h`` (``S = 384``
periods) and the nodes learn at 0.9 of it. With the defaults, in the 3rd epoch
the prequential accuracy was 54.83% (seed 0) against 54.45% for BP, and
the test accuracy 59.05% against 59.02%; in the 10th, 73.35% against 73.27%
and 73.79% against 72.61% (another run of the same settings reached 71.56% and
71.41%). The test accuracy
moves by a few points from epoch to epoch (the learning rate does not decay),
for both, and the GPU does not compute bit for bit the same in every run.

Needs a CUDA GPU, and the network once to download CIFAR-10 (about 160 MB)
into ``--data``. The defaults (3 epochs) take about 40 minutes, 10 epochs
about 2 hours. Run it from the repository::

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
    # A period of tau.h / 16, not the tau.h / 4 of pc.Chain: the blocks make the
    # relaxation stiff, and the explicit step of h is exact only while
    # (1 + |J|^2) * period / tau.h < 2 for the Jacobian J of the next block.
    period = args.tau_h / args.div
    shown = args.steps * period  # how long the host shows each batch
    net = pc.Chain(
        *modules,  # already on the device
        x=TensorSpec((args.batch, 3, 32, 32)),
        y=TensorSpec((args.batch,), torch.int64),
        batch=args.batch,
        loss=pc.loss.CrossEntropyLoss(args.batch),
        # A fast dual, and a decay of r = h - mu: lam reaches the credit sooner,
        # at the same fixed point.
        tau=pc.TimeConstants(
            h=args.tau_h, lam=args.tau_lam * args.tau_h, r=args.tau_r * args.tau_h
        ),
        optimizer=adamw(shown, args.lr),  # learn once per batch
        learn_interval=0.9 * shown,  # late in the showing, when lam has settled
        period=period,
        max_dt=2 * period,
        device=args.device,  # every node computes on the GPU
    )
    # --8<-- [end:chain]
    timer = mujou.time.Timer(time_scale=args.time_scale)
    test_seconds = 32 * period  # enough to reach the output; nothing learns

    results: list[Epoch] = []
    with mujou.run(net, timer=timer, workers=args.workers) as run:
        for epoch in range(args.epochs):
            right = total = 0
            start = time.perf_counter()
            for x, y, mask in batches(
                data.train_x, data.train_y, args.batch, shuffle(epoch)
            ):
                p = pc.show(run, net, pc.msg.Sample(x=x, y=y, mask=mask), seconds=shown)
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
                    pc.msg.Sample(x=x, y=y, mask=unlabeled),
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
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=384, help="periods per batch (S)")
    parser.add_argument("--tau-h", type=float, default=20e-3, help="seconds")
    parser.add_argument(
        "--div", type=float, default=16, help="the period is tau.h / div"
    )
    parser.add_argument("--tau-lam", type=float, default=0.25, help="tau.lam in tau.h")
    parser.add_argument("--tau-r", type=float, default=1 / 3, help="tau.r in tau.h")
    parser.add_argument("--lr", type=float, default=1.6e-2, help="the lr of AdamW")
    parser.add_argument("--time-scale", type=float, default=0.25)
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
    shown = args.steps * args.tau_h / args.div
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
