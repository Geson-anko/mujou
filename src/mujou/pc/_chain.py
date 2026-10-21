"""A chain of ``mujou.pc`` nodes in one ``Network``, and how the host shows it a
sample."""

from collections.abc import Callable
from typing import Never, Protocol

import torch
from torch import nn

from mujou import Network, Port, Run, TensorSpec
from mujou.pc._input import Input
from mujou.pc._layer import Layer
from mujou.pc._messages import Prediction, Sample
from mujou.pc._output import Output
from mujou.pc._time_constants import TimeConstants


class Chain(Network):
    """``Input -> Layer -> ... -> Output``: a network that learns ``modules`` in
    sequence by PC-ALM, as ``nn.Sequential(*modules)`` would compute them.

    ``modules[:-1]`` each become a ``Layer`` and ``modules[-1]`` the ``Output``,
    after an ``Input`` of the samples ``x`` and targets ``y``. Each node takes
    the ``act_spec`` of the one before and the nodes are connected in order.
    The children are ``input``, ``layer0``, ..., ``output`` (``modules[i]`` is
    in ``layer{i}``), which are also their paths in ``Run.state_dict()``. Every
    node runs with the period ``tau.period`` (``tau.h / 4``); for other
    periods, build the nodes one by one (``show`` shows them samples too).

    The host writes a ``Sample`` to ``prev`` and reads a ``Prediction`` from
    ``next``, or does both with ``show``.

    Example::

        B = 64
        net = pc.Chain(
            nn.Sequential(nn.Linear(784, 256), nn.ReLU()),
            nn.Linear(256, 10),
            x=TensorSpec((B, 784)),
            y=TensorSpec((B,), torch.int64),
            loss=pc.cross_entropy,
            tau=pc.TimeConstants(h=4e-3, lam=8e-3, theta=1.28),
            learn_interval=0.15,
        )
        with mujou.run(net) as run:
            prediction = pc.show(run, net, sample, seconds=0.2)

    Args:
        modules: The functions of the nodes, in order, on ``device``.
        x: The spec of the inputs, ``(B, *in)``.
        y: The spec of the targets, ``(B, *target)``.
        loss: The loss of each sample, as in ``Output``.
        tau: The time constants of every node.
        learn_interval: As in ``Layer``. Longer than the round trip of the
            credit, ``2 * len(modules) * tau.period``, is recommended.
        max_dt: As in ``Layer``.
        device: Where the nodes compute.

    Raises:
        ValueError: As ``Input``, ``Layer`` and ``Output`` raise.
    """

    def __init__(
        self,
        *modules: nn.Module,
        x: TensorSpec,
        y: TensorSpec,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tau: TimeConstants,
        learn_interval: float | None,
        max_dt: float | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.input = Input(x=x, y=y, period=tau.period, device=device)
        last: Input | Layer = self.input
        for i, module in enumerate(modules[:-1]):
            layer = Layer(
                module,
                act_spec=last.act_spec,
                tau=tau,
                learn_interval=learn_interval,
                max_dt=max_dt,
                device=device,
            )
            last.next.connect(layer.prev)
            setattr(self, f"layer{i}", layer)
            last = layer
        self.output = Output(
            modules[-1],
            act_spec=last.act_spec,
            loss=loss,
            tau=tau,
            learn_interval=learn_interval,
            max_dt=max_dt,
            device=device,
        )
        last.next.connect(self.output.prev)
        self.prev = self.input.prev
        self.next = self.output.next


class _Shown(Protocol):
    """What ``show`` shows a sample: a ``Chain``, or a network built by hand whose
    ``prev`` takes a ``Sample`` and whose ``next`` sends a ``Prediction``."""

    @property
    def prev(self) -> Port[Never, Sample]: ...

    @property
    def next(self) -> Port[Prediction, Never]: ...


def show(run: Run, chain: _Shown, sample: Sample, *, seconds: float) -> Prediction:
    """Return the prediction ``chain`` makes for ``sample`` before it learns from
    it (a prequential prediction), after showing it ``seconds`` long.

    ``chain`` is a ``Chain``, or any network of the nodes of ``mujou.pc`` that
    exposes the ``prev`` of its ``Input`` and the ``next`` of its ``Output``.
    Writes ``sample`` to ``chain.prev``, waits for the first prediction the
    host reads for that write, and returns it once ``seconds`` (of
    ``run.timer``) have passed since the write, while the nodes learn from the
    sample (``Sample.mask`` True) or only relax towards it (every ``mask``
    False, as to evaluate, where a ``seconds`` that lets the input reach the
    output is enough). A node learns from a sample from its second update with
    it on (``learn_interval`` after the write, when set), and the first
    prediction of the sample is made of the first activities with it, so the
    prediction comes before learning unless the host reads it later than that.

    Nothing moves the timer of a ``ManualRun``, so there it does not return.

    Example::

        mask = torch.ones(B, dtype=torch.bool)
        p = pc.show(run, net, pc.Sample(x=x, y=y, mask=mask), seconds=0.2)
        correct = int((p["y_hat"].argmax(1) == y).sum())

    Raises:
        ValueError: A tensor of ``sample`` has another shape or dtype than the
            spec of ``chain.prev``.
        TypeError: ``sample`` lacks a key, or has one too many.
        RuntimeError: The run shut down normally before the prediction.
        NetworkError: The run stopped because a node raised.
    """
    written = run.write({chain.prev: sample})
    prediction, _ = run.wait_for(chain.next, lambda p: int(p["time"]) == written.ns)
    run.timer.sleep(max(0.0, seconds - (run.timer.now() - written)))
    return prediction
