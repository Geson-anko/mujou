"""A chain of ``mujou.pc`` nodes in one ``Network``, and how the host shows it a
sample."""

from abc import ABC
from collections.abc import Callable
from typing import Never

import torch
from torch import nn

from mujou import Network, Port, Run, TensorSpec
from mujou.pc.input import Input
from mujou.pc.layer import Layer
from mujou.pc.msg import Prediction, Sample
from mujou.pc.optim import Optimizer
from mujou.pc.output import Output
from mujou.pc.relax import Euler, Relaxer
from mujou.pc.time_constants import TimeConstants
from mujou.types import Device, Size


class Shown(ABC):
    """What ``show`` shows a sample: a network whose ``prev`` takes a ``Sample``
    and whose ``next`` sends a ``Prediction``.

    ``Chain`` is one. A network built by hand is one by inheriting this class
    as well as ``Network``, and assigning the ``prev`` of its ``Input`` and the
    ``next`` of its ``Output`` to ``prev`` and ``next`` in ``__init__``::

        class Tower(mujou.Network, pc.Shown):
            def __init__(self) -> None:
                super().__init__()
                ...
                self.prev, self.next = self.input.prev, self.output.next
    """

    prev: Port[Never, Sample]
    next: Port[Prediction, Never]


class Chain(Network, Shown):
    """``Input -> Layer -> ... -> Output``: a network that learns ``modules`` in
    sequence by PC-ALM, as ``nn.Sequential(*modules)`` would compute them.

    ``modules[:-1]`` each become a ``Layer`` and ``modules[-1]`` the ``Output``,
    after an ``Input`` of the samples ``x`` and targets ``y``. Each node takes
    the ``act_spec`` of the one before and the nodes are connected in order.
    The children are ``input``, ``layer0``, ..., ``output`` (``modules[i]`` is
    in ``layer{i}``, also ``layers[i]``), which are also their paths in
    ``Run.state_dict()``. Every
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
            batch=B,
            loss=pc.loss.CrossEntropyLoss(B),
            tau=pc.TimeConstants(h=4e-3, lam=8e-3),
            optimizer=pc.optim.SGD(tau=1.28),
            learn_interval=0.15,
        )
        with mujou.run(net) as run:
            prediction = pc.show(run, net, sample, seconds=0.2)

    Args:
        modules: The functions of the nodes, in order, on ``device``.
        x: The spec of the inputs, ``(*batch, *in)``.
        y: The spec of the targets, ``(*batch, *target)``.
        loss: The loss of each sample, as in ``Output``.
        tau: The time constants of every node.
        optimizer: How the parameters of every node move, as in ``Layer``:
            each node binds it to its own parameters.
        relaxer: How ``h`` of every ``Layer`` moves, as in ``Layer``: each binds
            it to its own ``h``. The ``Output`` has no ``h``.
        learn_interval: As in ``Layer``. Longer than the round trip of the
            credit, ``2 * len(modules) * tau.period``, is recommended.
        batch: The batch, as in ``Input``: ``()`` for one sample, ``B`` for
            ``B`` samples.
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
        optimizer: Optimizer,
        relaxer: Relaxer = Euler(),
        learn_interval: float | None,
        batch: Size = (),
        max_dt: float | None = None,
        device: Device = "cpu",
    ) -> None:
        super().__init__()
        self.input = Input(x=x, y=y, period=tau.period, batch=batch, device=device)
        last: Input | Layer = self.input
        for i, module in enumerate(modules[:-1]):
            layer = Layer(
                module,
                act_spec=last.act_spec,
                tau=tau,
                optimizer=optimizer,
                relaxer=relaxer,
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
            optimizer=optimizer,
            learn_interval=learn_interval,
            max_dt=max_dt,
            device=device,
        )
        last.next.connect(self.output.prev)
        self.prev = self.input.prev
        self.next = self.output.next

    @property
    def layers(self) -> tuple[Layer, ...]:
        """The ``Layer`` of each module but the last, in order: ``layers[i]`` is
        the child ``layer{i}``."""
        return tuple(v for v in vars(self).values() if isinstance(v, Layer))


def show(run: Run, chain: Shown, sample: Sample, *, seconds: float) -> Prediction:
    """Return the prediction ``chain`` makes for ``sample`` before it learns from
    it (a prequential prediction), after showing it ``seconds`` long.

    ``chain`` is a ``Shown``: a ``Chain``, or a network of the nodes of
    ``mujou.pc`` built by hand that inherits ``Shown``.
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
        p = pc.show(run, net, pc.msg.Sample(x=x, y=y, mask=mask), seconds=0.2)
        correct = int((p["y_hat"].argmax(1) == y).sum())

    Raises:
        ValueError: A tensor of ``sample`` has another shape or dtype than the
            spec of ``chain.prev``.
        TypeError: ``sample`` lacks a key, or has one too many.
        RuntimeError: The run shut down normally before the prediction.
        NetworkError: The run stopped because a node raised.
    """
    written = run.write({chain.prev: sample})
    prediction, _ = run.wait_for(chain.next, lambda p: p["time"] == written)
    run.timer.sleep(max(0.0, seconds - (run.timer.now() - written)))
    return prediction
