"""The messages of ``mujou.pc``: the activity a node sends to the next one, what
the host writes to an input node and what an output node sends."""

from typing import TypedDict

import torch

from mujou._core.spec import DictSpec, TensorSpec


class Activity(TypedDict):
    """The activity a node sends to the next nodes, with the target of each of
    its samples.

    - ``h``: the activity, ``(B, *out)``.
    - ``epoch``: ``()`` int64, the number of times the sender's ``h`` has jumped
      by more than its continuous relaxation: an ``Input`` counts the writes of
      the host, and a ``Layer`` counts its moves with a new sample and with a
      jump of the ``h`` it receives, and the jumps of its own ``theta`` when it
      learns at intervals (``learn_interval`` not ``None``). A change tells the
      receiver to move its own ``h`` with its prediction, so that the jump
      does not enter its deviation ``h - mu`` (``Layer``).
    - ``y``: the target of each sample, ``(B, *target)``.
    - ``mask``: ``(B,)`` bool, whether each sample has a target. A sample whose
      ``mask`` is False adds nothing to the loss, whatever its ``y`` is.
    - ``time``: ``()`` int64, the time in nanoseconds at which the host wrote the
      input that ``h`` was computed from.

    ``y``, ``mask`` and ``time`` come from the ``Sample`` the host wrote, and
    every node passes them on with the ``h`` it computed from it, in the same
    message: the output node pairs each prediction with its own target, however
    fast the host writes.
    """

    h: torch.Tensor
    epoch: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor


class Sample(TypedDict):
    """What the host writes to an input node: the input ``x`` (``(B, *in)``), and
    the ``y`` and ``mask`` of ``Activity``."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor


class Prediction(TypedDict):
    """What an output node sends: the prediction ``y_hat`` (``(B, *out)``), and the
    ``y``, ``mask`` and ``time`` of the activity it was computed from (see
    ``Activity``)."""

    y_hat: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor


def activity_spec(h: TensorSpec, y: TensorSpec) -> DictSpec[Activity]:
    """Return the spec of an ``Activity`` whose ``h`` is of the spec ``h`` and whose
    targets are of the spec ``y``.

    Two calls with equal arguments return equal specs. A node of your own that
    connects to the nodes of ``mujou.pc`` declares its ports with it, or with
    the ``act`` of the node before.

    Example::

        act = pc.activity_spec(TensorSpec((64, 256)), TensorSpec((64,), torch.int64))

    Raises:
        ValueError: The first dimensions of ``h`` and ``y`` (the batch ``B``)
            differ.
    """
    batch = y.shape[:1]
    if h.shape[:1] != batch:
        raise ValueError(
            f"h and y must have the same batch (first dimension), got the shapes "
            f"{h.shape} and {y.shape}"
        )
    return DictSpec(
        Activity,
        h=h,
        epoch=TensorSpec((), torch.int64),
        y=y,
        mask=TensorSpec(batch, torch.bool),
        time=TensorSpec((), torch.int64),
    )
