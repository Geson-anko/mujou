"""The messages of ``mujou.pc``: the activity a node sends to the next one, what
the host writes to an input node and what an output node sends."""

from typing import TypedDict

import torch

from mujou import DictSpec, TensorSpec, TimestampSpec
from mujou.time import Timestamp
from mujou.types import Size

__all__ = [
    "Sample",
    "Activity",
    "Prediction",
    "activity_spec",
]


class Activity(TypedDict):
    """The activity a node sends to the next nodes, with the target of each of
    its samples.

    The tensors start with the batch, ``batch``: ``()`` for one sample, ``(B,)``
    for ``B`` samples, or more dimensions (see ``activity_spec``).

    - ``h``: the activity, ``(*batch, *out)``.
    - ``y``: the target of each sample, ``(*batch, *target)``.
    - ``mask``: ``batch`` bool, whether each sample has a target. A sample whose
      ``mask`` is False adds nothing to the loss, whatever its ``y`` is.
    - ``reset``: ``batch`` bool, whether each sample starts anew: a node that
      first sees the activity sets the state it holds for those samples to
      zero (``r`` and ``lam`` of a ``Layer``). A sample that continues the one
      before it, as the next frame of a stream does, keeps it.
    - ``time``: the ``Timestamp`` that ``Run.write`` returned for the input
      ``h`` was computed from; ``Timestamp(ns=0)`` before the host first
      writes.

    ``y``, ``mask``, ``reset`` and ``time`` come from the ``Sample`` the host
    wrote, and every node passes them on with the ``h`` it computed from it, in
    the same message: the output node pairs each prediction with its own target,
    however fast the host writes.
    """

    h: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    reset: torch.Tensor
    time: Timestamp


class Sample(TypedDict):
    """What the host writes to an input node: the input ``x`` (``(*batch, *in)``),
    and the ``y``, ``mask`` and ``reset`` of ``Activity``.

    ``reset`` is True for the samples that do not continue the ones written
    before: every sample of shuffled batches, or the first frame of a new
    episode in a stream. The nodes act on it once, when they first see the
    write.
    """

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    reset: torch.Tensor


class Prediction(TypedDict):
    """What an output node sends: the prediction ``y_hat`` (``(*batch, *out)``), and the
    ``y``, ``mask`` and ``time`` of the activity it was computed from (see
    ``Activity``)."""

    y_hat: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    time: Timestamp


def activity_spec(h: TensorSpec, y: TensorSpec, batch: Size = ()) -> DictSpec[Activity]:
    """Return the spec of an ``Activity`` whose ``h`` is of the spec ``h``, whose
    targets are of the spec ``y`` and whose samples are ``batch``.

    ``batch`` is the shape that ``h`` and ``y`` start with, and that of
    ``mask``: ``()`` (the default) for one sample, ``B`` or ``(B,)`` for ``B``
    samples, or more dimensions, such as ``(E, A)`` for ``A`` agents in each
    of ``E`` environments. The samples must be independent of each other
    (``Layer``); a dimension that a module mixes, such as the time of a
    sequence, belongs to ``h``, not to the batch.

    Two calls with equal arguments return equal specs. A node of your own that
    connects to the nodes of ``mujou.pc`` declares its ports with it, or with
    the ``act_spec`` of the node before.

    Example::

        act = pc.msg.activity_spec(TensorSpec((256,)), TensorSpec((), torch.int64))
        act = pc.msg.activity_spec(
            TensorSpec((64, 256)), TensorSpec((64,), torch.int64), batch=64
        )

    Raises:
        ValueError: ``h`` or ``y`` does not start with ``batch``.
    """
    batch = batch_shape(batch)
    n = len(batch)
    if h.shape[:n] != batch or y.shape[:n] != batch:
        raise ValueError(
            f"h and y must start with the batch {batch}, got the shapes "
            f"{h.shape} and {y.shape}"
        )
    return DictSpec(
        Activity,
        h=h,
        y=y,
        mask=TensorSpec(batch, torch.bool),
        reset=TensorSpec(batch, torch.bool),
        time=TimestampSpec(),
    )


def batch_shape(batch: Size) -> tuple[int, ...]:
    """Return ``batch`` as a shape: an int ``B`` is ``(B,)``."""
    return (batch,) if isinstance(batch, int) else tuple(batch)


def rows(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Return the ``mask`` of an ``Activity`` (``batch``) shaped to select the
    samples of ``like`` (``(*batch, *shape)``) in ``torch.where``.

    It is here, with the ``Activity`` whose ``mask`` it shapes, for both
    ``functional`` and ``Layer``, which apply the mask."""
    return mask.reshape(mask.shape + (1,) * (like.dim() - mask.dim()))
