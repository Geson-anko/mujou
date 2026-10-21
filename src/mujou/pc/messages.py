"""The messages of ``mujou.pc``: the activity a node sends to the next one, what
the host writes to an input node and what an output node sends."""

from typing import TypedDict

import torch

from mujou import DictSpec, TensorSpec
from mujou.types import Size


class Activity(TypedDict):
    """The activity a node sends to the next nodes, with the target of each of
    its samples.

    The shapes start with the batch, ``batch``: ``()`` for one sample, ``(B,)``
    for ``B`` samples, or more dimensions (see ``activity_spec``).

    - ``h``: the activity, ``(*batch, *out)``.
    - ``jumps``: ``()`` int64, the number of times the sender's ``h`` has jumped
      rather than relaxed continuously: an ``Input`` counts the writes of the
      host, and a ``Layer`` counts its moves with a new sample and with a jump
      of the ``h`` it receives, and the jumps of its own ``theta`` when it
      learns at intervals (``learn_interval`` not ``None``). A change tells the
      receiver to move its own ``h`` with its prediction, so that the jump
      does not enter its deviation ``h - mu`` (``Layer``).
    - ``y``: the target of each sample, ``(*batch, *target)``.
    - ``mask``: ``batch`` bool, whether each sample has a target. A sample whose
      ``mask`` is False adds nothing to the loss, whatever its ``y`` is.
    - ``time``: ``()`` int64, the time in nanoseconds that ``Run.write``
      returned for the input ``h`` was computed from; 0 before the host first
      writes.

    ``y``, ``mask`` and ``time`` come from the ``Sample`` the host wrote, and
    every node passes them on with the ``h`` it computed from it, in the same
    message: the output node pairs each prediction with its own target, however
    fast the host writes.
    """

    h: torch.Tensor
    jumps: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor


class Sample(TypedDict):
    """What the host writes to an input node: the input ``x`` (``(*batch, *in)``),
    and the ``y`` and ``mask`` of ``Activity``."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor


class Prediction(TypedDict):
    """What an output node sends: the prediction ``y_hat`` (``(*batch, *out)``), and the
    ``y``, ``mask`` and ``time`` of the activity it was computed from (see
    ``Activity``)."""

    y_hat: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor


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

        act = pc.activity_spec(TensorSpec((256,)), TensorSpec((), torch.int64))
        act = pc.activity_spec(
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
        jumps=TensorSpec((), torch.int64),
        y=y,
        mask=TensorSpec(batch, torch.bool),
        time=TensorSpec((), torch.int64),
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
