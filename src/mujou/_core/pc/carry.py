"""The messages of ``mujou.pc``: the activity a node sends to the next one, the
data carried with it, what the host writes to an input node and what an output
node sends."""

from collections.abc import Sequence
from typing import Any, TypedDict, cast

import torch

from mujou._core.spec import DictSpec, TensorSpec


class Activity(TypedDict):
    """The activity a node sends to the next nodes.

    - ``h``: the activity, ``(B, *out)``.
    - ``epoch``: ``()`` int64, the number of times the sender's ``h`` has jumped
      by more than its continuous relaxation: an ``Input`` counts the writes of
      the host, and a ``Layer`` counts its moves with a new sample and with a
      jump of the ``h`` it receives, and the jumps of its own ``theta`` when it
      learns at intervals (``learn_interval`` not ``None``). A change tells the
      receiver to move its own ``h`` with its prediction, so that the jump
      does not enter its deviation ``h - mu`` (``Layer``).
    """

    h: torch.Tensor
    epoch: torch.Tensor


class Carried(TypedDict):
    """The data that travels with an activity from the input node to the output
    node, relayed by every node on the way.

    - ``y``: the target of each sample, ``(B, *target)``, of the spec the user
      gives to ``carry_spec``.
    - ``mask``: ``(B,)`` bool, whether each sample has a target. A sample whose
      ``mask`` is False adds nothing to the loss, whatever its ``y`` is.
    - ``time``: ``()`` int64, the time in nanoseconds at which the host wrote the
      input that the activity was computed from.
    """

    y: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor


class Sample(TypedDict):
    """What the host writes to an input node: the input ``x`` (``(B, *in)``), and
    the ``y`` and ``mask`` of ``Carried``."""

    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor


class Prediction(TypedDict):
    """What an output node sends: the prediction ``y_hat`` (``(B, *out)``), and the
    ``y``, ``mask`` and ``time`` of the input it was computed from (see
    ``Carried``)."""

    y_hat: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    time: torch.Tensor


def carry_spec(target: TensorSpec, batch: int) -> DictSpec[Carried]:
    """Return the spec of the data carried with an activity of ``batch`` samples.

    Two calls with equal arguments return equal specs, so each node may make its
    own.

    Example::

        carry = carry_spec(TensorSpec((64,), torch.int64), batch=64)  # class labels

    Args:
        target: The spec of the targets ``y``, ``(batch, *target)``.
        batch: The number of samples ``B`` of an activity.

    Raises:
        TypeError: ``target`` is not a ``TensorSpec``, or ``batch`` is not an
            ``int``.
        ValueError: ``batch`` is less than 1, or the first dimension of
            ``target`` is not ``batch``.
    """
    # The annotation is not checked at runtime.
    if not isinstance(target, TensorSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"target must be a TensorSpec, got {type(target).__name__}")
    mask = TensorSpec((batch,), torch.bool)  # checks batch
    if target.shape[:1] != (batch,):
        raise ValueError(
            f"the first dimension of target must be the batch {batch}, "
            f"got shape {target.shape}"
        )
    return DictSpec(Carried, y=target, mask=mask, time=TensorSpec((), torch.int64))


def activity_spec(h: TensorSpec) -> DictSpec[Activity]:
    """Return the spec of an ``Activity`` whose ``h`` is of the spec ``h``."""
    return DictSpec(Activity, h=h, epoch=TensorSpec((), torch.int64))


def batch_of(carry: object) -> int:
    """Return the batch ``B`` of a spec made by ``carry_spec``, for the nodes that
    take one.

    Raises:
        TypeError: ``carry`` is not a ``DictSpec`` of ``Carried``.
    """
    # Any: only the TypedDict is compared.
    if not (
        isinstance(carry, DictSpec) and cast(DictSpec[Any], carry).typed_dict is Carried
    ):
        raise TypeError(
            f"carry must be a DictSpec of Carried, as carry_spec makes it, got {carry!r}"
        )
    return carry.specs["mask"].shape[0]


def check_batch(shape: Sequence[int], batch: int, what: str) -> None:
    """Check that ``shape`` starts with the batch of the carried data.

    Raises:
        ValueError: The first dimension of ``shape`` is not ``batch``.
    """
    if tuple(shape)[:1] != (batch,):
        raise ValueError(
            f"the first dimension of {what} must be the batch {batch} of carry, "
            f"got shape {tuple(shape)}"
        )
