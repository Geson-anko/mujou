"""The one conversion between torch and the native core (``specs/messages.md``).

A message is a list of tensors in field order: a tensor message is the one
field named ``""``, a dict message has one field per key. Only the structure of
a value (a tensor, or a dict of tensors with the spec's keys) is checked here;
the core checks each field's shape and dtype when it copies the value in, and
names the slot and the field.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from mujou import _native
from mujou.spec import DLPACK, DictSpec, MessageSpec, TensorSpec


def fields(spec: MessageSpec[Any]) -> Mapping[str, TensorSpec]:
    """The fields of a message of ``spec`` by name, in field order: a tensor
    message is the one field named ``""``."""
    if isinstance(spec, TensorSpec):
        return {"": spec}
    return cast("DictSpec[Any]", spec).specs  # the other kind (quoted: hot path)


def schema(spec: MessageSpec[Any]) -> _native.Schema:
    """The core's schema of messages of ``spec``."""
    return _native.Schema(
        [(name, f.shape, DLPACK[f.dtype]) for name, f in fields(spec).items()]
    )


def to_core(spec: MessageSpec[Any], msg: object, label: str) -> list[torch.Tensor]:
    """The tensors of ``msg`` in field order, detached and in host memory, for
    the core to copy. ``label`` names the message in the errors.

    A CUDA tensor is copied on the calling thread's current stream (a node's
    own, in its hooks), and only that stream is synchronized.

    Raises:
        TypeError: ``msg`` is not a tensor for a tensor message, or not a dict
            of tensors with the spec's keys for a dict message.
    """
    names = fields(spec)
    values = _values(msg)
    if values.keys() != names.keys():
        expected = "a torch.Tensor" if "" in names else _keys(names)
        got = type(values[""]).__name__ if "" in values else _keys(values)
        raise TypeError(f"{label}: expected {expected}, got {got}")
    return [_tensor(values[name], label, name).detach().to("cpu") for name in names]


def _values(msg: object) -> Mapping[str, object]:
    """The values of ``msg`` by field name: anything but a dict is the one field
    ``""``."""
    return cast(Mapping[str, object], msg) if isinstance(msg, Mapping) else {"": msg}


def _keys(values: Mapping[str, object]) -> str:
    return f"a dict with the keys {list(values)}"


def _tensor(value: object, label: str, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        at = f"{label}[{name!r}]" if name else label
        raise TypeError(f"{at}: expected a torch.Tensor, got {type(value).__name__}")
    return value


def from_core(
    spec: MessageSpec[Any], views: Sequence[_native.BufferView], device: torch.device
) -> object:
    """A message of ``spec`` made of copies of ``views`` on ``device``."""
    tensors = [torch.from_dlpack(view).to(device, copy=True) for view in views]
    if isinstance(spec, TensorSpec):
        (tensor,) = tensors
        return tensor
    return dict(zip(fields(spec), tensors, strict=True))
