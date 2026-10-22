"""The one conversion between torch and the native core (``specs/messages.md``).

A message is a list of values in field order: a message of one field (a tensor
or a scalar) is the field named ``""``, a dict message has one field per key.
Only the structure of a value (the kind of each field, and a dict's keys) is
checked here; the core checks each tensor's shape and dtype and each int's range
when it copies the value in, and names the slot and the field.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch

from mujou import _native
from mujou.spec import (
    DLPACK,
    BoolSpec,
    DictSpec,
    FieldSpec,
    FloatSpec,
    IntSpec,
    MessageSpec,
    TensorSpec,
    TimestampSpec,
)


def fields(spec: MessageSpec[Any]) -> Mapping[str, FieldSpec]:
    """The fields of a message of ``spec`` by name, in field order: a message of
    one field is the field named ``""``."""
    if isinstance(spec, DictSpec):
        return spec.specs
    return {"": cast("FieldSpec", spec)}  # every other spec (quoted: hot path)


def schema(spec: MessageSpec[Any]) -> _native.Schema:
    """The core's schema of messages of ``spec``."""
    return _native.Schema([(name, *_layout(f)) for name, f in fields(spec).items()])


def _layout(spec: FieldSpec) -> tuple[tuple[int, ...], tuple[int, int, int]]:
    """The shape of a field and its DLPack dtype: a scalar is of shape ``()``."""
    match spec:
        case TensorSpec():
            return spec.shape, DLPACK[spec.dtype]
        case IntSpec():
            return (), (0 if spec.signed else 1, spec.bits, 1)
        case FloatSpec():
            return (), DLPACK[torch.float64]
        case BoolSpec():
            return (), DLPACK[torch.bool]
        case TimestampSpec():
            return (), (1, 64, 1)  # uint64 nanoseconds


def to_core(spec: MessageSpec[Any], msg: object, label: str) -> list[object]:
    """The values of ``msg`` in field order for the core to copy: tensors
    detached and in host memory, and scalars (a ``Timestamp`` as its
    nanoseconds). ``label`` names the message in the errors.

    A CUDA tensor is copied on the calling thread's current stream (a node's
    own, in its hooks), and only that stream is synchronized.

    Raises:
        TypeError: ``msg`` is not a value of the spec's kind for a message of
            one field, or not a dict of such values with the spec's keys for a
            dict message.
    """
    names = fields(spec)
    values = _values(msg)
    if values.keys() != names.keys():
        expected = _EXPECTED[type(names[""])] if "" in names else _keys(names)
        got = type(values[""]).__name__ if "" in values else _keys(values)
        raise TypeError(f"{label}: expected {expected}, got {got}")
    return [_value(f, values[name], label, name) for name, f in names.items()]


def _values(msg: object) -> Mapping[str, object]:
    """The values of ``msg`` by field name: anything but a dict is the one field
    ``""``."""
    return cast(Mapping[str, object], msg) if isinstance(msg, Mapping) else {"": msg}


def _keys(values: Mapping[str, object]) -> str:
    return f"a dict with the keys {list(values)}"


# What each kind of field takes, as the errors say it.
_EXPECTED: Mapping[type, str] = {
    TensorSpec: "a torch.Tensor",
    IntSpec: "an int",
    FloatSpec: "a float",
    BoolSpec: "a bool",
    TimestampSpec: "a Timestamp",
}


def _value(spec: FieldSpec, value: object, label: str, name: str) -> object:
    """``value`` as the core takes it for a field of ``spec``: the values that
    the field's annotation takes in Python's typing (a ``bool`` is an ``int``,
    an ``int`` is a ``float``)."""
    match spec:
        case TensorSpec() if isinstance(value, torch.Tensor):
            return value.detach().to("cpu")
        case IntSpec() if isinstance(value, int):
            return value
        case FloatSpec() if isinstance(value, (int, float)):
            return value
        case BoolSpec() if isinstance(value, bool):
            return value
        case TimestampSpec() if isinstance(value, _native.Timestamp):
            return value.ns
        case _:
            at = f"{label}[{name!r}]" if name else label
            raise TypeError(
                f"{at}: expected {_EXPECTED[type(spec)]}, got {type(value).__name__}"
            )


def from_core(
    spec: MessageSpec[Any], views: Sequence[_native.BufferView], device: torch.device
) -> object:
    """A message of ``spec`` made of copies of ``views``: tensors on ``device``,
    and scalars."""
    names = fields(spec)
    values = [
        _read(f, view, device) for f, view in zip(names.values(), views, strict=True)
    ]
    if isinstance(spec, DictSpec):
        return dict(zip(names, values, strict=True))
    (value,) = values
    return value


def times_to_ns(spec: MessageSpec[Any], msg: object) -> object:
    """``msg`` with each ``Timestamp`` of a ``TimestampSpec`` field as its
    nanoseconds, as the state tree holds every time: ``torch.load`` with
    ``weights_only=True`` does not read a ``Timestamp``."""
    return _map_times(spec, msg, lambda t: cast(_native.Timestamp, t).ns)


def times_from_ns(spec: MessageSpec[Any], msg: object) -> object:
    """The inverse of ``times_to_ns``. Leaves anything it cannot convert to
    ``to_core``, which names what is wrong."""
    return _map_times(
        spec, msg, lambda ns: _native.Timestamp(ns=ns) if isinstance(ns, int) else ns
    )


def _map_times(
    spec: MessageSpec[Any], msg: object, f: Callable[[object], object]
) -> object:
    if isinstance(spec, TimestampSpec):
        return f(msg)
    if isinstance(spec, DictSpec) and isinstance(msg, Mapping):
        specs = spec.specs
        return {
            k: f(v) if isinstance(specs.get(k), TimestampSpec) else v
            for k, v in cast(Mapping[str, object], msg).items()
        }
    return msg


def _read(spec: FieldSpec, view: _native.BufferView, device: torch.device) -> object:
    match spec:
        case TensorSpec():
            return torch.from_dlpack(view).to(device, copy=True)
        case TimestampSpec():
            return _native.Timestamp(ns=view.item())
        case _:
            return view.item()
