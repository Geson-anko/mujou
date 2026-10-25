"""The one conversion between torch and the native core (``specs/messages.md``).

A message is a list of values in field order: a message of one field (a tensor
or a scalar) is the field named ``""``, a dict message has one field per key.
Only the structure of a value (the kind of each field, and a dict's keys) is
checked here; the core checks each tensor's shape and dtype and each int's range
when it copies the value in, and names the slot and the field.

Values cross in host memory, except that a node on a CUDA device sends a tensor
on that device as a clone on the device, which the core holds as it is
(``to_core``), and which readers copy device to device (``from_core``).
"""

from collections.abc import Callable, Mapping, Sequence
from enum import Enum, auto
from typing import Any, Final, cast, final

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


def to_core(
    spec: MessageSpec[Any],
    msg: object,
    label: str,
    *,
    device: torch.device | Their | None = None,
) -> list[object]:
    """The values of ``msg`` in field order for the core: tensors detached,
    and scalars (a ``Timestamp`` as its nanoseconds). ``label`` names the
    message in the errors.

    ``device`` says which dense tensors stay on their CUDA device, where the
    core holds a contiguous clone of each:

    - a CUDA device: those on it. A node on that device passes it when a node
      on CUDA reads the port (``Wire.cuda``). The clones queue on the calling
      thread's current stream of the device (the node's own, in its hooks),
      which this waits for once, so that no reader sees a clone unfinished.
      While the thread's current device is another (inside a
      ``torch.cuda.device(...)`` of the hook), none stays: torch exports a CUDA
      tensor over DLPack only on its current device
    - ``THEIR_DEVICES`` (the host's write): every one, each on its own device.
      This waits for the calling thread's current stream of each device, and
      each clone exports itself with its device current
    - ``None`` (any other sender, ``initial`` and a restore): none

    Any other tensor is copied to host memory on the current stream of its
    device, and only that stream is synchronized.

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
    if device is THEIR_DEVICES:
        out = [
            _value(f, values[name], label, name, device) for name, f in names.items()
        ]
        return _on_their_devices(out)
    if device is not None and torch.cuda.current_device() != device.index:
        device = None
    out = [_value(f, values[name], label, name, device) for name, f in names.items()]
    # Every other tensor is in host memory by now.
    if device is not None and any(
        isinstance(v, torch.Tensor) and v.is_cuda for v in out
    ):
        # One wait for every clone, on this stream only: past PyTorch's pool of
        # 32 streams per device, nodes share one, and the event does not wait
        # for what another node queues on it later.
        torch.cuda.current_stream(device).record_event().synchronize()
    return out


class Their(Enum):
    """The kind of ``THEIR_DEVICES``."""

    DEVICES = auto()


# For to_core: every dense CUDA tensor stays on its own device.
THEIR_DEVICES: Final = Their.DEVICES


def _on_their_devices(out: list[object]) -> list[object]:
    """``out`` with each CUDA tensor (a clone) wrapped to export itself with
    its device current, after waiting for the calling thread's current stream
    of each device, where the clones were queued."""
    devices = {v.device for v in out if isinstance(v, torch.Tensor) and v.is_cuda}
    for d in devices:
        with torch.cuda.device(d):
            torch.cuda.current_stream(d).record_event().synchronize()
    return [
        _OnItsDevice(v) if isinstance(v, torch.Tensor) and v.is_cuda else v for v in out
    ]


@final
class _OnItsDevice:
    """A CUDA tensor that the core holds, which makes its device current while
    it exports itself over DLPack: torch requires that, and the host's thread
    may have another current device."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def __dlpack__(self, **kwargs: Any) -> object:  # Any: torch's own arguments
        with torch.cuda.device(self.tensor.device):
            # Partly unknown in torch's stubs.
            return cast(object, self.tensor.__dlpack__(**kwargs))


def _values(msg: object) -> Mapping[str, object]:
    """The values of ``msg`` by field name: anything but a dict is the one field
    ``""``."""
    return cast(Mapping[str, object], msg) if isinstance(msg, Mapping) else {"": msg}


def _keys(values: Mapping[str, object]) -> str:
    return f"a dict with the keys {list(values)}"


# Where peek copies what the core holds in host memory.
_CPU: Final = torch.device("cpu")

# What each kind of field takes, as the errors say it.
_EXPECTED: Mapping[type, str] = {
    TensorSpec: "a torch.Tensor",
    IntSpec: "an int",
    FloatSpec: "a float",
    BoolSpec: "a bool",
    TimestampSpec: "a Timestamp",
}


def _value(
    spec: FieldSpec,
    value: object,
    label: str,
    name: str,
    stays_on: torch.device | Their | None,
) -> object:
    """``value`` as the core takes it for a field of ``spec``: the values that
    the field's annotation takes in Python's typing (a ``bool`` is an ``int``,
    an ``int`` is a ``float``). A dense tensor that stays on its device (as
    ``to_core`` says of ``stays_on``) is cloned there."""
    match spec:
        case TensorSpec() if isinstance(value, torch.Tensor):
            if (
                stays_on is not None
                # A node's device first: the hot path.
                and (
                    value.device == stays_on
                    or (stays_on is THEIR_DEVICES and value.is_cuda)
                )
                and value.layout == torch.strided
            ):
                return value.detach().clone(memory_format=torch.contiguous_format)
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


# The values of a message from the core (``Message.values``): a view of a value
# in host memory, or a tensor that the core holds on a CUDA device (wrapped if
# the host wrote it).
type Views = Sequence[_native.BufferView | torch.Tensor | _OnItsDevice]


def from_core(spec: MessageSpec[Any], views: Views, device: torch.device) -> object:
    """A message of ``spec`` made of copies of ``views``: tensors on ``device``,
    and scalars.

    A tensor that the core holds (on the CUDA device of the node or the host
    that wrote it) is copied on the calling thread's current stream of that device, without
    passing through host memory to another CUDA device.
    """
    names = fields(spec)
    values = [
        _read(f, view, device) for f, view in zip(names.values(), views, strict=True)
    ]
    if isinstance(spec, DictSpec):
        return dict(zip(names, values, strict=True))
    (value,) = values
    return value


def peek(spec: MessageSpec[Any], views: Views) -> object:
    """A message of ``spec`` as ``from_core`` makes it on the CPU, except that a
    tensor the core holds comes as it is, uncopied: for a paused run, whose
    tensors the caller copies to the CPU once it resumes (``state.cpu``)."""
    names = fields(spec)
    values = [
        view
        if isinstance(view, torch.Tensor)
        else view.tensor
        if isinstance(view, _OnItsDevice)
        else _read(f, view, _CPU)
        for f, view in zip(names.values(), views, strict=True)
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


def _read(
    spec: FieldSpec,
    view: _native.BufferView | torch.Tensor | _OnItsDevice,
    device: torch.device,
) -> object:
    match spec:
        case TensorSpec():
            # Not isinstance, which takes longer (hot path).
            if type(view) is _native.BufferView:
                return torch.from_dlpack(view).to(device, copy=True)
            # A tensor that the core holds on a CUDA device.
            held = view.tensor if type(view) is _OnItsDevice else view
            copy = held.to(device, copy=True)
            # Its memory goes back to the caching allocator once the core drops
            # it, and is reused only after the copy queued here.
            held.record_stream(torch.cuda.current_stream(held.device))
            return copy
        case TimestampSpec():
            # A scalar is always in host memory.
            return _native.Timestamp(ns=cast("_native.BufferView", view).item())
        case _:
            # A scalar is always in host memory: a BufferView (not cast, on the
            # hot path).
            return view.item()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
