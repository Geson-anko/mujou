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

from collections.abc import Callable, Iterable, Mapping, Sequence
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
    device: torch.device | None = None,
    wait: bool = True,
) -> list[object]:
    """The values of ``msg`` in field order for the core: tensors detached,
    and scalars (a ``Timestamp`` as its nanoseconds). ``label`` names the
    message in the errors.

    ``device`` is the CUDA device where the dense tensors on it stay, or
    ``None``. It is the device of the node that reads the values when the
    sender knows it, and that node is on CUDA: a node that sends to a node on
    CUDA (``Wire.cuda``), or the host writing to the port of such a node
    (``Wiring.write``). The core holds a contiguous clone of each such tensor.
    The clones queue on the calling thread's current stream of the device (a
    node's own, in its hooks), which this waits for once, so that no reader
    sees a clone unfinished; with ``wait=False`` the caller waits with
    ``wait_for_clones`` instead (once for the messages of one write). While
    the thread's current device is another (the host's, or a hook's inside
    ``torch.cuda.device(...)``), each clone exports itself with its device
    current, as torch exports a CUDA tensor over DLPack only on its current
    device.

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
    out = [_value(f, values[name], label, name, device) for name, f in names.items()]
    if device is None:
        return out
    # Every other tensor is in host memory by now.
    if torch.cuda.current_device() != device.index:
        out = [
            _OnItsDevice(v) if isinstance(v, torch.Tensor) and v.is_cuda else v
            for v in out
        ]
    if wait:
        wait_for_clones([out])
    return out


def wait_for_clones(messages: Iterable[Sequence[object]]) -> None:
    """Wait for the clones of ``messages`` (from ``to_core``): for the calling
    thread's current stream of each device they are on, once. Only that
    stream: past PyTorch's pool of 32 streams per device, nodes share one, and
    the event does not wait for what another node queues on it later."""
    devices = {
        v.device if isinstance(v, torch.Tensor) else v.tensor.device
        for m in messages
        for v in m
        if (isinstance(v, torch.Tensor) and v.is_cuda) or type(v) is _OnItsDevice
    }
    for d in devices:
        torch.cuda.current_stream(d).record_event().synchronize()


@final
class _OnItsDevice:
    """A CUDA tensor that the core holds, which makes its device current while
    it exports itself over DLPack: torch requires that, and the sender's thread
    has another current device."""

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
    stays_on: torch.device | None,
) -> object:
    """``value`` as the core takes it for a field of ``spec``: the values that
    the field's annotation takes in Python's typing (a ``bool`` is an ``int``,
    an ``int`` is a ``float``). A dense tensor on ``stays_on`` is cloned there."""
    match spec:
        case TensorSpec() if isinstance(value, torch.Tensor):
            if (
                stays_on is not None
                and value.device == stays_on
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
# its sender's thread had another current device).
type Views = Sequence[_native.BufferView | torch.Tensor | _OnItsDevice]


def from_core(spec: MessageSpec[Any], views: Views, device: torch.device) -> object:
    """A message of ``spec`` made of copies of ``views``: tensors on ``device``,
    and scalars.

    A tensor that the core holds (on the CUDA device of the node or the host
    that wrote it) is copied on the calling thread's current stream of that
    device, without passing through host memory to another CUDA device.
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
