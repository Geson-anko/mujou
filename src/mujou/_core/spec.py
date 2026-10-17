"""Specs of the messages that flow between nodes.

A message is either one ``torch.Tensor`` (``TensorSpec``) or a ``TypedDict`` whose
values are ``torch.Tensor`` (``DictSpec``). A spec fixes the shape and the dtype of
every value, so the slots that carry the message can be laid out when the network
is built (``specs/messages.md``).
"""

import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, Generic, TypeVar, cast, final, override

import torch

# The supported dtypes and their DLPack ``DLDataType`` (code, bits, lanes).
DLPACK_DTYPES: Final[Mapping[torch.dtype, tuple[int, int, int]]] = MappingProxyType(
    {
        # (DLDataTypeCode, bits, lanes). Codes: 0 int, 1 uint, 2 float, 4 bfloat,
        # 6 bool.
        torch.float16: (2, 16, 1),
        torch.bfloat16: (4, 16, 1),
        torch.float32: (2, 32, 1),
        torch.float64: (2, 64, 1),
        torch.int8: (0, 8, 1),
        torch.int16: (0, 16, 1),
        torch.int32: (0, 32, 1),
        torch.int64: (0, 64, 1),
        torch.uint8: (1, 8, 1),
        torch.bool: (6, 8, 1),
    }
)


# The message type, declared invariant: the inferred variance of a PEP 695
# parameter would be covariant, and a port's ``initial`` (of type ``S``) could then
# widen the message type solved from its ``send`` spec.
_M = TypeVar("_M")


class MessageSpec(ABC, Generic[_M]):
    """The spec of a message of type ``_M``: its shape, dtype and initial value."""

    @abstractmethod
    def zeros(self, *, device: torch.device | str = "cpu") -> _M:
        """Return a message filled with zeros on ``device``.

        On the CPU, this is the default initial value that a port receives before
        the first message is written.
        """

    @abstractmethod
    def empty(self, *, device: torch.device | str = "cpu") -> _M:
        """Return a message on ``device`` whose values are not set.

        Set them before use, in place (``fill_`` and the like).
        """


@final
class TensorSpec(MessageSpec[torch.Tensor]):
    """The shape and the dtype of a message that is one ``torch.Tensor``.

    ``dtype`` defaults to ``torch.get_default_dtype()`` at the time the spec is
    created; changing the default later does not change existing specs. Every
    dimension is an ``int`` of at least 1, and ``()`` is a scalar. A batch dimension
    is just a part of the shape.

    Two specs are equal when their shapes and dtypes are equal. A message whose shape
    or dtype differs from the spec is an error; dtypes are never converted. A spec
    has no device: a message of the spec may be on any device.

    ``zeros``, ``rand`` and ``empty`` make a tensor of the spec, on the CPU unless
    ``device`` says otherwise. For any other initial value, fill one of them in
    place (``spec.empty().fill_(0.5)``).

    Example::

        spec = TensorSpec((32, 256))  # the default dtype, usually float32
        spec = TensorSpec((32, 256), torch.bfloat16)
        x0 = spec.zeros()
        noise = TensorSpec((4,)).rand(seed=0)  # deterministic
        ones = spec.empty(device="cuda").fill_(1.0)

    Raises:
        TypeError: A dimension is not an ``int``, or ``dtype`` is not supported.
        ValueError: A dimension is less than 1.
    """

    def __init__(self, shape: Sequence[int], dtype: torch.dtype | None = None) -> None:
        # User input: copy first (an iterator would be consumed by the checks), then
        # check the types at runtime.
        dims = tuple(cast(Sequence[object], shape))
        for d in dims:
            if isinstance(d, bool) or not isinstance(d, int):
                raise TypeError(
                    f"TensorSpec dimensions must be int, got {d!r} in shape {dims!r}"
                )
            if d < 1:
                raise ValueError(
                    f"TensorSpec dimensions must be >= 1, got {d} in shape {dims!r}"
                )
        if dtype is None:
            dtype = torch.get_default_dtype()
        if dtype not in DLPACK_DTYPES:
            supported = ", ".join(str(t) for t in DLPACK_DTYPES)
            raise TypeError(
                f"TensorSpec dtype {dtype} is not supported; use one of {supported}"
            )
        self._shape = cast(tuple[int, ...], dims)
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the tensor."""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the tensor."""
        return self._dtype

    @override
    def zeros(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return a tensor of this spec filled with zeros on ``device``."""
        return torch.zeros(self._shape, dtype=self._dtype, device=device)

    @override
    def empty(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return a tensor of this spec on ``device`` whose values are not set.

        Set them before use, in place: ``spec.empty().fill_(0.5)``.
        """
        return torch.empty(self._shape, dtype=self._dtype, device=device)

    def rand(
        self,
        seed: int | None = None,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return a tensor of this spec drawn uniformly from [0, 1) on ``device``.

        Pass ``seed`` or a ``generator`` on ``device`` to make the result
        deterministic; without either, the global torch generator is used. The
        same seed draws different values on different kinds of device.

        Raises:
            TypeError: The dtype is not a floating point dtype.
            ValueError: Both ``seed`` and ``generator`` are given.
        """
        if not self._dtype.is_floating_point:
            raise TypeError(f"rand needs a floating point dtype, got {self._dtype}")
        if seed is not None:
            if generator is not None:
                raise ValueError("pass either seed or generator, not both")
            generator = torch.Generator(device).manual_seed(seed)
        return torch.rand(
            self._shape, dtype=self._dtype, generator=generator, device=device
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorSpec):
            return NotImplemented
        return self._shape == other._shape and self._dtype == other._dtype

    @override
    def __hash__(self) -> int:
        return hash((self._shape, self._dtype))

    @override
    def __repr__(self) -> str:
        return f"TensorSpec({self._shape}, {self._dtype})"


@final
class DictSpec[TD: Mapping[str, object]](MessageSpec[TD]):
    """The spec of a message that is a ``TypedDict`` of ``torch.Tensor`` values.

    Give the ``TypedDict`` class and one ``TensorSpec`` per key. Every key of the
    ``TypedDict`` must be required and annotated as ``torch.Tensor`` (no nesting),
    and the keyword arguments must name exactly its keys. The keys keep the order in
    which the ``TypedDict`` defines them.

    Two specs are equal when they use the same ``TypedDict`` class and equal
    ``TensorSpec`` for every key.

    Example::

        class Feedback(TypedDict):
            error: torch.Tensor
            target: torch.Tensor


        spec = DictSpec(
            Feedback, error=TensorSpec((32, 256)), target=TensorSpec((32, 10))
        )
        msg: Feedback = spec.zeros()

    Raises:
        TypeError: ``typed_dict`` is not a ``TypedDict``, a key is not required, a
            value is not annotated as ``torch.Tensor``, or a spec is not a
            ``TensorSpec``.
        ValueError: The ``TypedDict`` has no key, or the keyword arguments miss a
            key or name an unknown key.
    """

    def __init__(self, typed_dict: type[TD], /, **specs: TensorSpec) -> None:
        if not typing.is_typeddict(typed_dict):
            raise TypeError(f"DictSpec needs a TypedDict class, got {typed_dict!r}")
        name = typed_dict.__qualname__
        optional: frozenset[str] = getattr(typed_dict, "__optional_keys__")
        if optional:
            raise TypeError(
                f"every key of {name} must be required, got optional keys "
                f"{sorted(optional)}"
            )
        hints = typing.get_type_hints(typed_dict)
        if not hints:
            raise ValueError(f"DictSpec needs a TypedDict with keys, {name} has none")
        for key, hint in hints.items():
            if hint is not torch.Tensor:
                raise TypeError(
                    f"{name}[{key!r}] must be annotated as torch.Tensor, got {hint!r}"
                )
        missing = [k for k in hints if k not in specs]
        unknown = [k for k in specs if k not in hints]
        if missing or unknown:
            raise ValueError(
                f"DictSpec({name}) keys do not match the TypedDict: "
                f"missing {missing}, unknown {unknown}"
            )
        values = cast(Mapping[str, object], specs)  # user input: check at runtime
        for key, spec in values.items():
            if not isinstance(spec, TensorSpec):
                raise TypeError(
                    f"DictSpec({name}) needs a TensorSpec for {key!r}, "
                    f"got {type(spec).__name__}"
                )
        self._typed_dict = typed_dict
        self._specs: Mapping[str, TensorSpec] = MappingProxyType(
            {k: specs[k] for k in hints}
        )

    @property
    def typed_dict(self) -> type[TD]:
        """The ``TypedDict`` class of the message."""
        return self._typed_dict

    @property
    def specs(self) -> Mapping[str, TensorSpec]:
        """The ``TensorSpec`` of each key, in the order the ``TypedDict`` defines."""
        return self._specs

    @override
    def zeros(self, *, device: torch.device | str = "cpu") -> TD:
        """Return a message whose values are all zeros on ``device``."""
        return cast(TD, {k: s.zeros(device=device) for k, s in self._specs.items()})

    @override
    def empty(self, *, device: torch.device | str = "cpu") -> TD:
        """Return a message on ``device`` whose values are not set.

        Set them before use, in place: ``msg["error"].fill_(0.5)``.
        """
        return cast(TD, {k: s.empty(device=device) for k, s in self._specs.items()})

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictSpec):
            return NotImplemented
        # Any: the other spec may be of any TypedDict; identity decides.
        other_spec = cast(DictSpec[Any], other)
        # The same TypedDict fixes the key order, so comparing as mappings suffices.
        return (
            self._typed_dict is other_spec.typed_dict
            and self._specs == other_spec.specs
        )

    @override
    def __hash__(self) -> int:
        return hash((self._typed_dict, tuple(self._specs.items())))

    @override
    def __repr__(self) -> str:
        args = ", ".join(f"{k}={s!r}" for k, s in self._specs.items())
        return f"DictSpec({self._typed_dict.__qualname__}, {args})"


# Any: the TypedDict of a DictSpec is not known where a spec is handled generically.
type Spec = TensorSpec | DictSpec[Any]


def fields(spec: Spec) -> list[tuple[str, tuple[int, ...], tuple[int, int, int]]]:
    """Return the fields of ``spec`` as ``_native.Schema`` takes them.

    A tensor message is one field named ``""``; a dict message has one field per
    key, in key order.
    """
    if isinstance(spec, TensorSpec):
        return [("", spec.shape, DLPACK_DTYPES[spec.dtype])]
    return [(k, s.shape, DLPACK_DTYPES[s.dtype]) for k, s in spec.specs.items()]


def _check_tensor(spec: TensorSpec, value: object, where: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{where}: expected a torch.Tensor, got {type(value).__name__}")
    if tuple(value.shape) != spec.shape:
        raise ValueError(
            f"{where}: expected shape {spec.shape}, got {tuple(value.shape)}"
        )
    if value.layout != torch.strided:
        raise TypeError(
            f"{where}: expected a dense (strided) tensor, got {value.layout}"
        )
    if value.dtype != spec.dtype:
        raise TypeError(f"{where}: expected dtype {spec.dtype}, got {value.dtype}")
    return value


def split(spec: Spec, msg: object, *, where: str) -> list[torch.Tensor]:
    """Check ``msg`` against ``spec`` and return its tensors in field order.

    ``where`` names the place of the message (e.g. a port path) in error messages.
    The tensors are returned as they are: no copy, ``detach`` or device move.

    Raises:
        TypeError: ``msg`` is not of the message type, a tensor is not dense
            (strided), or a dtype differs.
        ValueError: The keys of a dict message differ, or a shape differs.
    """
    if isinstance(spec, TensorSpec):
        return [_check_tensor(spec, msg, where)]
    if not isinstance(msg, Mapping):
        raise TypeError(
            f"{where}: expected a {spec.typed_dict.__qualname__} dict, "
            f"got {type(msg).__name__}"
        )
    values = cast(Mapping[object, object], msg)
    missing = [k for k in spec.specs if k not in values]
    unknown = [k for k in values if k not in spec.specs]
    if missing or unknown:
        raise ValueError(
            f"{where}: keys do not match {spec.typed_dict.__qualname__}: "
            f"missing {missing}, unknown {unknown}"
        )
    return [
        _check_tensor(s, values[k], f"{where}[{k!r}]") for k, s in spec.specs.items()
    ]


def host_tensors(
    spec: Spec, msg: object, *, where: str, copy: bool = False
) -> list[torch.Tensor]:
    """Check ``msg`` against ``spec`` (as ``split``), cut the gradients and return
    its tensors in host memory, for the core to copy.

    ``copy`` makes new tensors even of those already in host memory, for a value
    kept beyond the call. A CUDA tensor is copied on the calling thread's current
    stream (a node's own, in its hooks), and only that stream is synchronized.

    Raises:
        TypeError, ValueError: As ``split``.
    """
    return [t.detach().to("cpu", copy=copy) for t in split(spec, msg, where=where)]


def join(
    spec: Spec, tensors: Sequence[torch.Tensor]
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Build a message of ``spec`` from its tensors in field order (``split``'s
    inverse)."""
    if isinstance(spec, TensorSpec):
        (tensor,) = tensors
        return tensor
    return dict(zip(spec.specs, tensors, strict=True))
