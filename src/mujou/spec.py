"""Specs of the messages that flow between nodes.

A message is one field, or a ``TypedDict`` of fields (``DictSpec``). A field is a
``torch.Tensor`` (``TensorSpec``) or a Python scalar: an ``int`` (``IntSpec``), a
``float`` (``FloatSpec``), a ``bool`` (``BoolSpec``) or a ``Timestamp``
(``TimestampSpec``). A spec fixes the size of every field, so the slots that hold
the message can be laid out when the network is built (``specs/messages.md``).

``DLPACK``, the DLPack code of each supported dtype, is here rather than in the
codec: ``TensorSpec.dtypes`` is its keys, and the codec reads the same one table.
"""

import typing
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    TypeVar,
    cast,
    final,
    override,
)

import torch

from mujou._native import Timestamp
from mujou.types import Device

# The supported dtypes and their DLPack ``DLDataType`` (code, bits, lanes): the
# one table, which the codec reads to lay out the core's schemas.
DLPACK: Final[Mapping[torch.dtype, tuple[int, int, int]]] = MappingProxyType(
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


class MessageSpec(Generic[_M]):
    """The spec of a message of type ``_M``.

    ``TensorSpec`` is a ``MessageSpec[torch.Tensor]``, ``IntSpec`` a
    ``MessageSpec[int]`` and ``DictSpec(Sample, ...)`` a ``MessageSpec[Sample]``.
    Annotate with it a function that takes any spec::

        def sender[T](spec: MessageSpec[T]) -> Port[T]:
            return Port(send=spec)
    """


@final
class TensorSpec(MessageSpec[torch.Tensor]):
    """The shape and the dtype of a message that is one ``torch.Tensor``.

    Every dimension is at least 1, and ``()`` is a scalar. A batch dimension is
    just a part of the shape.

    Two specs are equal when their shapes and dtypes are equal. A message whose shape
    or dtype differs from the spec is an error; dtypes are never converted. A spec
    has no device: a message of the spec may be on any device.

    ``zeros``, ``rand`` and ``empty`` make a tensor of the spec, on the CPU unless
    ``device`` says otherwise. For any other initial value, fill one of them in
    place (``spec.empty().fill_(0.5)``).

    Example::

        spec = TensorSpec((32, 256))  # float32
        spec = TensorSpec((32, 256), torch.bfloat16)
        x0 = spec.zeros()
        noise = TensorSpec((4,)).rand(generator=torch.Generator().manual_seed(0))
        ones = spec.empty(device="cuda").fill_(1.0)

    Raises:
        TypeError: ``dtype`` is not in ``dtypes``.
    """

    dtypes: ClassVar[frozenset[torch.dtype]] = frozenset(DLPACK)
    """The dtypes a spec takes."""

    def __init__(
        self, shape: Sequence[int], dtype: torch.dtype = torch.float32
    ) -> None:
        if dtype not in TensorSpec.dtypes:
            supported = ", ".join(str(t) for t in DLPACK)
            raise TypeError(
                f"TensorSpec dtype {dtype} is not supported; use one of "
                f"TensorSpec.dtypes ({supported})"
            )
        self._shape = tuple(shape)
        self._dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the tensor."""
        return self._shape

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the tensor."""
        return self._dtype

    def zeros(self, *, device: Device = "cpu") -> torch.Tensor:
        """Return a tensor of this spec filled with zeros on ``device``."""
        return torch.zeros(self._shape, dtype=self._dtype, device=device)

    def empty(self, *, device: Device = "cpu") -> torch.Tensor:
        """Return a tensor of this spec on ``device`` whose values are not set.

        Set them before use, in place: ``spec.empty().fill_(0.5)``.
        """
        return torch.empty(self._shape, dtype=self._dtype, device=device)

    def rand(
        self,
        *,
        generator: torch.Generator | None = None,
        device: Device = "cpu",
    ) -> torch.Tensor:
        """Return a tensor of this spec drawn uniformly from [0, 1) on ``device``.

        Pass a ``generator`` on ``device`` to make the result deterministic;
        without one, the global torch generator is used.
        """
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
class IntSpec(MessageSpec[int]):
    """A message that is one Python ``int`` of ``bits`` bits, ``signed`` or not.

    A value out of the range of ``bits`` and ``signed`` is an error when it is
    sent or written (``OverflowError``); it is never wrapped. A ``bool`` is an
    ``int`` and is taken as 0 or 1. The initial value is 0.

    Two specs are equal when their ``bits`` and ``signed`` are equal.

    Example::

        count = IntSpec()  # int64
        level = IntSpec(8, signed=False)  # 0 to 255

    Raises:
        ValueError: ``bits`` is not 8, 16, 32 or 64.
    """

    def __init__(self, bits: Literal[8, 16, 32, 64] = 64, signed: bool = True) -> None:
        if bits not in (8, 16, 32, 64):
            raise ValueError(f"IntSpec bits must be 8, 16, 32 or 64, got {bits!r}")
        self._bits: Literal[8, 16, 32, 64] = bits
        self._signed = signed

    @property
    def bits(self) -> Literal[8, 16, 32, 64]:
        """The number of bits of the integer."""
        return self._bits

    @property
    def signed(self) -> bool:
        """Whether the integer is signed."""
        return self._signed

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IntSpec):
            return NotImplemented
        return self._bits == other._bits and self._signed == other._signed

    @override
    def __hash__(self) -> int:
        return hash((IntSpec, self._bits, self._signed))

    @override
    def __repr__(self) -> str:
        return f"IntSpec({self._bits}, signed={self._signed})"


class _Unit:
    """A spec without parameters: every instance of a class is equal."""

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Unit):
            return NotImplemented
        return type(self) is type(other)

    @override
    def __hash__(self) -> int:
        return hash(type(self))

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


@final
class FloatSpec(_Unit, MessageSpec[float]):
    """A message that is one Python ``float`` (64 bits).

    An ``int`` is taken too, as a ``float`` is in Python's typing, and is read
    back as a ``float``. The initial value is 0.0. Every ``FloatSpec`` is equal.
    """


@final
class BoolSpec(_Unit, MessageSpec[bool]):
    """A message that is one Python ``bool``. The initial value is False. Every
    ``BoolSpec`` is equal."""


@final
class TimestampSpec(_Unit, MessageSpec[Timestamp]):
    """A message that is one ``mujou.time.Timestamp``, such as a time that
    ``Run.write`` returned, carried on with what was computed from it.

    The initial value is ``Timestamp(ns=0)``. Every ``TimestampSpec`` is equal.
    """


# The spec of a field: a value of a ``DictSpec``, or a whole message.
type FieldSpec = TensorSpec | IntSpec | FloatSpec | BoolSpec | TimestampSpec


@final
class DictSpec[TD: Mapping[str, object]](MessageSpec[TD]):
    """The spec of a message that is a ``TypedDict`` of fields.

    Give the ``TypedDict`` class and one ``FieldSpec`` per key: a ``TensorSpec``
    for a key annotated as ``torch.Tensor``, ``IntSpec`` for ``int``,
    ``FloatSpec`` for ``float``, ``BoolSpec`` for ``bool`` and
    ``TimestampSpec`` for ``Timestamp``. Every key of the ``TypedDict`` is
    required (no nesting), and the keyword arguments name exactly its keys. The
    keys keep the order in which the ``TypedDict`` defines them.

    Two specs are equal when they use the same ``TypedDict`` class and equal
    specs for every key.

    Example::

        class Feedback(TypedDict):
            error: torch.Tensor
            count: int


        spec = DictSpec(Feedback, error=TensorSpec((32, 256)), count=IntSpec())
        msg: Feedback = spec.zeros()
        error = spec.field("error", TensorSpec)  # a TensorSpec, to type checkers too

    Raises:
        TypeError: The keyword arguments miss a key of the ``TypedDict`` or name
            a key it does not have.
    """

    def __init__(self, typed_dict: type[TD], /, **specs: FieldSpec) -> None:
        keys = typing.get_type_hints(typed_dict)
        missing = [k for k in keys if k not in specs]
        unknown = [k for k in specs if k not in keys]
        if missing or unknown:
            raise TypeError(
                f"DictSpec({typed_dict.__qualname__}) keys do not match the "
                f"TypedDict: missing {missing}, unknown {unknown}"
            )
        self._typed_dict = typed_dict
        self._specs: Mapping[str, FieldSpec] = MappingProxyType(
            {k: specs[k] for k in keys}
        )

    @property
    def specs(self) -> Mapping[str, FieldSpec]:
        """The spec of each key, in the order the ``TypedDict`` defines.

        To use the spec of one key as its kind, take it with ``field``.
        """
        return self._specs

    def field[F: FieldSpec](self, key: str, kind: type[F]) -> F:
        """Return the spec of ``key``, checked to be a ``kind``.

        Example::

            h = act_spec.field("h", TensorSpec)
            h.shape

        Raises:
            KeyError: No such key.
            TypeError: The spec of ``key`` is not a ``kind``.
        """
        spec = self._specs[key]
        if not isinstance(spec, kind):
            raise TypeError(
                f"{self!r}: the spec of {key!r} is {type(spec).__name__}, "
                f"not {kind.__name__}"
            )
        return spec

    def zeros(self, *, device: Device = "cpu") -> TD:
        """Return a message whose tensors are zeros on ``device`` and whose
        scalars are zero (``0``, ``0.0``, ``False``, ``Timestamp(ns=0)``)."""
        return cast(
            TD,
            {
                k: s.zeros(device=device) if isinstance(s, TensorSpec) else _zero(s)
                for k, s in self._specs.items()
            },
        )

    def empty(self, *, device: Device = "cpu") -> TD:
        """Return a message whose tensors are on ``device`` and not set, and
        whose scalars are zero as in ``zeros``.

        Set the tensors before use, in place: ``msg["error"].fill_(0.5)``.
        """
        return cast(
            TD,
            {
                k: s.empty(device=device) if isinstance(s, TensorSpec) else _zero(s)
                for k, s in self._specs.items()
            },
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictSpec):
            return NotImplemented
        # Any: the other spec may be of any TypedDict; identity decides.
        other_spec = cast(DictSpec[Any], other)
        # The same TypedDict fixes the key order, so comparing as mappings suffices.
        return (
            self._typed_dict is other_spec._typed_dict
            and self._specs == other_spec.specs
        )

    @override
    def __hash__(self) -> int:
        return hash((self._typed_dict, tuple(self._specs.items())))

    @override
    def __repr__(self) -> str:
        args = ", ".join(f"{k}={s!r}" for k, s in self._specs.items())
        return f"DictSpec({self._typed_dict.__qualname__}, {args})"


def _zero(spec: IntSpec | FloatSpec | BoolSpec | TimestampSpec) -> object:
    """The zero of a scalar field: the value a slot holds before its first write."""
    match spec:
        case IntSpec():
            return 0
        case FloatSpec():
            return 0.0
        case BoolSpec():
            return False
        case TimestampSpec():
            return Timestamp(ns=0)
