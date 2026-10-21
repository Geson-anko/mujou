"""Specs of the messages that flow between nodes.

A message is either one ``torch.Tensor`` (``TensorSpec``) or a ``TypedDict`` whose
values are ``torch.Tensor`` (``DictSpec``). A spec fixes the shape and the dtype of
every value, so the slots that hold the message can be laid out when the network
is built (``specs/messages.md``).

``DLPACK``, the DLPack code of each supported dtype, is here rather than in the
codec: ``TensorSpec.dtypes`` is its keys, and the codec reads the same one table.
"""

import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar, Final, Generic, TypeVar, cast, final, override

import torch

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


class MessageSpec(ABC, Generic[_M]):
    """The spec of a message of type ``_M``: its shape, dtype and initial value.

    ``TensorSpec`` is a ``MessageSpec[torch.Tensor]`` and ``DictSpec(Sample,
    ...)`` a ``MessageSpec[Sample]``. Annotate with it a function that takes
    either::

        def sender[T](spec: MessageSpec[T]) -> Port[T]:
            return Port(send=spec)
    """

    @abstractmethod
    def zeros(self, *, device: Device = "cpu") -> _M:
        """Return a message filled with zeros on ``device``.

        On the CPU, this is the default initial value that a port receives before
        the first message is written.
        """

    @abstractmethod
    def empty(self, *, device: Device = "cpu") -> _M:
        """Return a message on ``device`` whose values are not set.

        Set them before use, in place (``fill_`` and the like).
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

    @override
    def zeros(self, *, device: Device = "cpu") -> torch.Tensor:
        """Return a tensor of this spec filled with zeros on ``device``."""
        return torch.zeros(self._shape, dtype=self._dtype, device=device)

    @override
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
class DictSpec[TD: Mapping[str, object]](MessageSpec[TD]):
    """The spec of a message that is a ``TypedDict`` of ``torch.Tensor`` values.

    Give the ``TypedDict`` class and one ``TensorSpec`` per key. Every key of the
    ``TypedDict`` is required and annotated as ``torch.Tensor`` (no nesting), and
    the keyword arguments name exactly its keys. The keys keep the order in which
    the ``TypedDict`` defines them.

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
        TypeError: The keyword arguments miss a key of the ``TypedDict`` or name
            a key it does not have.
    """

    def __init__(self, typed_dict: type[TD], /, **specs: TensorSpec) -> None:
        keys = typing.get_type_hints(typed_dict)
        missing = [k for k in keys if k not in specs]
        unknown = [k for k in specs if k not in keys]
        if missing or unknown:
            raise TypeError(
                f"DictSpec({typed_dict.__qualname__}) keys do not match the "
                f"TypedDict: missing {missing}, unknown {unknown}"
            )
        self._typed_dict = typed_dict
        self._specs: Mapping[str, TensorSpec] = MappingProxyType(
            {k: specs[k] for k in keys}
        )

    @property
    def specs(self) -> Mapping[str, TensorSpec]:
        """The ``TensorSpec`` of each key, in the order the ``TypedDict`` defines."""
        return self._specs

    @override
    def zeros(self, *, device: Device = "cpu") -> TD:
        """Return a message whose values are all zeros on ``device``."""
        return cast(TD, {k: s.zeros(device=device) for k, s in self._specs.items()})

    @override
    def empty(self, *, device: Device = "cpu") -> TD:
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
