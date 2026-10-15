"""Type stubs for the native extension module built from crates/mujou-python."""

from typing import Final, final

__version__: str

@final
class GlobalTimer:
    """Handle to the process-wide global timer.

    The only instance is ``timer``.
    """

    def time(self) -> float: ...
    def time_ns(self) -> int: ...
    def sleep(self, seconds: float) -> None: ...
    def time_scale(self) -> float: ...
    def set_time_scale(self, scale: float) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def is_paused(self) -> bool: ...
    def state_dict(self) -> dict[str, int | float]: ...
    def load_state_dict(self, state: dict[str, int | float]) -> None: ...
    def use_virtual_clock(self, enabled: bool) -> None: ...
    def advance(self, seconds: float) -> None: ...

timer: Final[GlobalTimer]

# --- slot (WP-A)

# Imported here rather than at the top so that the sections stay independent.
from collections.abc import Sequence  # noqa: E402
from types import CapsuleType  # noqa: E402
from typing import Protocol  # noqa: E402

type DLDataType = tuple[int, int, int]
"""``(code, bits, lanes)`` of a DLPack dtype."""

class SupportsDLPack(Protocol):
    """Any tensor that can be exported over DLPack, such as ``torch.Tensor``."""

    def __dlpack__(self) -> object: ...

@final
class Schema:
    """The fields of a message, fixed when the network is built.

    A tensor message has one field named ``""``; a TypedDict message has one
    field per key, in key order. Every extent must be at least 1.

    Raises:
        ValueError: No fields, duplicate names, an extent below 1, or a dtype
            that is not a scalar of whole bytes.
    """

    def __init__(
        self, fields: Sequence[tuple[str, Sequence[int], DLDataType]]
    ) -> None: ...

@final
class BufferView:
    """One field of a message, lent over DLPack.

    Read it with ``torch.from_dlpack(view)`` and copy the result at once: the
    memory belongs to the core and must not be written.
    """

    def __dlpack__(
        self,
        *,
        stream: object = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> CapsuleType: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

@final
class Message:
    """An immutable value of a slot."""

    @property
    def buffers(self) -> tuple[BufferView, ...]:
        """One view per field, in key order."""

    @property
    def time_ns(self) -> int:
        """The global time it was written at."""

@final
class Slot:
    """The latest value of one writer, read by any number of readers."""

    @property
    def name(self) -> str: ...
    def read(self) -> Message: ...
    def wait_newer(self, last: Message | None, timeout: float) -> Message | None:
        """Wait without the GIL until the value is not ``last``.

        Returns ``None`` after ``timeout`` seconds of real time. With ``last``
        ``None``, returns the current value at once.

        Only for the host's ``subscribe``: a node must never wait for another,
        so ports do not expose it.
        """

    def restore(self, values: Sequence[SupportsDLPack], time_ns: int) -> None:
        """Set the value and its time as they were when the state was taken.

        Call it only while no node runs (before the start, or while every node
        is paused): it writes beside the slot's own writer.
        """

@final
class Outlet:
    """The sending side of one port and the only writer of its slots.

    The caller keeps it the only writer: send from one thread at a time (the
    owning node's). Concurrent sends stay memory safe but may leave the slots
    out of time order, or the outward slot and the channels holding different
    messages.

    Values are copied in without the GIL, so changing them afterwards does not
    change what readers see. Resizing or freeing a value from another thread
    while it is being sent is a data race in the caller's code.

    Raises (``send``, ``send_to``):
        ValueError: The number of values, a shape or a dtype differs from the
            schema. The message names the slot and the field.
        TypeError: A value does not support DLPack or is not in host memory.
            Pinned memory is read without synchronizing any CUDA stream, so
            synchronize the copy that filled it before sending.
        IndexError: No such channel.
    """

    def __init__(self, schema: Schema, channels: int, name: str) -> None:
        """Create the slots ``"{name}/out"`` and ``"{name}/ch{k}"`` holding zeros."""

    @property
    def outbox(self) -> Slot:
        """The outward slot, which every ``send`` writes."""

    def channel(self, index: int) -> Slot: ...
    def send(self, values: Sequence[SupportsDLPack]) -> None:
        """Write one message to the outward slot and every channel."""

    def send_to(self, index: int, values: Sequence[SupportsDLPack]) -> None:
        """Write one message to channel ``index`` only."""

@final
class InputGroup:
    """The slots the host writes, switched together by one version counter."""

    def __init__(self) -> None: ...
    def input(self, schema: Schema, name: str) -> InputSlot:
        """Create the slot ``"{name}/in"`` holding zeros."""

    def write(
        self, items: Sequence[tuple[InputSlot, Sequence[SupportsDLPack]]]
    ) -> None:
        """Write every item at once, all stamped with the same time.

        No reader sees some of the slots new and others old at any one
        moment; a reader that reads two slots one after the other may still
        see a write complete in between. If any value is invalid, no slot
        changes. Raises as ``Outlet.send`` does, and ``ValueError`` for a slot
        of another group or a slot given twice.
        """

@final
class InputSlot:
    """A slot the host writes through its ``InputGroup``."""

    @property
    def name(self) -> str: ...
    def read(self) -> Message: ...
    def restore(self, values: Sequence[SupportsDLPack], time_ns: int) -> None:
        """Set the value and its time, dropping any pending write.

        Call it only while no node runs (before the start, or while every node
        is paused).
        """

# --- runtime (WP-B)
