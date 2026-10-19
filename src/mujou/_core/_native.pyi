"""
The native core of mujou: timers, slots and the runtime that runs the nodes.

Private: the framework wraps it. Values go in and out as tensors over DLPack,
always copied.
"""

from collections.abc import Callable, Sequence
from typing import (
    Any,
    Final,
    NoReturn,
    Self,
    SupportsFloat,
    SupportsIndex,
    final,
    override,
)

from _typeshed import SupportsGetItem
from typing_extensions import disjoint_base

__all__ = [
    "__version__",
    "timer",
    "BufferView",
    "GlobalTimer",
    "InputGroup",
    "InputSlot",
    "Message",
    "NetworkError",
    "Node",
    "Outlet",
    "PauseTimeoutError",
    "Runtime",
    "Schema",
    "Slot",
    "Timer",
    "TimerReader",
    "Timestamp",
]

__version__: Final[str]

timer: Final[GlobalTimer]
"""
The global timer.
"""

@final
class BufferView:
    """
    One field of a message, lent over DLPack.

    Read it with `torch.from_dlpack(view)` and copy the result at once: the
    memory belongs to the core and must not be written. `__dlpack__` raises
    `BufferError` for `copy=True` or a `dl_device` other than the CPU.
    """
    def __dlpack__(
        self,
        /,
        *,
        stream: Any | None = None,
        max_version: tuple[SupportsIndex, SupportsIndex] | None = None,
        dl_device: tuple[SupportsIndex, SupportsIndex] | None = None,
        copy: bool | None = None,
    ) -> Any: ...
    def __dlpack_device__(self, /) -> tuple[int, int]: ...
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@final
class GlobalTimer(Timer):
    """
    The global timer, which every run not given a timer of its own uses. The
    only instance is `timer`.
    """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@final
class InputGroup:
    """
    The slots the host writes, switched together by one version counter.
    """
    def __new__(cls, /, *, timer: Timer) -> Self:
        """
        Every write is stamped with the time on `timer`.

        The slots and the host's inputs of one run must use the timer of its
        runtime: on another timer, the times of the messages would not be on
        the axis the nodes' Δt is measured on, and a pause of the run would not
        stop them, so a state taken at T could hold later messages.
        """
    def input(self, /, schema: Schema, name: str) -> InputSlot:
        """
        Creates the slot `"{name}/in"` holding zeros.
        """
    def write(
        self, /, items: SupportsGetItem[int, tuple[InputSlot, Sequence[Any]]]
    ) -> Timestamp:
        """
        Writes every item at once, all stamped with the same time, and returns
        that time.

        No reader sees some of the slots new and others old at any one moment;
        a reader that reads two slots one after the other may still see a write
        complete in between. If any value is invalid, no slot changes. Raises as
        `Outlet.send` does, and `ValueError` for a slot of another group or a
        slot given twice.
        """

@final
class InputSlot:
    """
    A slot the host writes through its `InputGroup`.
    """
    @property
    def name(self, /) -> str: ...
    def read(self, /) -> Message: ...
    def restore(
        self, /, values: Sequence[Any], time_ns: SupportsIndex, sent: bool
    ) -> None:
        """
        Sets the value, its time and whether it was sent, dropping any pending
        write.

        Call it only while no node runs (before the start, or while every node
        is paused). Raises as `Outlet.send` does.
        """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@final
class Message:
    """
    An immutable value of a slot, the time it was written at on the writer's
    timer, and whether the writer sent it.
    """
    @property
    def buffers(self, /) -> tuple[BufferView, ...]:
        """
        One view per field, in key order.
        """
    @property
    def sent(self, /) -> bool:
        """
        Whether the slot's writer sent it (or a restore said so), rather than it
        being the value the slot was created or initialized with.
        """
    @property
    def timestamp(self, /) -> Timestamp:
        """
        The time it was written at.
        """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

class NetworkError(RuntimeError):
    """
    The network stopped because a node raised. `__cause__` is the first exception.
    """

class Node:
    """
    The base of every node.

    The runtime calls the hooks on the thread that runs the node: a worker of
    the run's pool, which may differ from one hook to the next, or the node's
    own thread if it was started as dedicated (`state_dict`, through
    `_capture_state`, on the control thread while the node is paused). The hooks
    of a node never overlap, and each runs in the node's own
    `contextvars.Context` (but `_capture_state` and `_before_hook`). They do
    nothing by default.
    The hooks are looked up by name until `on_start` (and `load_state_dict`)
    has returned, and once after that, so replacing one in those hooks takes
    effect. `state_dict` itself is looked up by the default `_capture_state` at
    each capture, so replacing it later takes effect too.
    """
    def __new__(cls, /, *_args: Any, **_kwargs: Any) -> Self:
        """
        Accepts and ignores any arguments, so that subclasses define `__init__`
        freely.
        """
    def _after_update(self, /) -> None:
        """
        Called right after each `on_update`, for the framework's per-step work
        (so that overriding `on_update` cannot skip it).
        """
    def _before_hook(self, /) -> None:
        """
        Called right before every public hook but `state_dict` (after
        `_on_enter` when the thread switches to this node), for the framework's
        settings that must hold at the start of each hook whatever the previous
        hook left. It runs outside the node's `contextvars.Context`, so a
        setting held in a context variable does not take effect here. An error
        here is reported as the error of that hook, which is then not called,
        except `on_exception` and `on_stop`, which still run.
        """
    def _capture_state(self, /) -> "object":
        """
        Called instead of `state_dict` when the runtime takes the state (on the
        control thread, while the node is paused), for the framework's work
        around it. Returns `self.state_dict()` by default.
        """
    def _on_enter(self, /) -> None:
        """
        Called right before a hook whenever a thread switches to this node (the
        first time a thread runs it, and on a worker whenever it ran another
        node in between), for the framework's per-thread settings, so that no
        override of a public hook can skip them. An error here is reported as
        the error of the hook that was to follow.
        """
    def _restore_state(self, /, state: "object") -> None:
        """
        Called instead of `load_state_dict` when the runtime restores the state
        (after `on_start`), for the framework's work around it. Calls
        `self.load_state_dict(state)` by default.
        """
    def load_state_dict(self, /, state: "object") -> None:
        """
        Restores `state`, a copy of what `state_dict` returned, after `on_start`
        when the run is restored. Does nothing by default.
        """
    def on_exception(self, /) -> None:
        """
        Called once on every node when a hook of any node has raised, before
        `on_stop`. What it raises is logged and ignored.
        """
    def on_pause(self, /) -> None:
        """
        Called when the node has stopped for a pause of the run (to take the
        state), at the end of its current `on_update`.
        """
    def on_resume(self, /) -> None:
        """
        Called when the pause ends, before the next `on_update`.
        """
    def on_start(self, /) -> None:
        """
        Called once when the run starts (also when it is restored), before any
        other public hook. Allocate and initialize the state here.
        """
    def on_stop(self, /) -> None:
        """
        Called once when the run stops, also after an exception. It must not
        change what `state_dict` returns. What it raises is logged and ignored.
        """
    def on_update(self, /, dt: "float") -> None:
        """
        Called repeatedly with `dt`, the seconds on the run's timer since the
        node's last update (always > 0). Read the ports, update in proportion to
        `dt` and send.
        """
    def state_dict(self, /) -> "object":
        """
        Returns the node's own state as a picklable object, when the run's state
        is taken (the node is paused or the run has ended). `None` by default.
        """

@final
class Outlet:
    """
    The sending side of one port and the only writer of its slots.

    The caller keeps it the only writer: send from one thread at a time (the
    owning node's). Concurrent sends stay memory safe but may leave the slots
    out of time order, or the outward slot and the channels holding different
    messages.

    Values are copied in without the GIL, so changing them afterwards does not
    change what readers see. Resizing or freeing a value from another thread
    while it is being sent is a data race in the caller's code.

    Raises (`send`, `send_to`):
        ValueError: The number of values, a shape or a dtype differs from the
            schema. The message names the slot and the field.
        TypeError: A value does not support DLPack or is not in host memory.
            Pinned memory is read without synchronizing any CUDA stream, so
            synchronize the copy that filled it before sending.
        IndexError: No such channel.
    """
    def __new__(
        cls, /, schema: Schema, channels: SupportsIndex, name: str, *, timer: Timer
    ) -> Self:
        """
        Creates the slots `"{name}/out"` and `"{name}/ch{k}"` holding zeros.
        Every message is stamped with the time on `timer`.

        The slots and the host's inputs of one run must use the timer of its
        runtime: on another timer, the times of the messages would not be on
        the axis the nodes' Δt is measured on, and a pause of the run would not
        stop them, so a state taken at T could hold later messages.
        """
    def channel(self, /, index: SupportsIndex) -> Slot: ...
    @property
    def outbox(self, /) -> Slot:
        """
        The outward slot, which every `send` writes.
        """
    def send(self, /, values: Sequence[Any]) -> None:
        """
        Writes one message to the outward slot and every channel.
        """
    def send_to(self, /, index: SupportsIndex, values: Sequence[Any]) -> None:
        """
        Writes one message to channel `index` only.
        """

class PauseTimeoutError(TimeoutError):
    """
    Some nodes did not pause in time, so the state was not taken. The network keeps running.
    """

@final
class Runtime:
    """
    Runs nodes on a pool of worker threads, or each dedicated node on a thread
    of its own, under one control thread.
    """
    def __new__(
        cls,
        /,
        nodes: SupportsGetItem[int, Node],
        names: SupportsGetItem[int, str],
        *,
        timer: Timer,
    ) -> Self:
        """
        The nodes run on `timer`: it measures their Δt, and the runtime freezes
        it to start and to take the state. Create the `Outlet`s and the
        `InputGroup` of this run with the same timer.
        """
    def capture(
        self,
        /,
        build: Callable[[list[object]], Any],
        timeout: SupportsFloat | SupportsIndex,
    ) -> Any:
        """
        Pauses every node, calls `build` with their `state_dict()` in node
        order while paused, resumes, and returns what `build` returned.

        `build` runs on the control thread and must not call this runtime's
        waiting methods. It may capture another runtime, but two builds must
        not wait for each other's runtime: that cycle deadlocks, and the control
        threads cannot see Ctrl-C. If `build` raises, the exception propagates
        and the network keeps running. Gives up with `PauseTimeoutError` after
        `timeout` real seconds if some node has not paused.
        """
    def check(self, /) -> None:
        """
        Raises `NetworkError` if a node has raised.
        """
    @property
    def is_finished(self, /) -> bool: ...
    def join(self, /, timeout: SupportsFloat | SupportsIndex | None = None) -> bool:
        """
        Waits until the network has stopped, for at most `timeout` seconds, and
        returns whether it has.
        """
    def last_update_ns(self, /) -> list[int]:
        """
        The time of each node's last update on the runtime's timer, in
        nanoseconds.
        """
    def request_stop(self, /) -> None:
        """
        Asks every node to stop and returns without waiting; `join` waits for
        the end. Does nothing before `start` or once the network has stopped.
        """
    def shutdown(self, /) -> None:
        """
        Stops every node and waits until the network has stopped. Does nothing
        before `start` or after a normal stop.
        """
    def start(
        self,
        /,
        *,
        manual: bool = False,
        states: SupportsGetItem[int, Any] | None = None,
        last_update_ns: SupportsGetItem[int, SupportsIndex] | None = None,
        periods: SupportsGetItem[int, SupportsFloat | SupportsIndex] | None = None,
        timer_state: "dict[str, int | float] | None" = None,
        workers: SupportsIndex | None = None,
        dedicated: SupportsGetItem[int, bool] | None = None,
    ) -> None:
        """
        Starts every node and returns once all of them have started (and
        restored `states`) and the timer runs again. If a node raised meanwhile,
        raises `NetworkError` after the network has stopped. If an argument is
        invalid, raises `ValueError` and nothing starts, so `start` can be
        called again.

        The timer is frozen first and loaded from `timer_state`
        (`Timer.state_dict()`), so the time does not run while the nodes start.
        Loading it raises `ValueError` while another runtime runs on the same
        timer, whose time would jump. `last_update_ns` is where each node's
        first Δt starts (the time at the start by default). `periods` is each
        node's period on the timer in seconds (>= 0), rounded to the nearest ns
        and at least 1 ns, so that 0.0 updates as soon as any time has passed,
        as every node does without `periods`; the manual mode ignores it.

        `dedicated` tells which nodes run on a thread of their own; the others,
        all of them by default, share a pool of `workers` threads (>= 1), by
        default as many as the available parallelism, never more than the nodes
        in the pool.

        If the wait is interrupted (Ctrl-C), the start goes on in the
        background; `shutdown` stops it.
        """
    def step(self, /, order: SupportsGetItem[int, SupportsIndex] | None = None) -> None:
        """
        Manual mode: lets the nodes in `order` (all, in index order, if `None`)
        update once each, one after another.
        """

@final
class Schema:
    """
    The fields of a message, fixed when the network is built.

    `Schema(fields)` takes `(name, shape, (code, bits, lanes))` per field, the
    dtype as a DLPack `DLDataType`. A tensor message has one field named `""`;
    a TypedDict message has one field per key, in key order. Every extent must
    be at least 1.

    Raises:
        ValueError: No fields, duplicate names, an extent below 1, a dtype that
            is not a scalar of whole bytes, or a field too large to allocate.
    """
    def __new__(
        cls,
        /,
        fields: SupportsGetItem[
            int,
            tuple[
                str,
                SupportsGetItem[int, SupportsIndex],
                tuple[SupportsIndex, SupportsIndex, SupportsIndex],
            ],
        ],
    ) -> Self: ...

@final
class Slot:
    """
    The latest value of one writer, read by any number of readers.
    """
    @property
    def name(self, /) -> str: ...
    def read(self, /) -> Message: ...
    def restore(
        self, /, values: Sequence[Any], time_ns: SupportsIndex, sent: bool
    ) -> None:
        """
        Sets the value, its time and whether it was sent as they were when the
        state was taken (`sent` is false for an initial value).

        Call it only while no node runs (before the start, or while every node
        is paused): it writes beside the slot's own writer. Raises as
        `Outlet.send` does.
        """
    def wait_newer(
        self, /, last: Message, timeout: SupportsFloat | SupportsIndex
    ) -> Message | None:
        """
        Waits without the GIL until the slot holds a message other than `last`.

        Returns `None` after `timeout` seconds of real time.

        Only for the host's `subscribe`: a node must never wait for another, so
        ports do not expose it.

        Raises:
            ValueError: `timeout` is negative or not finite.
        """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@disjoint_base
class Timer:
    """
    A time axis that nodes discretize their dynamics on.

    `Timer()` is a new timer at time 0 on the real clock at scale 1. Systems
    whose pauses should not stop each other's time run on separate timers;
    a run not given a timer runs on the global one, `timer`.

    The time is held in nanoseconds and saturates at about 584 years.
    """
    def __new__(cls, /) -> Self: ...
    def advance(self, /, seconds: SupportsFloat | SupportsIndex) -> None:
        """
        Moves the virtual clock forward by `seconds`. The scale does not apply.

        Raises:
            ValueError: `seconds` is negative or not finite.
            RuntimeError: The real clock is in use, or the time is paused.
        """
    def is_paused(self, /) -> bool:
        """
        Whether the time is frozen by `pause` or by a runtime. The virtual clock
        alone does not count as paused.
        """
    def load_state_dict(self, /, state: "dict[str, int | float]") -> None:
        """
        Continues the time from a `state_dict`.

        Raises:
            ValueError: The scale is not finite and > 0.
        """
    def now(self, /) -> Timestamp:
        """
        The time as a `Timestamp`.
        """
    def pause(self, /) -> None:
        """
        Freezes the time until `resume`. For hosts and tests: the runtime
        freezes the time with its own holds, which this switch neither ends nor
        is ended by. Pausing twice needs one resume.
        """
    def reader(self, /) -> TimerReader:
        """
        A `TimerReader` of this timer: it reads the time and the scale and has
        no way to change them.
        """
    def resume(self, /) -> None:
        """
        Turns off the switch set by `pause`. The time runs again only when no
        runtime holds it frozen either.
        """
    def set_time_scale(self, /, scale: SupportsFloat | SupportsIndex) -> None:
        """
        Sets how many seconds of this timer pass per real second.

        Raises:
            ValueError: `scale` is not finite and > 0. Use `pause` to stop the
                time.
        """
    def sleep(self, /, seconds: SupportsFloat | SupportsIndex) -> None:
        """
        Blocks for `seconds` of this timer's time, following the scale and
        pauses: while paused, it does not return until resumed. Waits without
        the GIL and checks for signals (Ctrl-C) every 50 ms.

        Raises:
            ValueError: `seconds` is negative or not finite.
        """
    def state_dict(self, /) -> "dict[str, int | float]":
        """
        `{"time_ns": int, "scale": float}`: what is needed to continue the time
        from where it was taken. Take it while paused.
        """
    def time(self, /) -> float:
        """
        The time in seconds.
        """
    def time_ns(self, /) -> int:
        """
        The time in nanoseconds.
        """
    def time_scale(self, /) -> float:
        """
        Seconds of this timer per real second.
        """
    def use_virtual_clock(self, /, enabled: bool) -> None:
        """
        Switches to the virtual clock, which moves only by `advance`, or back to
        the real clock. The time stays continuous. For tests.
        """

@final
class TimerReader:
    """
    The reading side of a `Timer`: the time and the scale, and nothing that
    changes them. Made by `Timer.reader()`.
    """
    def now(self, /) -> Timestamp:
        """
        The time as a `Timestamp`.
        """
    def time_scale(self, /) -> float:
        """
        Seconds of the timer per real second.
        """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@final
class Timestamp:
    """
    A time on a timer.

    `ts.float()` (or `float(ts)`) is the time in seconds and `int(ts)` its whole
    seconds; `ts.ns` is the exact time in nanoseconds. Timestamps compare and
    hash by time, so compare only those of one timer. `a - b` is the time from
    `b` to `a` in seconds, a `float` that is negative when `b` is later.
    Timestamps pickle and copy.
    """
    @override
    def __eq__(self, value: object, /) -> bool: ...
    def __float__(self, /) -> float: ...
    def __ge__(self, value: Timestamp, /) -> bool: ...
    def __gt__(self, value: Timestamp, /) -> bool: ...
    @override
    def __hash__(self, /) -> int: ...
    def __int__(self, /) -> int: ...
    def __le__(self, value: Timestamp, /) -> bool: ...
    def __lt__(self, value: Timestamp, /) -> bool: ...
    @override
    def __ne__(self, value: object, /) -> bool: ...
    def __new__(cls, /, *, ns: SupportsIndex) -> Self:
        """
        Takes the time in nanoseconds, by keyword only: a bare number could
        mean seconds.

        Raises:
            ValueError: `ns` is negative or does not fit in 64 bits.
        """
    @override
    def __reduce__(self, /) -> Any:
        """
        Pickles and copies as `Timestamp(ns=...)`.
        """
    @override
    def __repr__(self, /) -> str: ...
    def __sub__(self, other: Timestamp, /) -> float:
        """
        The time from `other` to this one, in seconds: `(self.ns - other.ns) /
        1e9`, negative when `other` is later.
        """
    def float(self, /) -> float:
        """
        The time in seconds.
        """
    @property
    def ns(self, /) -> int:
        """
        The time in nanoseconds.
        """
