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
    "global_timer",
    "BufferView",
    "Carry",
    "InputGroup",
    "InputSlot",
    "Message",
    "Outlet",
    "ReadOnlyTimer",
    "Runtime",
    "Schema",
    "Slot",
    "Timer",
    "Timestamp",
    "VirtualTimer",
    "current",
    "start",
]

global_timer: Final[Timer]
"""
The global timer, which every run not given a timer of its own uses.
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
class Carry:
    """
    The data one message carries, to read or to pass on with
    `Outlet.send(values, relay=carry)` without copying it.
    """
    @property
    def buffers(self, /) -> tuple[BufferView, ...]:
        """
        One view per carry field, in key order (none if nothing is carried).
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
    def __new__(cls, /, *, clock: ReadOnlyTimer) -> Self:
        """
        Every write is stamped with the time on `clock`, as `Outlet` says.
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
        complete in between. Each slot's values are its value fields, then its
        carry fields. If any value is invalid, no slot changes. Raises as
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
    def new_carry(self, /, values: Sequence[Any]) -> Carry:
        """
        A new carry of `values` (one per carry field), from the pools of this
        slot, as `Slot.new_carry`.
        """
    def read(self, /) -> Message: ...
    def restore(
        self, /, values: Sequence[Any], time_ns: SupportsIndex, sent: bool
    ) -> None:
        """
        Sets the value, its time and whether it was sent, dropping any pending
        write.

        `values` are the value fields, then the carry fields. Call it only while
        no node runs (before the start, or while every node is paused). Raises
        as `Outlet.send` does.
        """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@final
class Message:
    """
    An immutable value of a slot, the data it carries, the time it was written
    at on the writer's timer, and whether the writer sent it.
    """
    @property
    def buffers(self, /) -> tuple[BufferView, ...]:
        """
        One view per value field, in key order.
        """
    @property
    def carry(self, /) -> Carry:
        """
        The data it carries, shared with the message and not copied.
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

    `values` are the value fields, then the carry fields. With `relay`, they
    are the value fields only, and the message carries `relay` (the carry of a
    message this outlet's node received), shared and not copied.

    Raises (`send`, `send_to`):
        ValueError: The number of values, a shape or a dtype differs from the
            schema (the message names the slot and the field), or `relay`
            differs from the carry fields of the schema or the schema carries
            nothing.
        TypeError: A value does not support DLPack or is not in host memory.
            Pinned memory is read without synchronizing any CUDA stream, so
            synchronize the copy that filled it before sending.
        IndexError: No such channel.
    """
    def __new__(
        cls,
        /,
        schema: Schema,
        channels: SupportsIndex,
        name: str,
        *,
        clock: ReadOnlyTimer,
    ) -> Self:
        """
        Creates the slots `"{name}/out"` and `"{name}/ch{k}"` holding zeros.
        Every message is stamped with the time on `clock`.

        The slots and the host's inputs of one run must use the clock its
        runtime runs on: on another clock, the times of the messages would not
        be on the axis the nodes' Δt is measured on, a pause of the run would
        not stop them, and its end would not bound them, so a state taken at T
        could hold later messages.
        """
    def channel(self, /, index: SupportsIndex) -> Slot: ...
    @property
    def outbox(self, /) -> Slot:
        """
        The outward slot, which every `send` writes.
        """
    def send(self, /, values: Sequence[Any], relay: Carry | None = None) -> None:
        """
        Writes one message to the outward slot and every channel.
        """
    def send_to(
        self, /, index: SupportsIndex, values: Sequence[Any], relay: Carry | None = None
    ) -> None:
        """
        Writes one message to channel `index` only.
        """

@final
class ReadOnlyTimer:
    """
    The reading side of a timer: its time and its scale, and nothing that
    changes them. What a node sees of its run's timer as `Node.timer`.

    `ReadOnlyTimer(timer)` reads `timer`. The one a run gives its nodes stops
    at the time the run ends: every time of the run is at most that time.
    """
    def __new__(cls, /, timer: Timer) -> Self: ...
    @override
    def __repr__(self, /) -> str: ...
    def now(self, /) -> Timestamp:
        """
        The time now.
        """
    def time_scale(self, /) -> float:
        """
        How many seconds of the timer pass per real second.
        """

@final
class Runtime:
    """
    The running nodes of one run, from `start`: on a pool of worker threads,
    or each dedicated node on a thread of its own, under one control thread.
    """
    def capture(
        self,
        /,
        build: Callable[[list[object]], Any],
        timeout: SupportsFloat | SupportsIndex,
    ) -> Any:
        """
        Pauses every node, calls `build` with the states their drivers'
        `capture()` return, in node order, while paused, resumes, and returns
        what `build` returned. After a normal end, calls their `capture()` and
        `build` on this thread instead, at once.

        `build` runs on the control thread and must not call this runtime's
        waiting methods. It may capture another runtime, but two builds must
        not wait for each other's runtime: that cycle deadlocks, and the control
        threads cannot see Ctrl-C. If `build` raises, the exception propagates
        and the network keeps running. Gives up with `TimeoutError` after
        `timeout` real seconds if some node has not paused; the network keeps
        running.
        """
    def check(self, /) -> None:
        """
        Raises `NetworkError` if a node has raised.
        """
    @property
    def finished(self, /) -> bool:
        """
        The network has stopped.
        """
    def join(self, /, timeout: SupportsFloat | SupportsIndex | None = None) -> bool:
        """
        Waits until the network has stopped, for at most `timeout` seconds (no
        limit if `None`), and returns whether it has. Raises `NetworkError` if
        it stopped because a node raised.
        """
    def last_update_ns(self, /) -> list[int]:
        """
        The time of each node's last update on the runtime's timer, in
        nanoseconds.
        """
    def step(self, /, order: SupportsGetItem[int, SupportsIndex] | None = None) -> None:
        """
        Manual mode: lets the nodes in `order` (all, in index order, if `None`)
        update once each, one after another.
        """
    def stop(self, /) -> None:
        """
        Asks every node to stop and returns without waiting; `join` waits for
        the end. Does nothing once the network has stopped.
        """
    @property
    def workers(self, /) -> int:
        """
        The number of worker threads of the pool: the default or the given
        number, but not more than the nodes in the pool (0 if every node has a
        thread of its own).
        """
    def __new__(cls, /) -> NoReturn:
        """
        No constructor: calling the class raises `TypeError`.
        """

@final
class Schema:
    """
    The fields of a message, fixed when the network is built.

    `Schema(fields, carry)` takes `(name, shape, (code, bits, lanes))` per
    field, the dtype as a DLPack `DLDataType`: `fields` for the value and
    `carry` for the data the message carries (none by default). A tensor
    message has one field named `""`; a TypedDict message has one field per
    key, in key order. Every extent must be at least 1.

    Raises:
        ValueError: No value fields, duplicate names among the value fields or
            among the carry fields, an extent below 1, a dtype that is not a
            scalar of whole bytes, or a field too large to allocate.
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
        carry: SupportsGetItem[
            int,
            tuple[
                str,
                SupportsGetItem[int, SupportsIndex],
                tuple[SupportsIndex, SupportsIndex, SupportsIndex],
            ],
        ] = ...,
    ) -> Self: ...

@final
class Slot:
    """
    The latest value of one writer, read by any number of readers.
    """
    @property
    def name(self, /) -> str: ...
    def new_carry(self, /, values: Sequence[Any]) -> Carry:
        """
        A new carry of `values` (one per carry field), from the pools of this
        slot's writer: what a restore gives back to the port that reads this
        slot. Raises as `Outlet.send` does.
        """
    def read(self, /) -> Message: ...
    def restore(
        self, /, values: Sequence[Any], time_ns: SupportsIndex, sent: bool
    ) -> None:
        """
        Sets the value, its time and whether it was sent as they were when the
        state was taken (`sent` is false for an initial value).

        `values` are the value fields, then the carry fields. Call it only while
        no node runs (before the start, or while every node is paused): it
        writes beside the slot's own writer. Raises as `Outlet.send` does.
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
    A time axis that a run's nodes measure their `dt` on.

    `Timer()` starts at `start` and runs with the real clock, `time_scale`
    seconds of the timer per real second. Give it to
    `mujou.run(..., timer=...)` to run a network on a time of its own: a pause
    of that run (as `Run.state_dict` takes) then stops only this timer, not
    `global_timer`, which every other run uses by default. Every time of the
    run is a `Timestamp` on it.

    Only the host controls a timer: `sleep`, `pause`, `resume`,
    `set_time_scale` (and `VirtualTimer.advance`) raise `RuntimeError` in a
    node's hook, where a node would wait for, or change the `dt` of, the other
    nodes. A node reads the time of its run with `self.timer`.

    `pause` stops the time of every run on the timer at one time T until
    `resume`. Take the states of several runs on one timer while it is paused,
    so that all of them are of the same time T.

    The time is held in nanoseconds and saturates at about 584 years.

    Raises:
        ValueError: `time_scale` is not finite and > 0.
    """
    def __new__(
        cls,
        /,
        *,
        start: Timestamp = ...,
        time_scale: SupportsFloat | SupportsIndex = 1.0,
    ) -> Self: ...
    @override
    def __repr__(self, /) -> str: ...
    def is_paused(self, /) -> bool:
        """
        Whether the time is stopped, by `pause` or by a run on this timer that
        is starting or taking its state. The clock of a `VirtualTimer` standing
        still does not count as paused.
        """
    def now(self, /) -> Timestamp:
        """
        The time now.
        """
    def pause(self, /) -> None:
        """
        Stops the time until `resume`.

        While paused, the nodes of the runs on this timer do not update (the
        first `dt` after `resume` does not count the pause), and `sleep` does
        not return until resumed. `Run.state_dict` still works. This is a
        switch, not a count: pausing twice needs one `resume`. A run that takes
        its state holds the time paused on its own, apart from this switch, so
        `resume` does not end that, and its end does not end this pause.

        Raises:
            RuntimeError: This is called from a node's hook.
        """
    def resume(self, /) -> None:
        """
        Turns off the switch of `pause`: the time moves again from where it
        stopped, unless a run on this timer is taking its state.

        Raises:
            RuntimeError: This is called from a node's hook.
        """
    def set_time_scale(self, /, scale: SupportsFloat | SupportsIndex) -> None:
        """
        Sets how many seconds of this timer pass per real second.

        Raises:
            ValueError: `scale` is not finite and > 0. Use `pause` to stop the
                time.
            RuntimeError: This is called from a node's hook.
        """
    def sleep(self, /, seconds: SupportsFloat | SupportsIndex) -> None:
        """
        Blocks for `seconds` of this timer's time, following the scale and the
        pauses: while paused, it does not return until resumed. Waits without
        the GIL and checks for signals (Ctrl-C) in between.

        Raises:
            ValueError: `seconds` is negative or not finite.
            RuntimeError: This is called from a node's hook.
        """
    def time_scale(self, /) -> float:
        """
        How many seconds of this timer pass per real second.
        """

@final
class Timestamp:
    """
    A time on a timer.

    `float(ts)` is the time in seconds and `int(ts)` its whole seconds; `ts.ns`
    is the exact time in nanoseconds. Timestamps compare and hash by time, so
    compare only those of one timer: a timestamp does not say which timer it is
    on. `a - b` is the time from `b` to `a` in seconds, a `float` that is
    negative when `b` is later. Timestamps pickle and copy.
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
    @property
    def ns(self, /) -> int:
        """
        The time in nanoseconds.
        """

@final
class VirtualTimer(Timer):
    """
    A timer for tests: its time moves only when `advance` moves it.

    It starts at `start`. Its time scale does not apply to `advance`.
    `mujou.testing.run` runs a network on a new one unless given one, and
    `ManualRun.step(dt)` advances it.
    """
    def __new__(cls, /, *, start: Timestamp = ...) -> Self: ...
    def advance(self, /, seconds: SupportsFloat | SupportsIndex) -> None:
        """
        Moves the time forward by `seconds`, at once.

        Raises:
            ValueError: `seconds` is negative or not finite.
            RuntimeError: The timer is paused (by `pause`, or by a run on it
                that is starting or taking its state), or this is called from a
                node's hook.
        """

def current() -> "object | None":
    """
    The driver (`mujou._core.driver.Driver`) of the node whose hook the
    calling thread runs; `None` outside a hook.
    """

def start(
    nodes: SupportsGetItem[
        int,
        tuple[
            Any,
            str,
            SupportsFloat | SupportsIndex,
            bool,
            SupportsIndex | None,
            Any | None,
        ],
    ],
    clock: ReadOnlyTimer,
    *,
    manual: bool = False,
    workers: SupportsIndex | None = None,
) -> Runtime:
    """
    Starts `nodes` on `clock` and returns once all of them have started (and
    restored their states) and the time runs again. If a node raised
    meanwhile, raises `NetworkError` after the network has stopped.

    Each node is a record `(driver, name, period, dedicated_thread,
    last_update_ns, state)`. The runtime runs each hook of the node as one call
    of a method of `driver` (`mujou._core.driver.Driver`): `start()`,
    `restore(state)`, `update(dt)`, `pause()`, `resume()`, `exception()`,
    `stop()` and `capture()`, which returns the node's state. `name` names the
    node in the errors and the logs, which name the hooks as the node's author
    knows them (`on_update`, `load_state_dict`, `state_dict`, ...). `period` is
    its period on the timer in seconds (>= 0), rounded to the nearest ns and at
    least 1 ns, so that 0.0 updates as soon as any time has passed; the manual
    mode ignores it. `dedicated_thread` runs the node on a thread of its own
    instead of the pool. `last_update_ns` is where its first Δt starts (the
    time at the start if `None`), and `state` what `restore` restores after
    `start` (no call if `None`).

    The nodes run on `clock`, the reading side of the run's timer: it measures
    their Δt, and the runtime freezes its timer to start (so the time does not
    run while the nodes start) and to take the state. Create the `Outlet`s and
    the `InputGroup` of this run with the same clock. The runtime stops the
    clock when it is asked to stop or a node raises, so that no time of the run
    is later than its end.

    `current()` returns the driver while one of its methods runs.

    The nodes not on a thread of their own share a pool of `workers` threads,
    by default as many as the available parallelism (half of it, at least 1,
    while the GIL is enabled), never more than the nodes in the pool.

    If the wait is interrupted (Ctrl-C), stops the network and raises once it
    has stopped; interrupting that wait too leaves it stopping in the background.
    """
