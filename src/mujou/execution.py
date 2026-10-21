"""Running a tree of nodes once: ``mujou.run``, ``mujou.testing.run`` and the
``Run`` they return (``specs/network.md``).

A run starts in three stages:

1. Check: read the tree of the root (``tree.walk``), and check that none of its
   nodes has run and that the arguments go together. Nothing changes.
2. Lay out: make the run's timer, its slots (``wiring.wire``) and the drivers
   of its nodes, restore the slots of a state (``state.restore``, which also
   checks that the state has the shape of the tree) and write the inputs
   (checking that the host writes those ports). Only new objects change.
3. Mark every node as run, and start the nodes.

The kind of run (``Kind``) says what changes between ``mujou.run`` and
``mujou.testing.run``: the class of the run, whether the nodes update by hand,
and the timer.

A run refused in the first two stages has changed nothing that the user made:
the same instances run once the cause is fixed. Only the hooks of the nodes can
fail in the third, which a later run of the same nodes refuses.
"""

import atexit
import contextlib
import math
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, NamedTuple, Self, cast, final, override

import torch

from mujou import _native
from mujou._native import (
    ReadOnlyTimer,
    Timer,
    Timestamp,
    VirtualTimer,
    global_timer,
)
from mujou.codec import from_core
from mujou.current import running
from mujou.driver import Driver
from mujou.errors import NetworkError
from mujou.network import Network
from mujou.node import Node
from mujou.port import MultiPort, Port
from mujou.state import capture, cpu, restore, saved_time
from mujou.tree import Tree, walk
from mujou.types import Message
from mujou.wiring import Wiring, wire

# Real seconds between the checks of `subscribe` for the end of the run.
_POLL: Final = 0.05
# Where the host receives values.
_CPU: Final = torch.device("cpu")


class _NodeEntry(NamedTuple):
    """A node as ``_native.start`` takes it."""

    driver: Driver
    # How the runtime's errors and logs name the node.
    name: str
    period: float
    dedicated_thread: bool
    # Where the first dt starts (the time at the start if None).
    last_update_ns: int | None
    # What load_state_dict restores after on_start (nothing if None).
    state: object


class Run:
    """A run of a tree of nodes, as ``mujou.run`` returns it: what the host uses
    to talk to the running nodes.

    The nodes run in the background, on the run's worker threads (or threads of
    their own); the host's thread only writes inputs, reads outputs and takes
    the state:

    - ``write`` inputs, ``read`` / ``read_with_time`` / ``subscribe`` /
      ``wait_for`` outputs, take a ``state_dict``.
    - ``shutdown()`` to stop the run, or ``join()`` to wait until it stops.
    - After a normal end: take the final ``state_dict``, and read the nodes'
      attributes (a module they learned, say).

    A ``with`` statement shuts the run down at its end. These methods are for
    the host: calling them from a node's hook raises ``RuntimeError``.

    If a node raises, the whole run stops. If that happens while ``mujou.run``
    starts the nodes, ``mujou.run`` raises ``NetworkError``; otherwise every later
    call of ``write``, ``read``, ``read_with_time``, ``subscribe`` (and its
    iterator), ``wait_for``, ``state_dict``, ``shutdown``, ``join`` and
    ``ManualRun.step`` raises it. Its ``__cause__`` is the node's exception.
    The state at the exception is not kept. A ``with`` left by another
    exception lets that exception through instead.

    A run keeps running when the host drops it, until ``shutdown`` or, at the
    latest, the interpreter's exit: once the non-daemon threads have ended, the
    exit shuts down every run still running and waits for their ``on_stop``. A
    thread that iterates over ``subscribe`` keeps the exit waiting until its run
    ends, so shut the run down before the program ends (``with`` does).

    ``mujou.run`` and ``mujou.testing.run`` make it.

    Example::

        from typing import override

        import torch

        import mujou
        from mujou import Node, Port, TensorSpec

        VECTOR = TensorSpec((2,))


        class Error(Node):  # sends its input minus its teacher
            def __init__(self) -> None:
                super().__init__()
                self.prev = Port(recv=VECTOR)
                self.teacher = Port(recv=VECTOR)
                self.next = Port(send=VECTOR)

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                self.next.send(self.prev.recv() - self.teacher.recv())


        node = Error()
        x, y = torch.ones(2), torch.zeros(2)
        with mujou.run(node, inputs={node.prev: x, node.teacher: y}) as run:
            run.write({node.prev: 2 * x, node.teacher: x})  # switch together
            run.timer.sleep(0.1)  # present the input for 0.1 s of the run's time
            error, t = run.read_with_time(node.next)
        state = run.state_dict()  # the state at the end, to save
    """

    def __init__(
        self,
        wiring: Wiring,
        runtime: _native.Runtime,
        timer: Timer,
        clock: ReadOnlyTimer,
    ) -> None:
        self._wiring = wiring
        self._runtime = runtime
        self._timer = timer
        # The one clock of the run on its timer, which stops at its end.
        self._clock = clock

    @property
    def timer(self) -> Timer:
        """The run's timer: the ``timer`` given to ``mujou.run``,
        ``mujou.time.global_timer`` when none was given, or, for a run restored
        from a state, a new ``Timer`` that started at the time of the state.

        Every node's ``dt`` and every time of the run is on it. The host waits,
        pauses and scales the time of the run with it."""
        return self._timer

    # P: a dict built beforehand has a precise key type, which a Mapping keyed by
    # Port[Any, Any] would reject (Mapping keys are invariant).
    def write[P: Port[Any, Any]](self, values: Mapping[P, Message]) -> Timestamp:
        """Write messages to unconnected ports that receive, and return the time
        they are stamped with.

        The values are detached and copied. All of them switch together and bear
        the same time: at any one moment, a node sees either all of them or none.
        (A node that reads two of them one after the other may still see a later
        write in between; compare the times to tell.) Any such port of the run
        can be written, exposed by the root or not.

        The time is the run's timer at the write, as a node's ``recv_with_time``
        and the ``"in"`` slots of ``state_dict`` show it; ``read_with_time``
        shows the times of sends, not of writes. An empty ``values`` changes
        nothing and returns the time all the same.

        Raises:
            RuntimeError: The run has stopped normally (no node reads what is
                written then).
            ValueError: A port is not an unconnected ``Port`` of this run that
                receives; or a shape or a dtype differs from the spec.
            TypeError: A message is not a tensor, or a dict of tensors with the
                spec's keys.
            NetworkError: The run stopped because a node raised.
        """
        _host()
        self._ensure_running()
        return self._wiring.write(values)

    def read[S](self, port: Port[S, Any] | MultiPort[S, Any]) -> S:
        """Return the latest message ``port`` sent.

        Any port of the run that sends can be read, so a connected port shows
        an inner node. Only ``send`` (to every peer, not ``MultiPort.send_to``)
        updates the value read here; before the first one it is the port's
        ``initial`` (zeros by default) or the value restored from ``state``. The
        tensors are new copies on the CPU. ``read_with_time`` also returns the
        time it was sent at.

        Raises:
            ValueError: ``port`` is not a port of this run that sends.
            NetworkError: The run stopped because a node raised.
        """
        return self.read_with_time(port)[0]

    def read_with_time[S](
        self, port: Port[S, Any] | MultiPort[S, Any]
    ) -> tuple[S, Timestamp | None]:
        """Return the latest message ``port`` sent, as ``read`` does, and the time
        it was sent at: ``None`` before the first ``send`` (the port's
        ``initial``, or the restored value of a port that had not sent).

        The time is on the run's timer. A ``Timestamp`` does not say which timer
        it is on: compare it only with times of the same timer (``run.timer``).

        Raises:
            ValueError: As ``read`` raises.
            NetworkError: The run stopped because a node raised.
        """
        _host()
        self._check()
        outbox, spec = self._wiring.outbox(port)
        message = outbox.read()
        return cast(S, from_core(spec, message.values, _CPU)), message.timestamp

    def subscribe[S](
        self, port: Port[S, Any] | MultiPort[S, Any], *, timeout: float = math.inf
    ) -> Iterator[tuple[S, Timestamp | None]]:
        """Iterate over the messages ``port`` sends, with their times.

        The first step yields the latest message at the time of this call without
        waiting (before the first ``send``, the port's ``initial`` or the restored
        value, with the time ``read_with_time`` gives it). Each
        later step waits until a newer message is sent and yields the latest;
        messages sent in between are skipped. Only the consumer waits: the
        sending node never does. The iterator ends when the run shuts down
        normally. Without a ``timeout``, a thread iterating here ends only
        with the run. In a ``ManualRun`` nothing is sent unless ``step`` is
        called, so a wait here without a ``timeout`` on the thread that steps
        never ends.

        To wait for one message that meets a condition, use ``wait_for``.

        Args:
            port: The port to follow.
            timeout: Real (wall-clock) seconds, counted from this call, that the
                whole iteration may take, the time spent in the loop's body
                included. ``math.inf`` iterates as long as the run runs.

        Raises:
            ValueError: As ``read`` raises.
            TimeoutError: From the iterator, at a step after the first that
                begins once ``timeout`` has passed, or that would wait past it.
                The iterator ends; the run keeps running.
            NetworkError: The run stopped because a node raised (also from the
                iterator).
        """
        _host()
        self._check()
        outbox, spec = self._wiring.outbox(port)
        deadline = time.monotonic() + timeout
        # The value at the time of this call comes first.
        messages = self._follow(outbox, outbox.read(), timeout, deadline)
        return (
            (cast(S, from_core(spec, message.values, _CPU)), message.timestamp)
            for message in messages
        )

    def wait_for[S](
        self,
        port: Port[S, Any] | MultiPort[S, Any],
        until: Callable[[S], bool],
        *,
        timeout: float = math.inf,
    ) -> tuple[S, Timestamp | None]:
        """Wait until ``port`` sends a message that meets ``until``, and return it
        with its time (``None`` if the message that meets it is the one before
        the first ``send``, as ``read_with_time`` gives it).

        Checks the latest message first, then each newer one as ``subscribe``
        yields them (messages sent in between are skipped). Only the host waits:
        the nodes never do. In a ``ManualRun`` nothing is sent unless ``step`` is
        called, so a wait without a ``timeout`` on the thread that steps never
        ends. Here with the node ``Error`` of the example of ``Run``::

            node = Error()
            x = torch.ones(2)
            with mujou.run(node, inputs={node.prev: x}) as run:
                error, t = run.wait_for(
                    node.next, lambda e: torch.equal(e, x), timeout=5.0
                )

        Args:
            port: The port to follow.
            until: Called on each message on the host's thread; ``True`` ends
                the wait.
            timeout: Real (wall-clock) seconds, counted from this call, to wait
                at most. ``math.inf`` waits as long as the run runs.

        Raises:
            ValueError: As ``read`` raises.
            TimeoutError: ``timeout`` passed first. The run keeps running.
            RuntimeError: The run shut down normally first.
            NetworkError: The run stopped because a node raised.
        """
        for message, time_ in self.subscribe(port, timeout=timeout):
            if until(message):
                return message, time_
        raise RuntimeError(
            f"the run shut down before {self._wiring.tree.label(port)} sent a "
            "message that meets the condition"
        )

    def state_dict(self, *, timeout: float = math.inf) -> dict[str, Any]:
        """Copy the state of the run at one time, while it runs or after it ended
        normally.

        While it runs, this pauses the run's timer and nodes, copies the state
        and resumes, and returns once the run has resumed. It waits at most
        ``timeout`` for every node to pause. After a normal end,
        this copies the state as the nodes left it, calling each node's
        ``state_dict`` on this thread. The tree's timer holds the time at which
        the run was asked to stop, as if the run had paused then: no update or
        message of the run is later, though its time scale is the timer's when
        the state is taken. A run restored from it goes on from there, without
        the time that passed after the end.

        The tree is a plain ``dict`` of copies on the CPU (later updates of the
        nodes do not change it): save it with ``torch.save``, load it with
        ``torch.load`` (``weights_only=True`` reads it), and pass it to
        ``mujou.run(..., state=tree)`` to restore it::

            {
                "timer": {"time_ns": int, "time_scale": float},
                "nodes": {
                    "body.l1": {
                        "state": ...,  # the node's state_dict()
                        "last_update_ns": int,
                        "ports": {
                            "prev": {"in": E},
                            "next": {"out": E, "to": {"body.l2.prev": E}},
                        },
                    },
                    ...,
                },
            }

        A node is keyed by its path from the root (``""`` for a node that is the
        root itself), and each of its ports by its attribute. A port that sends
        has ``"out"``, its latest message (what ``read`` returns), and ``"to"``,
        its latest message to each peer (``MultiPort.send_to`` writes only one),
        by the path of the peer's port. An unconnected ``Port`` that receives has
        ``"in"``, what the host wrote last. Each ``E`` is ``{"value": message,
        "time_ns": int | None}``: ``time_ns`` is ``None`` while nothing was
        written.

        Args:
            timeout: Real (wall-clock) seconds to wait for every node to pause
                (the end of its current ``on_update``). ``math.inf`` waits as
                long as it takes.

        Raises:
            RuntimeError: This is called from a node's hook.
            TimeoutError: A node did not pause within ``timeout``; no state
                is taken and the run keeps running.
            NetworkError: The run stopped because a node raised (its state is
                not kept).
            Exception: After a normal end, what a node's ``state_dict``
                raised; the run stays ended normally.
        """
        _host()
        self._check()
        runtime = self._runtime
        tree = runtime.capture(
            lambda states: capture(
                self._wiring,
                states,
                runtime.last_update_ns(),
                self._clock.now().ns,
                self._clock.time_scale(),
            ),
            timeout,
        )
        return cpu(tree)

    def shutdown(self, *, timeout: float = math.inf) -> None:
        """Stop every node and return once every ``on_stop`` has returned.

        Does nothing after a normal end.

        Args:
            timeout: Real (wall-clock) seconds to wait for the end.
                ``math.inf`` waits as long as it takes.

        Raises:
            TimeoutError: The run did not stop within ``timeout``. The stop stays
                asked: the nodes take no more updates and the run goes on
                stopping in the background (an ``on_stop`` may still be
                running). ``join`` waits for the end.
            ValueError: ``timeout`` is negative or NaN; nothing changes.
            NetworkError: The run stopped because a node raised.
        """
        _host()
        try:
            stopped = self._runtime.shutdown(timeout)
        finally:
            self._forget_if_ended()
        if not stopped:
            raise TimeoutError(
                f"the run did not stop within {timeout} s; it goes on stopping, "
                "and join() waits for it"
            )

    def join(self, *, timeout: float = math.inf) -> None:
        """Wait until the run stops, without asking it to stop.

        It returns once another thread called ``shutdown`` and every ``on_stop``
        has returned; if a node raises, it raises instead. If nothing stops the
        run and ``timeout`` is ``math.inf``, it never returns.

        Args:
            timeout: Real (wall-clock) seconds to wait for the end.
                ``math.inf`` waits as long as it takes.

        Raises:
            TimeoutError: The run did not stop within ``timeout``. Nothing
                changes: a running run keeps running.
            NetworkError: The run stopped because a node raised.
        """
        _host()
        if not self._join(timeout):
            raise TimeoutError(
                f"the run did not stop within {timeout} s; it keeps running"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Shut the run down. When the body raised, its exception goes on and a
        ``NetworkError`` of the shutdown is dropped (the node's exception is in
        the log)."""
        try:
            self.shutdown()
        except NetworkError:
            if exc is None:
                raise

    def _check(self) -> None:
        """Raise ``NetworkError`` if a node raised."""
        try:
            self._runtime.check()
        finally:
            self._forget_if_ended()

    def _ensure_running(self) -> None:
        """Return if the run has not stopped.

        Raises:
            NetworkError: A node raised; the run has stopped.
            RuntimeError: The run has stopped normally.
        """
        finished = self._runtime.finished
        self._check()  # after finished: an end by an exception raises here
        if finished:
            raise RuntimeError("the run has stopped")

    def _join(self, timeout: float) -> bool:
        try:
            return self._runtime.join(timeout)
        finally:
            self._forget_if_ended()

    def _forget_if_ended(self) -> None:
        """Stop keeping the run for the exit once it has ended: every call that
        can see the end (a check, a join, a step) comes here, so a run that
        ended, by an exception or not, is not kept until the exit."""
        if self._runtime.finished:
            _RUNNING.discard(self)

    def _follow(
        self,
        outbox: _native.Slot,
        last: _native.Message,
        timeout: float,
        deadline: float,
    ) -> Iterator[_native.Message]:
        self._check()
        yield last
        while True:
            finished = self._runtime.finished
            self._check()  # after finished: an end by an exception raises here
            if finished:
                # After the end nothing is written, so one last look without
                # waiting.
                message = outbox.wait_newer(last, 0.0)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"no new message within {timeout} s; the run keeps running"
                    )
                # remaining first: a NaN timeout reaches wait_newer, which
                # rejects it.
                message = outbox.wait_newer(last, min(remaining, _POLL))
            if message is None:
                if finished:
                    return
                continue
            last = message
            yield message


class ManualRun(Run):
    """A run whose nodes update only when ``step`` lets them, for tests.

    ``mujou.testing.run`` returns it. The nodes start and stop as in any run, but
    between updates each waits for ``step``, which advances the run's
    ``VirtualTimer`` first: every ``dt`` is exactly the time the test stepped,
    and the result follows the discrete algorithm of the chosen order.
    """

    def __init__(
        self,
        wiring: Wiring,
        runtime: _native.Runtime,
        timer: VirtualTimer,
        clock: ReadOnlyTimer,
    ) -> None:
        super().__init__(wiring, runtime, timer, clock)
        self._virtual = timer
        # The index of each node in the runtime, for step.
        self._indices = {member.node: i for i, member in enumerate(wiring.tree.members)}

    @property
    @override
    def timer(self) -> VirtualTimer:
        """The run's timer: the ``VirtualTimer`` that ``mujou.testing.run`` made,
        or the one given to it."""
        return self._virtual

    def step(self, dt: float, *nodes: Node) -> None:
        """Advance the timer by ``dt`` seconds, then update each of ``nodes``
        once, one after another in the given order, and return when done.

        Without ``nodes``, every node updates in the order of the tree (each
        sees what the earlier ones sent in the same step, as Gauss-Seidel does;
        give them from the output side back for the order of Jacobi). The
        period of the nodes does not apply. Each ``dt`` a node sees is the time
        since its last update, the first one from the start of the run (or the
        last update restored). A node given twice in one ``step`` updates once:
        the second time, its time has not moved. A timer shared with other runs
        moves for them too.

        Raises:
            ValueError: A node is not in this run, or ``dt`` is negative or not
                finite.
            RuntimeError: The run has stopped normally, or the timer is paused.
            NetworkError: A node raised; the run has stopped.
        """
        _host()
        order = [self._index(node) for node in nodes] if nodes else None
        self._ensure_running()
        self.timer.advance(dt)
        try:
            self._runtime.step(order)
        finally:
            self._forget_if_ended()

    def _index(self, node: Node) -> int:
        index = self._indices.get(node)
        if index is None:
            raise ValueError(f"{type(node).__name__} is not a node of this run")
        return index


# P: as in Run.write.
def run[P: Port[Any, Any]](
    root: Node | Network,
    *,
    timer: Timer | None = None,
    inputs: Mapping[P, Message] | None = None,
    state: Mapping[str, Any] | None = None,
    workers: int | None = None,
) -> Run:
    """Run ``root``, a node or a network, in the background; return once every
    node has started.

    The nodes are at their paths from the root: those of a network at
    ``"l1"``, ``"body.l1"``, ..., and a node given as the root at ``""``. To run
    several side by side, put them in a ``Group``. The host writes and reads
    the ports themselves (``run.write({node.prev: x})``).

    Every node's ``on_start`` (and, with ``state``, the node's
    ``load_state_dict``) has returned, and the run's timer runs when this
    returns. A node runs once: run new instances to run again (the attributes
    of the nodes of a run stay readable after it). A run refused before any
    node started (by any error but ``NetworkError``) has changed nothing, and
    the same instances run once the cause is fixed.

    The run's nodes share a pool of ``workers`` threads, which it creates for
    itself; a node with ``dedicated_thread=True`` runs on a thread of its own
    instead. A worker runs one hook of one node at a time, the updates that are
    due earliest first. A hook that blocks holds its worker, so the other
    nodes' ``dt`` grows meanwhile.

    Args:
        root: The node or network to run.
        timer: The time axis of the run: every node's ``dt``, every message's
            time and every pause of the run is on it. ``None`` is
            ``mujou.time.global_timer``, which every run without a timer of its
            own shares.
        inputs: The initial values of unconnected ports that receive, as
            ``Run.write`` takes them, written when the run starts. Written after
            ``state``, they replace its values.
        state: A tree from ``Run.state_dict`` to restore. The slots, the
            nodes' states (through each node's ``load_state_dict``, after its
            ``on_start``) and their last update times continue from the time
            the tree was taken, on a new ``Timer`` of the run's own that starts
            at that time and time scale (so ``global_timer`` and the other runs
            do not move, and a pause of ``global_timer`` does not pause it).
            Build the tree to run as the one the tree came from: the same node
            paths, ports and peers.
        workers: The number of worker threads of the run (>= 1). ``None`` is
            as many as the CPUs the process may use (the logical cores, or
            fewer when the CPU affinity or a quota limits them), or half of
            them (at least 1) while the GIL is enabled at run time
            (``sys._is_gil_enabled()``), never more than the nodes on the
            workers. With no such node, the run has no worker.

    Example, with the node ``Error`` of the example of ``Run``::

        node = Error()
        with mujou.run(node, inputs={node.prev: torch.ones(2)}) as run:
            run.timer.sleep(0.1)
            error = run.read(node.next)

    Raises:
        RuntimeError: A node of the tree has already run (``"Leaky has already
            run; create a new instance (its attributes stay readable)"``); or
            this is called from a node's hook.
        TypeError, ValueError: The tree is invalid (as ``Node`` and
            ``Network`` say); or ``inputs`` or the ``initial`` value of a port
            do not match their specs (as ``Run.write`` raises).
        ValueError: ``workers`` is less than 1; ``timer`` and ``state`` are
            both given (a restored run runs on a timer of its own); or ``state``
            does not have the timer, nodes, ports and peers of the tree.
        NetworkError: A node raised while starting; the run has stopped.
    """
    return start(RUN, root, timer=timer, inputs=inputs, state=state, workers=workers)


@final
@dataclass(frozen=True, slots=True)
class Kind[R: Run, T: Timer]:
    """A kind of run: the class ``start`` makes, whether its nodes update only
    by ``ManualRun.step``, and the timer it runs on."""

    make: Callable[[Wiring, _native.Runtime, T, ReadOnlyTimer], R]
    manual: bool
    # The timer of a run given none.
    default: Callable[[], T]
    # The class of the timer of a restored run, made at the time and the time
    # scale of the state.
    restored: type[T]

    def timer(self, given: T | None, state: Mapping[str, Any] | None) -> T:
        """The timer of a run given ``given`` and ``state`` (not both).

        Raises:
            ValueError: The top of ``state`` is not that of a state tree.
        """
        if state is None:
            return self.default() if given is None else given
        time_ns, time_scale = saved_time(state)
        return self.restored(start=Timestamp(ns=time_ns), time_scale=time_scale)


# mujou.run: the nodes run freely, on global_timer by default.
RUN: Final = Kind(Run, manual=False, default=lambda: global_timer, restored=Timer)
# mujou.testing.run: the nodes update by hand, on a new VirtualTimer by default.
MANUAL_RUN: Final = Kind(
    ManualRun, manual=True, default=VirtualTimer, restored=VirtualTimer
)


def start[R: Run, T: Timer](
    kind: Kind[R, T],
    root: Node | Network,
    *,
    timer: T | None,
    inputs: Mapping[Any, object] | None,
    state: Mapping[str, Any] | None,
    workers: int | None,
) -> R:
    """Run ``root`` as ``mujou.run`` does, as a run of ``kind``."""
    _host()
    # 1. Check.
    tree = walk(root)
    _check_arguments(tree, timer, state, workers)
    # 2. Lay out. The timer of a restored run runs from here: the rest of the
    # layout (a few ms at most) is in its time and in the first dt of its nodes.
    run_timer = kind.timer(timer, state)
    clock = ReadOnlyTimer(run_timer)
    wiring = wire(tree, clock)
    restored = None if state is None else restore(wiring, state)
    if inputs:
        wiring.write(inputs)
    nodes = _nodes(wiring, clock, restored)
    # What this thread queued for the nodes (tensors made in __init__, say) is
    # done before their streams use it.
    _synchronize(tree)
    # 3. Mark and start.
    _RAN.update(member.node for member in tree.members)
    runtime = _native.start(nodes, clock, manual=kind.manual, workers=workers)
    handle = kind.make(wiring, runtime, run_timer, clock)
    _RUNNING.add(handle)
    return handle


def _check_arguments(
    tree: Tree,
    timer: Timer | None,
    state: Mapping[str, Any] | None,
    workers: int | None,
) -> None:
    """Check what the first stage of a run checks besides the tree.

    Raises:
        RuntimeError: A node of ``tree`` has already run.
        ValueError: ``timer`` and ``state`` are both given, or ``workers`` is
            less than 1.
    """
    for member in tree.members:
        if member.node in _RAN:
            raise RuntimeError(
                f"{tree.name(member.path)} has already run; create a new instance "
                "(its attributes stay readable)"
            )
    if timer is not None and state is not None:
        raise ValueError(
            "a restored run runs on a timer of its own, which starts at the time "
            "of the state; leave out timer"
        )
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")


def _nodes(
    wiring: Wiring, clock: ReadOnlyTimer, restored: list[tuple[object, int]] | None
) -> list[_NodeEntry]:
    """The nodes of ``wiring`` as ``_native.start`` takes them, each with its
    driver on ``clock``, and the state and the last update time of each from
    ``restored``."""
    members = wiring.tree.members
    saved = restored or [(None, None)] * len(members)
    return [
        _NodeEntry(
            Driver(member.node, clock, wires),
            wiring.tree.name(member.path),
            member.node.period,
            member.node.dedicated_thread,
            last_update_ns,
            state,
        )
        for member, wires, (state, last_update_ns) in zip(
            members, wiring.wires, saved, strict=True
        )
    ]


def _synchronize(tree: Tree) -> None:
    """Wait for the calling thread's current stream on each CUDA device the
    nodes use: only this thread's work, not the whole device, and no context on
    a device the run does not use."""
    devices = {member.node.device for member in tree.members}
    for device in devices:
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()


def _host() -> None:
    """Return if the calling thread is not running a node's hook.

    Raises:
        RuntimeError: It is.
    """
    if running() is not None:
        raise RuntimeError(
            "a node's hook cannot call the host methods of a run; a node talks "
            "only through its own ports"
        )


# The nodes of every run that has started: a node runs once. By identity (a node
# hashes by it).
_RAN: Final[weakref.WeakSet[Node]] = weakref.WeakSet()
# The runs that have started and not yet been seen to end. Held strongly: a run
# keeps running when the host drops it, until the exit at the latest.
_RUNNING: Final[set[Run]] = set()


def _stop_every_run() -> None:
    """Shut down the runs still running at the interpreter's exit, waiting for
    their ``on_stop``: their threads must not call into a finalizing
    interpreter."""
    for left in list(_RUNNING):
        with contextlib.suppress(NetworkError):  # logged, and the host has gone
            left.shutdown()


atexit.register(_stop_every_run)
