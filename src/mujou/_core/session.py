"""The run of a built network: slots for its ports, the runtime of its nodes, and the
host's reads and writes (``specs/network.md``, ``specs/execution-model.md``).

``mujou.run`` creates one ``Session`` per run, on the run's timer. It lays out the
slots of every port on that timer, binds each port to them (``Binding``), and
starts the runtime on the same timer. Values cross the boundary to the core as
copies: a received message is a new tensor, and a sent one is copied into the core.
"""

import atexit
import contextlib
import copy
import math
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast, final

import torch

from mujou._core import _native
from mujou._core.network import ports
from mujou._core.node import (
    Node,
    capture_state,
    end_thread,
    period_ns,
    set_timer,
    thread_ident,
)
from mujou._core.port import Channel, Port, bind, initial
from mujou._core.spec import Spec, fields, join, split
from mujou._core.timer import ReadOnlyTimer, Timer, native, stopped

if TYPE_CHECKING:
    from mujou._core.network import Graph

# Any (here and below): ports and messages of every type are handled alike.
type AnyPort = Port[Any, Any, Any]
type AnySlot = _native.Slot | _native.InputSlot

# Real seconds between the checks of `subscribe` for the end of the run.
_POLL: Final = 0.05
# Where the host receives values.
_CPU: Final = torch.device("cpu")


def _spec(spec: object) -> Spec | None:
    # Port accepts only TensorSpec, DictSpec or None (checked when it is made).
    return cast(Spec | None, spec)


def _host_tensors(spec: Spec, msg: object, where: str) -> list[torch.Tensor]:
    """Split ``msg`` by ``spec``, cut the gradients and bring the tensors to host
    memory for the core to copy.

    A CUDA tensor is copied on the calling thread's current stream (a node's own,
    in its hooks), and only that stream is synchronized.
    """
    return [t.detach().cpu() for t in split(spec, msg, where=where)]


def _value(spec: Spec, message: _native.Message, device: torch.device) -> object:
    """Copy ``message`` into new tensors on ``device`` and build the message."""
    return join(
        spec, [torch.from_dlpack(b).to(device, copy=True) for b in message.buffers]
    )


@final
class Binding:
    """What one port of a running network reads and writes.

    ``Port.recv`` / ``send`` delegate here once the network is built.
    """

    def __init__(
        self,
        owner: Node,
        path: str,
        port: AnyPort,
        outlet: _native.Outlet | None,
        sources: Sequence[tuple[Channel | None, AnySlot]],
    ) -> None:
        self._owner = owner
        self._path = path
        self._send_spec = _spec(port.send_spec)
        self._recv_spec = _spec(port.recv_spec)
        self._multi = port.multi
        self._outlet = outlet
        # (channel, slot) in connection order; a single-peer port has exactly one,
        # whose channel is None while unconnected (the host writes it).
        self._sources = list(sources)
        self._indices = {channel: k for k, channel in enumerate(port.channels)}

    def recv(self, with_time: bool) -> object:
        self._check_owner("receives on")
        spec = self._recv_spec
        if spec is None:
            raise TypeError(f"the port {self._path!r} receives no messages")
        device = self._owner.device
        received: dict[Channel | None, object] = {}
        for channel, slot in self._sources:
            message = slot.read()
            value = _value(spec, message, device)
            received[channel] = (value, message.timestamp) if with_time else value
        if self._multi:
            return received
        (value,) = received.values()
        return value

    def send(self, msg: object) -> None:
        outlet = self._sending()
        outlet.send(self._tensors(msg))

    def send_to(self, channel: Channel, msg: object) -> None:
        outlet = self._sending()
        index = self._indices.get(channel)
        if index is None:
            raise ValueError(f"{channel!r} is not a channel of {self._path!r}")
        outlet.send_to(index, self._tensors(msg))

    def _check_owner(self, verb: str) -> None:
        # A port is read and written only by its owner: this keeps each slot to
        # one writer, and a node to the slots of its own ports (locality).
        if threading.get_ident() != thread_ident(self._owner):
            raise RuntimeError(
                f"only the node that owns the port {self._path!r} {verb} it, from its "
                "own hooks while the network runs (state_dict runs on another thread)"
            )

    def _sending(self) -> _native.Outlet:
        self._check_owner("sends on")
        if self._outlet is None:
            raise TypeError(f"the port {self._path!r} sends no messages")
        return self._outlet

    def _tensors(self, msg: object) -> list[torch.Tensor]:
        spec = cast(Spec, self._send_spec)  # an outlet exists only with a send spec
        return _host_tensors(spec, msg, self._path)


@final
class Session:
    """The slots and the runtime of one run of a built network, on one timer.

    ``start`` restores the state, writes the initial inputs and starts the nodes,
    once.
    """

    def __init__(self, graph: Graph, timer: Timer) -> None:
        self._graph = graph
        self._paths = [path for path, _ in graph.nodes]
        self._nodes = [node for _, node in graph.nodes]
        # The one timer of the run: every slot, the host's inputs and the runtime
        # use it, so that every time in the run is on the axis dt is measured on.
        self._clock = native(timer)
        # What the nodes see of it: they read the time and change nothing.
        self._view = ReadOnlyTimer(timer)
        # One group for the whole network: a write switches together only within a
        # group.
        self._group = _native.InputGroup(timer=self._clock)
        self._outlets: dict[AnyPort, _native.Outlet] = {}
        self._inputs: dict[AnyPort, tuple[_native.InputSlot, Spec]] = {}
        # Every slot by name, in the order of the graph's ports, for the state tree.
        self._slots: dict[str, tuple[AnySlot, Spec]] = {}

        owners = ports(graph)
        # Every outlet first: a port reads the channel slots of its peers' outlets.
        for port in owners:
            self._add_outlet(port)
        for port, (index, _) in owners.items():
            path = graph.port_path(port)
            sources = self._sources(port, path)
            owner = graph.nodes[index][1]
            bind(port, Binding(owner, path, port, self._outlets.get(port), sources))

        self._runtime: _native.Runtime | None = None
        # The timer when shutdown first asked the run to stop.
        self._stop_lock = threading.Lock()
        self._stop_timer: dict[str, int | float] | None = None
        # Set once by _settle: the nodes' timer after the end (None after an
        # exception).
        self._settle_lock = threading.Lock()
        self._ended: tuple[ReadOnlyTimer | None] | None = None

    def _add_outlet(self, port: AnyPort) -> None:
        spec = _spec(port.send_spec)
        if spec is None:
            return
        outlet = _native.Outlet(
            _native.Schema(fields(spec)),
            len(port.channels),
            self._graph.port_path(port),
            timer=self._clock,
        )
        self._outlets[port] = outlet
        slots = [outlet.outbox, *(outlet.channel(k) for k in range(len(port.channels)))]
        for slot in slots:
            self._slots[slot.name] = (slot, spec)
        values = initial(port)
        if values is not None:
            # In place of the zeros, with their time (the timer's when the run was
            # laid out). A restore later replaces them, as it does the zeros.
            time_ns = outlet.outbox.read().timestamp.ns
            for slot in slots:
                slot.restore(values, time_ns)

    def _sources(
        self, port: AnyPort, path: str
    ) -> list[tuple[Channel | None, AnySlot]]:
        """The slots ``port`` reads, as ``Binding`` takes them."""
        spec = _spec(port.recv_spec)
        if spec is None:
            return []
        if port.channels:
            # Each end writes its k-th channel slot for its k-th channel.
            sources: list[tuple[Channel | None, AnySlot]] = []
            for channel in port.channels:
                peer = channel.other(port)
                k = peer.channels.index(channel)
                sources.append((channel, self._outlets[peer].channel(k)))
            return sources
        if port.multi:
            return []
        # An unconnected single-peer port reads what the host writes.
        slot = self._group.input(_native.Schema(fields(spec)), path)
        self._inputs[port] = (slot, spec)
        self._slots[slot.name] = (slot, spec)
        return [(None, slot)]

    # --- run and control

    def start(
        self,
        *,
        manual: bool,
        state: Mapping[str, Any] | None,
        inputs: Mapping[AnyPort, object] | None,
        pause_timeout: float,
    ) -> None:
        """Restore ``state``, write ``inputs`` and start the nodes.

        Raises as ``mujou.run`` does. Anything but ``NetworkError`` comes before any
        node started.
        """
        periods = [period_ns(node) for node in self._nodes]
        restored = None if state is None else self._restore(state)
        if inputs:
            items = self._items(inputs)
            if restored is None:
                self._group.write(items)
            else:
                # The values belong to the restored time T, not to the timer's
                # time now. Nobody reads yet, so restoring each slot (after
                # checking them all above) switches them together.
                time_ns = int(restored["timer_state"]["time_ns"])
                for slot, tensors in items:
                    slot.restore(tensors, time_ns)
        # The names of the nodes in the runtime's errors and logs: a node that is
        # itself the root (at the path "") by its class.
        names = [
            path or type(node).__name__
            for path, node in zip(self._paths, self._nodes, strict=True)
        ]
        runtime = _native.Runtime(self._nodes, names, timer=self._clock)
        self._runtime = runtime
        for session in list(_RUNNING):
            session._settle()
        _RUNNING.add(self)
        _watch_exit()
        for node in self._nodes:
            set_timer(node, self._view)
        # What this thread queued for the nodes (tensors made in __init__ or by
        # the restore) is done before their streams use it.
        self._synchronize_current_streams()
        try:
            runtime.start(
                manual=manual,
                pause_timeout=pause_timeout,
                period_ns=periods,
                **(restored or {}),
            )
        except _native.NetworkError:
            self._settle()  # a node raised while starting: this run is over
            raise
        except Exception:
            # Refused before any node started (e.g. the timer cannot be set back):
            # the network may run once the cause is gone.
            self._runtime = None
            _RUNNING.discard(self)
            for node in self._nodes:
                set_timer(node, None)
            raise

    def step(self, order: Sequence[Node]) -> None:
        runtime = self._started()
        indices = [self._index(node) for node in order] if order else None
        try:
            runtime.step(indices)
        finally:
            self._settle()  # a node may have raised: the host sees the end here

    def check(self) -> None:
        if self._runtime is not None:
            self._settle()
            self._runtime.check()

    def shutdown(self, timeout: float) -> None:
        wait = _wait(timeout)
        runtime = self._runtime
        if runtime is None:
            return
        try:
            with self._stop_lock:
                if not runtime.is_finished:
                    # A normal end is a pause at this time that never resumes:
                    # the state after the end is taken at it.
                    if self._stop_timer is None:
                        self._stop_timer = self._clock.state_dict()
                    # The runtime's shutdown waits without a limit, so a finite
                    # timeout asks to stop and waits here with the limit. On a
                    # timeout the request stands and the network goes on
                    # stopping.
                    if wait is not None:
                        runtime.request_stop()
            if wait is None:
                runtime.shutdown()
                return
            if not runtime.join(wait):
                raise TimeoutError(
                    f"the run did not stop within {timeout} s; it goes on "
                    "stopping, and join() waits for it"
                )
        finally:
            self._settle()

    def join(self, timeout: float) -> None:
        wait = _wait(timeout)
        runtime = self._started()
        try:
            if not runtime.join(wait):
                raise TimeoutError(
                    f"the run did not stop within {timeout} s; it keeps running"
                )
        finally:
            self._settle()

    def _settle(self) -> None:
        """Once the run has ended, stop keeping the session, forget the nodes'
        threads and stop their ``timer``, once.

        After a normal end each node's ``timer`` stays at the time of the state
        from then on (``state_dict`` after the end is taken at it); after an end
        by an exception the nodes have no timer.
        """
        runtime = self._runtime
        if runtime is None or not runtime.is_finished:
            return
        with self._settle_lock:
            if self._ended is not None:
                return
            _RUNNING.discard(self)
            ended = self._ended_timer(runtime)
            for node in self._nodes:
                end_thread(node)
                set_timer(node, ended)
            self._ended = (ended,)

    def _ended_timer(self, runtime: _native.Runtime) -> ReadOnlyTimer | None:
        """The timer stopped at the end of a normal end, or ``None`` after an
        exception.

        The time is the one ``shutdown`` read when it asked the run to stop, as a
        pause would have frozen it, and not earlier than any update or slot (an
        ``on_stop`` may still send after the request).
        """
        try:
            runtime.check()
        except _native.NetworkError:
            return None
        # Only shutdown ends a run normally, and it reads the timer first.
        stop_timer = cast(dict[str, int | float], self._stop_timer)
        latest = max(
            [
                int(stop_timer["time_ns"]),
                *runtime.last_update_ns(),
                *(slot.read().time_ns for slot, _ in self._slots.values()),
            ]
        )
        return stopped(_native.Timestamp(ns=latest), float(stop_timer["scale"]))

    def _started(self) -> _native.Runtime:
        # Only a started session reaches the host (through its Run).
        return cast(_native.Runtime, self._runtime)

    def _index(self, node: Node) -> int:
        for i, n in enumerate(self._nodes):
            if n is node:
                return i
        raise ValueError(f"{type(node).__name__} is not a node of this run")

    # --- host input and output

    def write[P: AnyPort](self, values: Mapping[P, object]) -> _native.Timestamp:
        runtime = self._started()
        finished = runtime.is_finished
        self.check()  # after is_finished: an end by an exception raises here
        if finished:
            raise RuntimeError("the run has stopped; no node reads what is written now")
        return self._group.write(self._items(values))

    def _items(
        self, values: Mapping[Any, object]
    ) -> list[tuple[_native.InputSlot, list[torch.Tensor]]]:
        """Check ``values`` for the host's input slots and copy them to the host."""
        items: list[tuple[_native.InputSlot, list[torch.Tensor]]] = []
        for port, value in values.items():
            slot, spec = self._writable(port)
            items.append(
                (slot, _host_tensors(spec, value, self._graph.port_path(port)))
            )
        return items

    def read(self, port: AnyPort) -> tuple[object, _native.Timestamp]:
        self.check()
        outbox, spec = self._outbox(port)
        message = outbox.read()
        return _value(spec, message, _CPU), message.timestamp

    def subscribe(self, port: AnyPort) -> Iterator[tuple[object, _native.Timestamp]]:
        runtime = self._started()
        self.check()
        outbox, spec = self._outbox(port)
        # The value at the time of this call comes first.
        return self._follow(runtime, outbox, spec, outbox.read())

    def _follow(
        self,
        runtime: _native.Runtime,
        outbox: _native.Slot,
        spec: Spec,
        last: _native.Message,
    ) -> Iterator[tuple[object, _native.Timestamp]]:
        self.check()
        yield _value(spec, last, _CPU), last.timestamp
        while True:
            self.check()
            finished = runtime.is_finished
            # After the end nothing is written, so one last look without waiting.
            message = outbox.wait_newer(last, 0.0 if finished else _POLL)
            if message is None:
                if finished:
                    self.check()  # the end may have come from an exception
                    return
                continue
            last = message
            yield _value(spec, message, _CPU), message.timestamp

    def _writable(self, port: AnyPort) -> tuple[_native.InputSlot, Spec]:
        if port in self._inputs:
            return self._inputs[port]
        path = self._graph.port_path(port)  # ValueError for a port outside the tree
        if port.recv_spec is None:
            reason = "receives no messages"
        elif port.channels:
            reason = "is connected (only unconnected ports can be written)"
        else:
            reason = "takes many peers (only single-peer ports can be written)"
        raise ValueError(f"cannot write the port {path!r}: it {reason}")

    def _outbox(self, port: AnyPort) -> tuple[_native.Slot, Spec]:
        path = self._graph.port_path(port)  # ValueError for a port outside the tree
        outlet = self._outlets.get(port)
        if outlet is None:
            raise ValueError(f"cannot read the port {path!r}: it sends no messages")
        return outlet.outbox, cast(Spec, port.send_spec)

    # --- state

    def state_dict(self) -> dict[str, Any]:
        runtime = self._started()
        finished = runtime.is_finished
        self.check()  # after is_finished: an end by an exception raises here
        if not finished:
            try:
                return runtime.capture(
                    lambda states: self._tree(
                        states, runtime.last_update_ns(), self._clock.state_dict()
                    )
                )
            except _native.NetworkError:
                raise
            except RuntimeError:
                # The run ended while the request waited: take the state at the
                # end below. Anything else is a real error.
                if not runtime.is_finished:
                    raise
                self.check()
        self._settle()
        return self._ended_tree(runtime)

    def _ended_tree(self, runtime: _native.Runtime) -> dict[str, Any]:
        """Take the state of a run that has ended normally, on this thread.

        The nodes' threads are gone, so this thread calls their ``state_dict``
        (through ``_capture_state``, which waits for a CUDA node's stream). The
        tree's timer is the stopped one ``_settle`` gave the nodes.
        """
        ended = cast(tuple[ReadOnlyTimer], self._ended)[0]
        stop_timer = cast(dict[str, int | float], self._stop_timer)
        states = [capture_state(node) for node in self._nodes]
        timer = {**stop_timer, "time_ns": ended.now().ns}
        return self._tree(states, runtime.last_update_ns(), timer)

    def _restore(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Check ``state``, restore the slots and return what ``Runtime.start``
        restores (the nodes' states, their last update times and the timer).

        Raises:
            ValueError: The tree does not match the network; nothing changed.
        """
        for key in ("nodes", "slots", "last_update_ns", "timer"):
            if key not in state:
                raise ValueError(f"the state tree has no {key!r}")
        nodes = cast(Mapping[str, object], state["nodes"])
        slots = cast(Mapping[str, object], state["slots"])
        last_update_ns = cast(Mapping[str, object], state["last_update_ns"])
        timer = cast(Mapping[str, object], state["timer"])
        _same_keys("nodes", nodes, self._paths)
        _same_keys("slots", slots, self._slots)
        _same_keys("last_update_ns", last_update_ns, self._paths)
        _same_keys("timer", timer, ("time_ns", "scale"))
        for path in self._paths:
            if not _is_int(last_update_ns[path]):
                raise ValueError(
                    f"the state tree's last_update_ns[{path!r}] is not an int"
                )
        if not _is_int(timer["time_ns"]) or not isinstance(timer["scale"], int | float):
            raise ValueError(
                "the state tree's timer must have an int time_ns and a scale"
            )
        # Check and copy everything before changing anything, so that an error
        # changes nothing.
        restores = self._slot_restores(slots)
        # Copies: the nodes update their state in place, which must not reach the
        # caller's tree (or another network loaded from it).
        states = [copy.deepcopy(nodes[path]) for path in self._paths]
        # The copies were made on this thread's streams; the nodes use them on
        # their own streams, which do not wait for these.
        self._synchronize_current_streams()
        for slot, tensors, time_ns in restores:
            slot.restore(tensors, time_ns)
        return {
            "states": states,
            "last_update_ns": [cast(int, last_update_ns[path]) for path in self._paths],
            "timer_state": {
                "time_ns": cast(int, timer["time_ns"]),
                "scale": cast(float, timer["scale"]),
            },
        }

    def _slot_restores(
        self, slots: Mapping[str, object]
    ) -> list[tuple[AnySlot, list[torch.Tensor], int]]:
        """Check each slot's ``(value, time_ns)`` in the tree and copy the value to
        the host, without restoring anything yet."""
        restores: list[tuple[AnySlot, list[torch.Tensor], int]] = []
        for name, (slot, spec) in self._slots.items():
            entry = slots[name]
            if (
                not isinstance(entry, tuple)
                or len(cast(tuple[object, ...], entry)) != 2
            ):
                raise ValueError(
                    f"the state tree's slot {name!r} is not (value, time_ns)"
                )
            value, time_ns = cast(tuple[object, object], entry)
            if not _is_int(time_ns):
                raise ValueError(
                    f"the state tree's slot {name!r} has a time that is not an int"
                )
            restores.append(
                (slot, _host_tensors(spec, value, name), cast(int, time_ns))
            )
        return restores

    def _synchronize_current_streams(self) -> None:
        """Wait for the calling thread's current stream on each CUDA device the
        nodes use: only this thread's work, not the nodes' streams nor the whole
        device, and no context on a device the network does not use."""
        for device in {n.device for n in self._nodes if n.device.type == "cuda"}:
            torch.cuda.current_stream(device).synchronize()

    def _tree(
        self,
        states: Sequence[object],
        last_update_ns: Sequence[int],
        timer: Mapping[str, int | float],
    ) -> dict[str, Any]:
        """Copy the state of the paused or ended network into a tree
        (``state_dict``).

        The nodes' streams are idle here (``Node._capture_state`` waited for them).
        """
        slots: dict[str, tuple[object, int]] = {}
        for name, (slot, spec) in self._slots.items():
            message = slot.read()
            slots[name] = (_value(spec, message, _CPU), message.time_ns)
        tree = {
            "nodes": {
                path: copy.deepcopy(state)
                for path, state in zip(self._paths, states, strict=True)
            },
            "slots": slots,
            "last_update_ns": dict(zip(self._paths, last_update_ns, strict=True)),
            "timer": dict(timer),
        }
        # The copies above queue on this thread's current streams. Finish them
        # before the runtime resumes the nodes, which update in place.
        self._synchronize_current_streams()
        return tree


def _wait(timeout: float) -> float | None:
    """Return ``timeout`` as the runtime's ``join`` takes it (``None``: no limit).

    Raises:
        ValueError: ``timeout`` is negative or NaN.
    """
    if not timeout >= 0:
        raise ValueError(f"timeout must be a number of seconds >= 0, got {timeout}")
    return None if timeout == math.inf else timeout


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same_keys(
    what: str, given: Mapping[str, object], expected: Sequence[str] | Mapping[str, Any]
) -> None:
    missing = [k for k in expected if k not in given]
    unknown = [k for k in given if k not in expected]
    if missing or unknown:
        raise ValueError(
            f"the state tree's {what} do not match this network: missing {missing}, "
            f"unknown {unknown}"
        )


# The sessions that have run and not yet been seen to end. Held strongly: a running
# network keeps running when the host drops it, until it is shut down here at the
# latest. The runtime's threads are not Python threads, and must not call into a
# finalizing interpreter.
_RUNNING: Final[set[Session]] = set()


def _shutdown_all() -> None:
    # Stop every session even if one fails (e.g. Ctrl-C while waiting for its
    # on_stop), then raise the first failure.
    failure: BaseException | None = None
    for session in list(_RUNNING):
        try:
            session.shutdown(math.inf)
        except _native.NetworkError:
            pass  # already logged, and the host has gone
        except BaseException as e:  # noqa: BLE001 - re-raised below
            failure = failure or e
    if failure is not None:
        raise failure


def _stop_at_exit() -> None:
    """Stop every running network once the main thread has ended.

    At the interpreter's exit, ``threading`` marks the main thread as ended
    before it joins the non-daemon threads, so this starts stopping the networks
    before that join and a thread iterating over ``subscribe`` ends with its
    network. That order is CPython's implementation, not its documented
    behavior; ``TestInterpreterExit`` in ``tests/mujou/_core/test_run.py`` guards
    it.
    """
    threading.main_thread().join()
    # On a failure the atexit call below tries again, and raises.
    with contextlib.suppress(Exception):
        _shutdown_all()


_watching = False
_watching_lock: Final = threading.Lock()


def _watch_exit() -> None:
    """Start the daemon thread of ``_stop_at_exit`` once, at the first run."""
    global _watching
    with _watching_lock:
        if _watching:
            return
        _watching = True
    threading.Thread(target=_stop_at_exit, name="mujou-exit", daemon=True).start()


# After the join of the non-daemon threads, wait until the networks have stopped
# (the daemon thread may still be stopping them) and stop those that started
# since.
atexit.register(_shutdown_all)
