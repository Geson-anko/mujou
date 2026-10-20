"""The run of a built network: slots for its ports, the runtime of its nodes, and the
host's reads and writes (``specs/network.md``, ``specs/execution-model.md``).

``mujou.run`` creates one ``Session`` per run, on the run's timer. It lays out the
slots of every port on that timer, binds each port to them (``binding.py``), and
starts the runtime on the same timer. The state tree is ``state.py``'s, and the
registry of running runs ``lifetime.py``'s.
"""

import math
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final, cast, final

import torch

from mujou._core import _native, lifetime
from mujou._core.binding import AnyPort, AnySlot, Binding, as_spec, message_value
from mujou._core.graph import Graph, ports
from mujou._core.node import (
    Node,
    capture_state,
    checked_dedicated,
    checked_period,
    set_timer,
)
from mujou._core.port import Channel, bind, initial, label
from mujou._core.spec import Spec, fields, host_tensors
from mujou._core.state import SlotSpecs, StateDict, make_tree, read_tree
from mujou._core.timer import ReadOnlyTimer, Timer, native, stopped

# Real seconds between the checks of `subscribe` for the end of the run.
_POLL: Final = 0.05
# Where the host receives values.
_CPU: Final = torch.device("cpu")


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
        self._slots: dict[str, SlotSpecs] = {}
        # Every port with a recv_carry by its path: its binding, which holds the
        # carry it last received, the spec of that carry and the slot it reads
        # (whose writer's pools a restored carry is made from).
        self._carries: dict[str, tuple[Binding, Spec, AnySlot]] = {}
        # Every port of the run by how its errors name it.
        self._labels: dict[AnyPort, str] = {}
        self._bindings: dict[AnyPort, Binding] = {}

        owners = ports(graph)
        # Every outlet first: a port reads the channel slots of its peers' outlets.
        for port in owners:
            self._add_outlet(port)
        for port, (index, _) in owners.items():
            path = graph.port_path(port)
            sources = self._sources(port, path)
            owner_path, owner = graph.nodes[index]
            # The path alone ("next") would not say which node a root's port is of.
            name = path if owner_path else label(port)
            self._labels[port] = name
            binding = Binding(owner, name, port, self._outlets.get(port), sources)
            self._bindings[port] = binding
            bind(port, binding)
            carry = as_spec(port.recv_carry_spec)
            if carry is not None:
                # A port with a recv_carry reads exactly one slot.
                ((_, source),) = sources
                self._carries[path] = (binding, carry, source)
        # Once every port is bound: a port may pass on the carry of one bound
        # after it (build_tree checked it is a port of the same node).
        for port, binding in self._bindings.items():
            if port.carry_source is not None:
                binding.relay_from(self._bindings[port.carry_source])

        self._runtime: _native.Runtime | None = None
        # The timer when shutdown first asked the run to stop.
        self._stop_lock = threading.Lock()
        self._stop_timer: dict[str, int | float] | None = None
        # Set once by _settle: the nodes' timer after the end (None after an
        # exception).
        self._settle_lock = threading.Lock()
        self._ended: tuple[ReadOnlyTimer | None] | None = None

    def _add_outlet(self, port: AnyPort) -> None:
        spec = as_spec(port.send_spec)
        if spec is None:
            return
        carry = as_spec(port.send_carry_spec)
        outlet = _native.Outlet(
            _schema(spec, carry),
            len(port.channels),
            self._graph.port_path(port),
            timer=self._clock,
        )
        self._outlets[port] = outlet
        slots = [outlet.outbox, *(outlet.channel(k) for k in range(len(port.channels)))]
        for slot in slots:
            self._slots[slot.name] = (slot, spec, carry)
        values = initial(port)
        if values is not None:
            if carry is not None:
                # The initial value carries zeros, as the slots do from the start.
                zeros = carry.zeros()
                values = values + host_tensors(carry, zeros, where=label(port))
            # In place of the zeros, with their time (the timer's when the run was
            # laid out). A restore later replaces them, as it does the zeros.
            time_ns = outlet.outbox.read().timestamp.ns
            for slot in slots:
                slot.restore(values, time_ns, sent=False)

    def _sources(
        self, port: AnyPort, path: str
    ) -> list[tuple[Channel | None, AnySlot]]:
        """The slots ``port`` reads, as ``Binding`` takes them."""
        spec = as_spec(port.recv_spec)
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
        # An unconnected single-peer port reads what the host writes (its carry
        # stays zeros: the host cannot write one).
        carry = as_spec(port.recv_carry_spec)
        slot = self._group.input(_schema(spec, carry), path)
        self._inputs[port] = (slot, spec)
        self._slots[slot.name] = (slot, spec, carry)
        return [(None, slot)]

    # --- run and control

    def start(
        self,
        *,
        manual: bool,
        state: Mapping[str, Any] | None,
        inputs: Mapping[AnyPort, object] | None,
        workers: int | None,
    ) -> None:
        """Restore ``state``, write ``inputs`` and start the nodes, the ones not on
        a thread of their own on ``workers`` threads (by default as many as the
        CPUs).

        Raises as ``mujou.run`` does. Anything but ``NetworkError`` comes before any
        node started.
        """
        # Forget the runs that have ended, and keep this one (a refused start
        # discards it below).
        for session in lifetime.running(Session):
            session._settle()
        lifetime.add(self)
        try:
            periods = [checked_period(node) for node in self._nodes]
            dedicated = [checked_dedicated(node) for node in self._nodes]
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
                        slot.restore(tensors, time_ns, sent=True)
            runtime = _native.Runtime(self._nodes, self._names(), timer=self._clock)
            self._runtime = runtime
            for node in self._nodes:
                set_timer(node, self._view)
            # What this thread queued for the nodes (tensors made in __init__ or by
            # the restore) is done before their streams use it.
            self._synchronize_current_streams()
            runtime.start(
                manual=manual,
                periods=periods,
                workers=workers,
                dedicated=dedicated,
                **(restored or {}),
            )
        except _native.NetworkError:
            self._settle()  # a node raised while starting: this run is over
            raise
        except Exception:
            # Refused before any node started (e.g. the timer cannot be set back):
            # the network may run once the cause is gone.
            self._runtime = None
            lifetime.discard(self)
            for node in self._nodes:
                set_timer(node, None)
            raise

    def _names(self) -> list[str]:
        """The names of the nodes in the runtime's errors and logs: a node that is
        itself the root (at the path "") by its class."""
        return [
            path or type(node).__name__
            for path, node in zip(self._paths, self._nodes, strict=True)
        ]

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
        """Once the run has ended, stop keeping the session and stop the nodes'
        ``timer``, once.

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
            lifetime.discard(self)
            ended = self._ended_timer(runtime)
            for node in self._nodes:
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
                *(slot.read().timestamp.ns for slot, _, _ in self._slots.values()),
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
            items.append((slot, host_tensors(spec, value, where=self._labels[port])))
        return items

    def read(self, port: AnyPort) -> tuple[object, _native.Timestamp]:
        self.check()
        outbox, spec = self._outbox(port)
        message = outbox.read()
        return message_value(spec, message, _CPU), message.timestamp

    def subscribe(
        self, port: AnyPort, timeout: float
    ) -> Iterator[tuple[object, _native.Timestamp]]:
        _wait(timeout)  # checks it
        # The whole iteration counts from this call, on the real clock.
        deadline = time.monotonic() + timeout
        runtime = self._started()
        self.check()
        outbox, spec = self._outbox(port)
        # The value at the time of this call comes first.
        return self._follow(runtime, outbox, spec, outbox.read(), timeout, deadline)

    def _follow(
        self,
        runtime: _native.Runtime,
        outbox: _native.Slot,
        spec: Spec,
        last: _native.Message,
        timeout: float,
        deadline: float,
    ) -> Iterator[tuple[object, _native.Timestamp]]:
        self.check()
        yield message_value(spec, last, _CPU), last.timestamp
        while True:
            self.check()
            finished = runtime.is_finished
            if finished:
                # After the end nothing is written, so one last look without
                # waiting.
                message = outbox.wait_newer(last, 0.0)
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"subscribe did not end within {timeout} s; the run keeps "
                        "running"
                    )
                message = outbox.wait_newer(last, min(_POLL, remaining))
            if message is None:
                if finished:
                    self.check()  # the end may have come from an exception
                    return
                continue
            last = message
            yield message_value(spec, message, _CPU), message.timestamp

    def _writable(self, port: AnyPort) -> tuple[_native.InputSlot, Spec]:
        name = self.label(port)
        if port.recv_carry_spec is not None:
            raise ValueError(
                f"cannot write the port {name!r}: it receives carried data, which "
                "the host cannot write"
            )
        if port in self._inputs:
            return self._inputs[port]
        if port.recv_spec is None:
            reason = "receives no messages"
        elif port.channels:
            reason = "is connected (only unconnected ports can be written)"
        else:
            reason = "takes many peers (only single-peer ports can be written)"
        raise ValueError(f"cannot write the port {name!r}: it {reason}")

    def _outbox(self, port: AnyPort) -> tuple[_native.Slot, Spec]:
        name = self.label(port)
        outlet = self._outlets.get(port)
        if outlet is None:
            raise ValueError(f"cannot read the port {name!r}: it sends no messages")
        if port.send_carry_spec is not None:
            raise ValueError(
                f"cannot read the port {name!r}: it sends carried data, which the "
                "host cannot read"
            )
        return outlet.outbox, cast(Spec, port.send_spec)

    def label(self, port: AnyPort) -> str:
        """Name ``port`` in the host's errors: by its path, or as ``"Owner.attr"``
        when its owner is the root.

        Raises:
            ValueError: ``port`` is not a port of a node in this run.
        """
        name = self._labels.get(port)
        if name is None:
            raise ValueError(f"{port!r} is not a port of a node in this run")
        return name

    # --- state

    def state_dict(self, timeout: float) -> StateDict:
        runtime = self._started()
        finished = runtime.is_finished
        self.check()  # after is_finished: an end by an exception raises here
        if not finished:
            try:
                return runtime.capture(
                    lambda states: self._tree(
                        states, runtime.last_update_ns(), self._clock.state_dict()
                    ),
                    timeout,
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

    def _ended_tree(self, runtime: _native.Runtime) -> StateDict:
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
        # Check and copy everything before changing anything, so that an error
        # changes nothing.
        specs = {path: spec for path, (_, spec, _) in self._carries.items()}
        restores, carries, restored = read_tree(state, self._paths, self._slots, specs)
        # The copies were made on this thread's streams; the nodes use them on
        # their own streams, which do not wait for these.
        self._synchronize_current_streams()
        for slot, tensors, time_ns, sent in restores:
            slot.restore(tensors, time_ns, sent)
        for path, tensors in carries.items():
            binding, _, source = self._carries[path]
            # Made from the source's pools, as if received from it.
            binding.carry = source.new_carry(tensors)
        return restored

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
    ) -> StateDict:
        """Copy the state of the paused or ended network into a tree
        (``state_dict``).

        The nodes' streams are idle here (``Node._capture_state`` waited for them).
        """
        carries = {
            path: (cast(_native.Carry, binding.carry), spec)
            for path, (binding, spec, _) in self._carries.items()
        }
        tree = make_tree(
            self._paths, states, self._slots, carries, last_update_ns, timer
        )
        # The copies queue on this thread's current streams. Finish them before
        # the runtime resumes the nodes, which update in place.
        self._synchronize_current_streams()
        return tree


def _schema(spec: Spec, carry: Spec | None) -> _native.Schema:
    """The schema of messages of ``spec`` carrying ``carry`` (or nothing)."""
    return _native.Schema(fields(spec), [] if carry is None else fields(carry))


def _wait(timeout: float) -> float | None:
    """Return ``timeout`` as the runtime's ``join`` takes it (``None``: no limit).

    Raises:
        ValueError: ``timeout`` is negative or NaN.
    """
    if not timeout >= 0:
        raise ValueError(f"timeout must be a number of seconds >= 0, got {timeout}")
    return None if timeout == math.inf else timeout
