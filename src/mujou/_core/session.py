"""The run of a built network: slots for its ports, the runtime of its nodes, and the
host's reads and writes (``specs/network.md``, ``specs/execution-model.md``).

``mujou.run`` creates one ``Session`` per run, on the run's timer. It lays out the
slots of every port on the run's clock (the reading side of that timer), binds
each port to them (``binding.py``), and starts the runtime on the same clock. The
state tree is ``state.py``'s, and the registry of running runs ``lifetime.py``'s.
"""

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final, NamedTuple, cast, final

import torch

from mujou._core import _native, lifetime
from mujou._core.binding import AnyPort, AnySlot, Binding, as_spec
from mujou._core.codec import from_core, schema, to_core
from mujou._core.driver import Driver
from mujou._core.graph import Graph, ports
from mujou._core.node import Node, checked_dedicated, checked_period
from mujou._core.port import Channel, bind, initial, label
from mujou._core.spec import Spec
from mujou._core.state import SlotSpecs, StateDict, make_tree, read_tree

# Real seconds between the checks of `subscribe` for the end of the run.
_POLL: Final = 0.05
# Where the host receives values.
_CPU: Final = torch.device("cpu")


class NodeSpec(NamedTuple):
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


@final
class Session:
    """The slots and the runtime of one run of a built network, on one timer.

    ``start`` restores the state, writes the initial inputs and starts the nodes,
    once.
    """

    def __init__(self, graph: Graph, timer: _native.Timer) -> None:
        self._graph = graph
        self._paths = [path for path, _ in graph.nodes]
        self._nodes = [node for _, node in graph.nodes]
        # The one clock of the run, on its timer: every slot, the host's inputs
        # and the runtime use it, so that every time in the run is on the axis dt
        # is measured on, and none is later than the end. The nodes see it too.
        self._clock = _native.ReadOnlyTimer(timer)
        # One group for the whole network: a write switches together only within a
        # group.
        self._group = _native.InputGroup(clock=self._clock)
        self._outlets: dict[AnyPort, _native.Outlet] = {}
        self._inputs: dict[AnyPort, tuple[_native.InputSlot, Spec]] = {}
        # Every slot by name, in the order of the graph's ports, for the state tree.
        self._slots: dict[str, SlotSpecs] = {}
        # Every port of the run by how its errors name it.
        self._labels: dict[AnyPort, str] = {}

        owners = ports(graph)
        for port, (index, _) in owners.items():
            # The path alone ("next") would not say which node a root's port is
            # of. The slots take the name too, so that the core's errors use it.
            root = not graph.nodes[index][0]
            self._labels[port] = label(port) if root else graph.port_path(port)
        # Every outlet first: a port reads the channel slots of its peers' outlets.
        for port in owners:
            self._add_outlet(port)
        for port, (index, _) in owners.items():
            name = self._labels[port]
            sources = self._sources(port, name)
            owner = graph.nodes[index][1]
            bind(port, Binding(owner, name, port, self._outlets.get(port), sources))

        self._runtime: _native.Runtime | None = None

    def _add_outlet(self, port: AnyPort) -> None:
        spec = as_spec(port.send_spec)
        if spec is None:
            return
        outlet = _native.Outlet(
            schema(spec),
            len(port.channels),
            self._labels[port],
            clock=self._clock,
        )
        self._outlets[port] = outlet
        slots = [outlet.outbox, *(outlet.channel(k) for k in range(len(port.channels)))]
        for slot in slots:
            self._slots[slot.name] = (slot, spec)
        values = initial(port)
        if values is not None:
            # In place of the zeros, and like them without a time: the port has
            # not sent them. A restore later replaces them, as it does the zeros.
            for slot in slots:
                slot.restore(values, None)

    def _sources(
        self, port: AnyPort, name: str
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
        # An unconnected single-peer port reads what the host writes.
        slot = self._group.input(schema(spec), name)
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
        workers: int | None,
    ) -> None:
        """Restore ``state``, write ``inputs`` and start the nodes, the ones not on
        a thread of their own on ``workers`` threads (by default as many as the
        CPUs).

        Raises as ``mujou.run`` does. Anything but ``NetworkError`` comes before any
        node started.
        """
        # Kept until it ends (a refused start discards it below).
        lifetime.add(self)
        try:
            restored = None if state is None else self._restore(state)
            if inputs:
                # On the run's clock, which a restored run starts at the time of
                # the state.
                self._group.write(self._items(inputs))
            nodes = self._specs(restored)
            # What this thread queued for the nodes (tensors made in __init__ or by
            # the restore) is done before their streams use it.
            self._synchronize_current_streams()
            self._runtime = _native.start(
                nodes, self._clock, manual=manual, workers=workers
            )
        except BaseException:
            # Refused before any node started, or a node raised while starting, or
            # Ctrl-C (``_native.start`` raises once the nodes have stopped): this
            # run is over.
            lifetime.discard(self)
            raise

    def _specs(self, restored: Mapping[str, Any] | None) -> list[NodeSpec]:
        """The nodes as ``_native.start`` takes them, restoring ``restored``."""
        n = len(self._nodes)
        states: Sequence[object] = [None] * n
        last_update_ns: Sequence[int | None] = [None] * n
        if restored is not None:
            states, last_update_ns = restored["states"], restored["last_update_ns"]
        return [
            NodeSpec(
                Driver(node, self._clock),
                name,
                checked_period(node),
                checked_dedicated(node),
                ns,
                s,
            )
            for node, name, ns, s in zip(
                self._nodes, self._names(), last_update_ns, states, strict=True
            )
        ]

    def _names(self) -> list[str]:
        """The names of the nodes in the runtime's errors and logs: a node that is
        itself the root (at the path "") by its class."""
        return [
            path or type(node).__name__
            for path, node in zip(self._paths, self._nodes, strict=True)
        ]

    def indices(self, nodes: Sequence[Node]) -> list[int] | None:
        """The indices of ``nodes`` for ``step`` (``None``: every node).

        Raises:
            ValueError: A node is not in this run.
        """
        return [self._index(node) for node in nodes] if nodes else None

    def step(self, indices: list[int] | None) -> None:
        runtime = self._started()
        try:
            runtime.step(indices)
        finally:
            self._forget_if_ended(runtime)

    def ensure_running(self) -> None:
        """Return if the run has not stopped.

        Raises:
            NetworkError: A node raised; the run has stopped.
            RuntimeError: The run has stopped normally.
        """
        finished = self._started().finished
        self.check()  # after finished: an end by an exception raises here
        if finished:
            raise RuntimeError("the run has stopped")

    def check(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        try:
            runtime.check()
        finally:
            self._forget_if_ended(runtime)

    def shutdown(self, timeout: float) -> None:
        wait = _wait(timeout)
        runtime = self._runtime
        if runtime is None:
            return
        # Does nothing once the run has ended.
        runtime.stop()
        # On a timeout the request stands and the network goes on stopping.
        if not self._join(runtime, wait):
            raise TimeoutError(
                f"the run did not stop within {timeout} s; it goes on "
                "stopping, and join() waits for it"
            )

    def join(self, timeout: float) -> None:
        if not self._join(self._started(), _wait(timeout)):
            raise TimeoutError(
                f"the run did not stop within {timeout} s; it keeps running"
            )

    def _join(self, runtime: _native.Runtime, wait: float | None) -> bool:
        try:
            return runtime.join(wait)
        finally:
            self._forget_if_ended(runtime)

    def _forget_if_ended(self, runtime: _native.Runtime) -> None:
        """Stop keeping the session once the host has seen its run end: every
        call that can see the end (a step, a check, a join) comes here, so a
        run that ended, by an exception or not, is not kept until the exit."""
        if runtime.finished:
            lifetime.discard(self)

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
        self.ensure_running()
        return self._group.write(self._items(values))

    def _items(
        self, values: Mapping[Any, object]
    ) -> list[tuple[_native.InputSlot, list[torch.Tensor]]]:
        """Check ``values`` for the host's input slots and copy them to the host."""
        items: list[tuple[_native.InputSlot, list[torch.Tensor]]] = []
        for port, value in values.items():
            slot, spec = self._writable(port)
            items.append((slot, to_core(spec, value, self._labels[port])))
        return items

    def read(self, port: AnyPort) -> tuple[object, _native.Timestamp | None]:
        self.check()
        outbox, spec = self._outbox(port)
        message = outbox.read()
        return from_core(spec, message.values, _CPU), message.timestamp

    def subscribe(
        self, port: AnyPort, timeout: float
    ) -> Iterator[tuple[object, _native.Timestamp | None]]:
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
    ) -> Iterator[tuple[object, _native.Timestamp | None]]:
        self.check()
        yield from_core(spec, last.values, _CPU), last.timestamp
        while True:
            self.check()
            finished = runtime.finished
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
            yield from_core(spec, message.values, _CPU), message.timestamp

    def _writable(self, port: AnyPort) -> tuple[_native.InputSlot, Spec]:
        name = self.label(port)
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
        self.check()
        # While the run is paused for it, or after its end: the clock stands
        # still either way.
        return runtime.capture(
            lambda states: self._tree(
                states,
                runtime.last_update_ns(),
                {"time_ns": self._clock.now().ns, "scale": self._clock.time_scale()},
            ),
            timeout,
        )

    def _restore(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Check ``state``, restore the slots and return what ``Runtime.start``
        restores (the nodes' states and their last update times).

        Raises:
            ValueError: The tree does not match the network. Only the slots of
                this run, which is refused, may have changed.
        """
        restores, restored = read_tree(state, self._paths, self._slots)
        # The copies were made on this thread's streams; the nodes use them on
        # their own streams, which do not wait for these.
        self._synchronize_current_streams()
        for slot, tensors, time_ns in restores:
            slot.restore(tensors, time_ns)
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

        The nodes' streams are idle here (``Driver.capture`` waited for them).
        """
        tree = make_tree(self._paths, states, self._slots, last_update_ns, timer)
        # The copies queue on this thread's current streams. Finish them before
        # the runtime resumes the nodes, which update in place.
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
