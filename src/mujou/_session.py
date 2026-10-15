"""The run of a built network: slots for its ports, the runtime of its nodes, and the
host's reads and writes (``docs/network.md``, ``docs/execution-model.md``).

``Network.build`` creates one ``Session``. It lays out the slots of every port,
binds each port to them (``Binding``), and later starts the runtime. Values cross
the boundary to the core as copies: a received message is a new tensor, and a sent
one is copied into the core.
"""

import atexit
import contextlib
import copy
import threading
import traceback
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast, final

import torch

from mujou import _core
from mujou._node import Node, thread_ident
from mujou._port import Channel, Port, bind
from mujou._spec import Spec, fields, join, split

if TYPE_CHECKING:
    from mujou._network import Graph

# Any (here and below): ports and messages of every type are handled alike.
type AnyPort = Port[Any, Any, Any]
type AnySlot = _core.Slot | _core.InputSlot

# Real seconds between the checks of `subscribe` for the end of the run.
_POLL: Final = 0.05


def _spec(spec: object) -> Spec | None:
    # Port accepts only TensorSpec, DictSpec or None (checked when it is made).
    return cast(Spec | None, spec)


def _to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Cut the gradient and bring ``tensor`` to host memory for the core to copy.

    A CUDA tensor is copied on the current stream (the node's own), and only that
    stream is synchronized.
    """
    return tensor.detach().cpu()


def _value(spec: Spec, message: _core.Message, device: torch.device) -> object:
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
        outlet: _core.Outlet | None,
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
        spec = self._recv_spec
        if spec is None:
            raise TypeError(f"the port {self._path!r} receives no messages")
        device = self._owner.device
        received: dict[Channel | None, object] = {}
        for channel, slot in self._sources:
            message = slot.read()
            value = _value(spec, message, device)
            received[channel] = (value, message.time_ns) if with_time else value
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

    def _sending(self) -> _core.Outlet:
        # Keeps the owning node the only writer of the port's slots.
        if threading.get_ident() != thread_ident(self._owner):
            raise RuntimeError(
                f"only the node that owns the port {self._path!r} sends on it, from "
                "its own hooks"
            )
        if self._outlet is None:
            raise TypeError(f"the port {self._path!r} sends no messages")
        return self._outlet

    def _tensors(self, msg: object) -> list[torch.Tensor]:
        spec = cast(Spec, self._send_spec)  # an outlet exists only with a send spec
        return [_to_host(t) for t in split(spec, msg, where=self._path)]


@final
class Session:
    """The slots and the runtime of one built network.

    Before ``run`` the host may ``write`` and ``load_state_dict``; ``run`` starts
    the nodes once.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        self._paths = [path for path, _ in graph.nodes]
        self._nodes = [node for _, node in graph.nodes]
        # One group for the whole network: a write switches together only within a
        # group.
        self._group = _core.InputGroup()
        self._outlets: dict[AnyPort, _core.Outlet] = {}
        self._inputs: dict[AnyPort, tuple[_core.InputSlot, Spec]] = {}
        # Every slot by name, in the order of graph.ports, for the state tree.
        self._slots: dict[str, tuple[AnySlot, Spec]] = {}

        for port in graph.ports:
            spec = _spec(port.send_spec)
            if spec is None:
                continue
            outlet = _core.Outlet(
                _core.Schema(fields(spec)), len(port.channels), graph.port_path(port)
            )
            self._outlets[port] = outlet
            self._slots[outlet.outbox.name] = (outlet.outbox, spec)
            for k in range(len(port.channels)):
                slot = outlet.channel(k)
                self._slots[slot.name] = (slot, spec)

        for port, (index, _) in graph.ports.items():
            path = graph.port_path(port)
            spec = _spec(port.recv_spec)
            sources: list[tuple[Channel | None, AnySlot]] = []
            if spec is not None and port.channels:
                for channel in port.channels:
                    # Each end writes its k-th channel slot for its k-th channel.
                    peer = channel.other(port)
                    k = peer.channels.index(channel)
                    sources.append((channel, self._outlets[peer].channel(k)))
            elif spec is not None and not port.multi:
                # An unconnected single-peer port reads what the host writes.
                slot = self._group.input(_core.Schema(fields(spec)), path)
                self._inputs[port] = (slot, spec)
                self._slots[slot.name] = (slot, spec)
                sources.append((None, slot))
            owner = graph.nodes[index][1]
            bind(port, Binding(owner, path, port, self._outlets.get(port), sources))

        self._runtime: _core.Runtime | None = None
        # Restored by load_state_dict and passed to the runtime by run.
        self._states: list[object] | None = None
        self._last_update_ns: list[int] | None = None
        self._timer_state: dict[str, int | float] | None = None

    # --- run and control

    def run(
        self, *, manual: bool, pause_timeout: float, capture_state_on_exception: bool
    ) -> None:
        if self._runtime is not None:
            raise RuntimeError(
                "the network has already run; build a new instance to run again"
            )
        runtime = _core.Runtime(self._nodes, self._paths)
        self._runtime = runtime
        _RUNNING.add(self)
        runtime.start(
            manual=manual,
            states=self._states,
            last_update_ns=self._last_update_ns,
            timer_state=self._timer_state,
            pause_timeout=pause_timeout,
            on_exception_capture=(
                _exception_capture(weakref.ref(self))
                if capture_state_on_exception
                else None
            ),
        )

    def step(self, order: Sequence[Node] | None) -> None:
        runtime = self._started()
        indices = None if order is None else [self._index(node) for node in order]
        runtime.step(indices)

    def check(self) -> None:
        if self._runtime is not None:
            self._runtime.check()

    def shutdown(self) -> None:
        if self._runtime is not None:
            self._runtime.shutdown()

    def join(self) -> None:
        self._started().join()

    def _started(self) -> _core.Runtime:
        if self._runtime is None:
            raise RuntimeError("the network is not running; call run() first")
        return self._runtime

    def _index(self, node: Node) -> int:
        for i, n in enumerate(self._nodes):
            if n is node:
                return i
        raise ValueError(f"{type(node).__name__} is not a node of this network")

    # --- host input and output

    def write(self, values: Mapping[AnyPort, object]) -> None:
        self.check()
        items: list[tuple[_core.InputSlot, list[torch.Tensor]]] = []
        for port, value in values.items():
            slot, spec = self._writable(port)
            where = self._graph.port_path(port)
            items.append((slot, [_to_host(t) for t in split(spec, value, where=where)]))
        self._group.write(items)

    def read(self, port: AnyPort) -> tuple[object, float]:
        self.check()
        outbox, spec = self._outbox(port)
        message = outbox.read()
        return _value(spec, message, _CPU), message.time_ns / 1e9

    def subscribe(self, port: AnyPort) -> Iterator[tuple[object, float]]:
        self.check()
        outbox, spec = self._outbox(port)
        # Only values written after this call are new.
        return self._follow(outbox, spec, outbox.read())

    def _follow(
        self, outbox: _core.Slot, spec: Spec, last: _core.Message
    ) -> Iterator[tuple[object, float]]:
        while True:
            self.check()
            finished = self._runtime is not None and self._runtime.is_finished
            # After the end nothing is written, so one last look without waiting.
            message = outbox.wait_newer(last, 0.0 if finished else _POLL)
            if message is None:
                if finished:
                    self.check()  # the end may have come from an exception
                    return
                continue
            last = message
            yield _value(spec, message, _CPU), message.time_ns / 1e9

    def _writable(self, port: AnyPort) -> tuple[_core.InputSlot, Spec]:
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

    def _outbox(self, port: AnyPort) -> tuple[_core.Slot, Spec]:
        path = self._graph.port_path(port)  # ValueError for a port outside the tree
        outlet = self._outlets.get(port)
        if outlet is None:
            raise ValueError(f"cannot read the port {path!r}: it sends no messages")
        return outlet.outbox, cast(Spec, port.send_spec)

    # --- state

    def state_dict(self) -> dict[str, Any]:
        runtime = self._started()
        return runtime.capture(
            lambda states: self._tree(dict(enumerate(states)), runtime.last_update_ns())
        )

    def exception_state_dict(self) -> dict[str, Any] | None:
        if self._runtime is None:
            return None
        return cast(dict[str, Any] | None, self._runtime.exception_state())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self._runtime is not None:
            raise RuntimeError("load_state_dict is only possible before run()")
        for key in ("nodes", "slots", "last_update_ns", "timer"):
            if key not in state:
                raise ValueError(f"the state tree has no {key!r}")
        nodes = cast(Mapping[str, object], state["nodes"])
        slots = cast(Mapping[str, tuple[object, int]], state["slots"])
        last_update_ns = cast(Mapping[str, int], state["last_update_ns"])
        _same_keys("nodes", nodes, self._paths)
        _same_keys("slots", slots, self._slots)
        _same_keys("last_update_ns", last_update_ns, self._paths)
        # Check every value before restoring any, so that an error changes nothing.
        restores: list[tuple[AnySlot, list[torch.Tensor], int]] = []
        for name, (slot, spec) in self._slots.items():
            value, time_ns = slots[name]
            tensors = [_to_host(t) for t in split(spec, value, where=name)]
            restores.append((slot, tensors, time_ns))
        for slot, tensors, time_ns in restores:
            slot.restore(tensors, time_ns)
        # Copies: the nodes update their state in place, which must not reach the
        # caller's tree (or another network loaded from it).
        self._states = [copy.deepcopy(nodes[path]) for path in self._paths]
        self._last_update_ns = [last_update_ns[path] for path in self._paths]
        self._timer_state = dict(cast(Mapping[str, int | float], state["timer"]))

    def _tree(
        self, states: Mapping[int, object], last_update_ns: Sequence[int]
    ) -> dict[str, Any]:
        """Copy the state of the paused network into a tree (``state_dict``)."""
        slots: dict[str, tuple[object, int]] = {}
        for name, (slot, spec) in self._slots.items():
            message = slot.read()
            slots[name] = (_value(spec, message, _CPU), message.time_ns)
        return {
            "nodes": {
                self._paths[i]: copy.deepcopy(state)
                for i, state in sorted(states.items())
            },
            "slots": slots,
            "last_update_ns": dict(zip(self._paths, last_update_ns, strict=True)),
            "timer": _core.timer.state_dict(),
        }


_CPU: Final = torch.device("cpu")


def _exception_capture(
    session: weakref.ref[Session],
) -> Callable[[dict[int, object], int, str, BaseException], object]:
    """Return the runtime's callback that builds the state tree on an exception.

    It holds the session weakly: the runtime is not tracked by the garbage collector,
    so a strong reference back to the session would never be collected.
    """

    def capture(
        states: dict[int, object], index: int, hook: str, error: BaseException
    ) -> object:
        s = session()
        if s is None or s._runtime is None:  # pyright: ignore[reportPrivateUsage]  # this module
            return None
        tree = s._tree(states, s._runtime.last_update_ns())  # pyright: ignore[reportPrivateUsage]  # this module
        tree["exception"] = {
            "node": s._paths[index],  # pyright: ignore[reportPrivateUsage]  # this module
            "hook": hook,
            "error": "".join(traceback.format_exception(error)),
        }
        return tree

    return capture


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


# The sessions that may be running, stopped at interpreter exit: the runtime's
# threads are not Python threads, and must not call into a finalizing interpreter.
_RUNNING: Final = weakref.WeakSet[Session]()


@atexit.register
def _shutdown_all() -> None:
    for session in list(_RUNNING):
        with contextlib.suppress(_core.NetworkError):
            session.shutdown()
