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
from mujou._node import Node, end_thread, synchronize, thread_ident
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
        self._check_owner("receives on")
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

    def _check_owner(self, verb: str) -> None:
        # A port is read and written only by its owner: this keeps each slot to
        # one writer, and a node to the slots of its own ports (locality).
        if threading.get_ident() != thread_ident(self._owner):
            raise RuntimeError(
                f"only the node that owns the port {self._path!r} {verb} it, from its "
                "own hooks while the network runs (state_dict runs on another thread)"
            )

    def _sending(self) -> _core.Outlet:
        self._check_owner("sends on")
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
        # Makes the check and the start of a run one step for concurrent callers.
        self._run_lock = threading.Lock()
        # Restored by load_state_dict and passed to the runtime by run.
        self._states: list[object] | None = None
        self._last_update_ns: list[int] | None = None
        self._timer_state: dict[str, int | float] | None = None

    # --- run and control

    def run(
        self, *, manual: bool, pause_timeout: float, capture_state_on_exception: bool
    ) -> None:
        with self._run_lock:
            if self._runtime is not None:
                raise RuntimeError(
                    "the network has already run; build a new instance to run again"
                )
            runtime = _core.Runtime(self._nodes, self._paths)
            self._runtime = runtime
        for session in list(_RUNNING):
            session._settle()
        _RUNNING.add(self)
        try:
            runtime.start(
                manual=manual,
                states=self._states,
                last_update_ns=self._last_update_ns,
                timer_state=self._timer_state,
                pause_timeout=pause_timeout,
                on_exception_capture=(
                    _weakly(self._capture_on_exception)
                    if capture_state_on_exception
                    else None
                ),
            )
        except _core.NetworkError:
            raise  # a node raised while starting: this run is over
        except Exception:
            # Refused before any node started (e.g. an invalid argument): the
            # network may run once the cause is gone.
            self._runtime = None
            _RUNNING.discard(self)
            raise

    def step(self, order: Sequence[Node] | None) -> None:
        runtime = self._started()
        indices = None if order is None else [self._index(node) for node in order]
        runtime.step(indices)

    def check(self) -> None:
        if self._runtime is not None:
            self._settle()
            self._runtime.check()

    def shutdown(self) -> None:
        if self._runtime is not None:
            try:
                self._runtime.shutdown()
            finally:
                self._settle()

    def join(self) -> None:
        runtime = self._started()
        try:
            runtime.join()
        finally:
            self._settle()

    def _settle(self) -> None:
        """Once the run has ended, stop keeping the session and forget the nodes'
        threads."""
        if self._runtime is not None and self._runtime.is_finished:
            _RUNNING.discard(self)
            for node in self._nodes:
                end_thread(node)

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

    def write[P: AnyPort](self, values: Mapping[P, object]) -> None:
        self.check()
        items: list[tuple[_core.InputSlot, list[torch.Tensor]]] = []
        for port, value in values.items():
            slot, spec = self._writable(port)
            where = self._graph.port_path(port)
            items.append((slot, [_to_host(t) for t in split(spec, value, where=where)]))
        if self._runtime is None and self._timer_state is not None:
            # Restored but not running: the values belong to the restored time T,
            # not to the process's time now. Nobody reads yet, so restoring each
            # slot (after checking them all above) switches them together.
            time_ns = int(self._timer_state["time_ns"])
            for slot, tensors in items:
                slot.restore(tensors, time_ns)
            return
        self._group.write(items)

    def read(self, port: AnyPort) -> tuple[object, float]:
        self.check()
        outbox, spec = self._outbox(port)
        message = outbox.read()
        return _value(spec, message, _CPU), message.time_ns / 1e9

    def subscribe(self, port: AnyPort) -> Iterator[tuple[object, float]]:
        self._started()  # a subscription before run would wait for nothing
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

    def _capture_on_exception(
        self, states: dict[int, object], index: int, hook: str, error: BaseException
    ) -> object:
        """Build the state tree when a node has raised (the runtime calls this)."""
        runtime = cast(_core.Runtime, self._runtime)
        try:
            last_update_ns = runtime.last_update_ns()
        except RuntimeError:
            # A node raised while starting, before run had the runtime: no node
            # has updated, so each dt starts where the run started.
            last_update_ns = self._last_update_ns or [_core.timer.time_ns()] * len(
                self._nodes
            )
        tree = self._tree(states, last_update_ns)
        tree["exception"] = {
            "node": self._paths[index],
            "hook": hook,
            "error": "".join(traceback.format_exception(error)),
        }
        return tree

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
        # The copies were made on this thread's stream; the nodes use them on
        # their own streams, which do not wait for this one.
        _synchronize_cuda()
        self._last_update_ns = [cast(int, last_update_ns[path]) for path in self._paths]
        self._timer_state = {
            "time_ns": cast(int, timer["time_ns"]),
            "scale": timer["scale"],
        }

    def _tree(
        self, states: Mapping[int, object], last_update_ns: Sequence[int]
    ) -> dict[str, Any]:
        """Copy the state of the paused network into a tree (``state_dict``)."""
        # The copies below run on this (the control) thread's stream; wait for the
        # nodes' own streams first.
        for node in self._nodes:
            synchronize(node)
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


def _weakly[**P, T](
    method: Callable[P, T],
) -> Callable[P, T | None]:
    """Wrap a bound method so that it holds its object weakly.

    The runtime is not tracked by the garbage collector, so a strong reference
    from its callback back to the session would never be collected.
    """
    ref = weakref.WeakMethod(cast(Any, method))

    def call(*args: P.args, **kwargs: P.kwargs) -> T | None:
        bound = ref()
        return None if bound is None else cast(Callable[P, T], bound)(*args, **kwargs)

    return call


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _synchronize_cuda() -> None:
    """Wait for every CUDA device, if CUDA is in use (only while no node runs)."""
    if torch.cuda.is_initialized():
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)


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
    for session in list(_RUNNING):
        with contextlib.suppress(_core.NetworkError):
            session.shutdown()


# Before the interpreter joins the non-daemon threads: a thread iterating over
# subscribe ends only when its network does. threading._register_atexit is private,
# so fall back to atexit (after the join) where it is missing.
_register: Callable[[Callable[[], None]], object] = getattr(
    threading, "_register_atexit", atexit.register
)
_register(_shutdown_all)
