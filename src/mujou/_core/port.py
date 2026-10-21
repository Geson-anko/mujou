"""Ports, the connection points of nodes, the channels between them, and the
``Wire`` through which a port of a running node reads and writes
(``specs/network.md``).

A port declares the spec of the message it sends and of the one it receives, and
its connections: a dict from each peer to the ``Channel`` between them. A port
holds nothing of a run: the driver of the running node (``driver.py``) holds the
wire of each of its ports, and a port finds its own there.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Never, Protocol, cast, final, override

import torch

from mujou._core import _native
from mujou._core.codec import from_core, to_core
from mujou._core.spec import MessageSpec

# Any (here and below): ports and messages of every type are handled alike.
type AnyPort = Port[Any, Any] | MultiPort[Any, Any]
type AnySlot = _native.Slot | _native.InputSlot


class _Port[S, R]:
    """What ``Port`` and ``MultiPort`` share: the specs, the initial value and
    the connections."""

    def __init__(
        self,
        *,
        send: MessageSpec[S] | None = None,
        recv: MessageSpec[R] | None = None,
        initial: S | None = None,
    ) -> None:
        self._send_spec = send
        self._recv_spec = recv
        self._initial = initial
        # Each peer and the channel to it, in the order they were connected.
        self._peers: dict[AnyPort, Channel] = {}

    @property
    def send_spec(self) -> MessageSpec[S] | None:
        """The spec of the messages this port sends, or ``None``."""
        return self._send_spec

    @property
    def recv_spec(self) -> MessageSpec[R] | None:
        """The spec of the messages this port receives, or ``None``."""
        return self._recv_spec

    @property
    def initial(self) -> S | None:
        """What the port has sent before its first ``send``, or ``None`` for the
        ``zeros()`` of its send spec."""
        return self._initial

    @property
    def channels(self) -> tuple[Channel, ...]:
        """The channels of this port, in the order they were connected."""
        return tuple(self._peers.values())

    def connect(self, other: Port[R, S] | MultiPort[R, S]) -> Channel:
        """Connect this port to ``other`` and return the channel between them.

        The channel is registered at both ports, so each sends to and receives
        from the other. Connecting the same two ports again, in either order,
        returns the same channel. A port may connect to itself.

        Raises:
            TypeError: The specs are not mutually reversed: ``self.send_spec !=
                other.recv_spec`` or ``self.recv_spec != other.send_spec``.
            ValueError: A ``Port`` that receives would get a second peer (use a
                ``MultiPort`` to receive from several).
        """
        channel = self._peers.get(other)
        if channel is not None:
            return channel
        # The ports have no names yet (a run names them by their paths): the
        # line of the call does.
        mismatches = [
            f"{a_name} sends {_or_nothing(a._send_spec)}, but {b_name} receives "
            f"{_or_nothing(b._recv_spec)}"
            for (a, a_name), (b, b_name) in (
                ((self, "this port"), (other, "the other")),
                ((other, "the other"), (self, "this port")),
            )
            if a._send_spec != b._recv_spec
        ]
        if mismatches:
            raise TypeError("port specs do not match: " + "; ".join(mismatches))
        for port, name in ((self, "this port"), (other, "the other")):
            if isinstance(port, Port) and port._recv_spec is not None and port._peers:
                raise ValueError(
                    f"{name} is a Port that receives and has a peer already; use a "
                    "MultiPort to receive from several"
                )
        me = cast(AnyPort, self)  # every port is a Port or a MultiPort
        channel = Channel(me, other)
        self._peers[other] = channel
        other._peers[me] = channel
        return channel

    def send(self, msg: S) -> None:
        """Send ``msg`` to every peer and to what the host reads (``Run.read``).

        ``msg`` must match the send spec exactly; dtypes are never converted. The
        values are detached and copied, so changing ``msg`` afterwards changes
        nothing that was sent. A CUDA tensor is copied to the host on the node's
        stream. It never waits for the peers to read.

        Raises:
            RuntimeError: The caller is not a hook of the node that owns the port,
                while it runs (or is its ``state_dict``).
            TypeError: A value is not a ``torch.Tensor``, or a ``DictSpec``
                message is not a dict with the spec's keys.
            ValueError: A shape or a dtype differs from the spec.
            BufferError: A tensor is not dense (``torch.strided``).
        """
        _wire(cast(AnyPort, self)).send(msg)  # as in connect

    @override
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(send={self._send_spec!r}, recv={self._recv_spec!r})"
        )


class Port[S = Never, R = Never](_Port[S, R]):
    """A connection point that sends messages of type ``S`` and receives at most
    one peer's messages of type ``R``.

    Declare ports as attributes of a node in ``__init__``. ``send`` is the spec of
    what the port sends and ``recv`` the spec of what it receives; omit one for a
    one-way port (its message type is then ``Never``). ``recv()`` returns one
    message, so a port that receives takes one peer; a port that only sends takes
    any number, and ``send`` reaches them all. To receive from several peers of
    one kind, use a ``MultiPort``; for inputs of different roles, use one port
    each.

    Two ports connect when the ``send`` of each equals the ``recv`` of the other.
    A port may stay unconnected: the host then writes what it receives
    (``Run.write``).

    ``initial`` is what the port has sent before its first ``send``: what its
    peers receive and the host reads until then. A run copies it when it
    starts. Without it, that is the ``zeros()`` of ``send``. A restored run
    takes the values from the state instead.

    A run names the port in its errors by the path of its node and its
    attribute (``"body.l1.next"``; ``"Shift.next"`` for a port of the node run
    as the root).

    Example::

        class Sensor(Node):
            def __init__(self) -> None:
                super().__init__()
                self.next = Port(send=TensorSpec((3,)), initial=torch.ones(3))


        class Monitor(Node):
            def __init__(self) -> None:
                super().__init__()
                self.prev = Port(recv=TensorSpec((3,)))

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                x = self.prev.recv()  # torch.Tensor; ones until the sensor sends


        sensor, monitor = Sensor(), Monitor()
        sensor.next.connect(monitor.prev)
    """

    def recv(self) -> R:
        """Return the latest message received, without waiting for a new one.

        The message comes from the peer's ``send``; an unconnected port reads
        what the host writes with ``Run.write`` instead. Until that writer first
        writes, it is the initial value: the peer port's ``initial``, or else the
        recv spec's ``zeros()``. For a port the host writes, it is the value in
        ``inputs`` of ``mujou.run``, or else the ``zeros()``. A restored run takes
        it from ``state``. ``recv_with_time`` tells whether the writer has
        written yet.

        Every call copies into new tensors on the owning node's device, without
        gradients, so changing them in place changes nothing else.

        Raises:
            RuntimeError: The caller is not a hook of the node that owns the port,
                while it runs (or is its ``state_dict``).
        """
        return cast(R, _one(self, with_time=False))

    def recv_with_time(self) -> tuple[R, _native.Timestamp | None]:
        """Like ``recv``, but return ``(message, time)``.

        ``time`` is the ``Timestamp`` on the run's timer at which the writer wrote
        the message, or ``None`` while the writer has not written yet (the
        message is then the initial value). The writer is the peer (its ``send``
        or ``send_to``), or the host for an unconnected port (``inputs`` of
        ``mujou.run`` or ``Run.write``). A restored run keeps the times of the
        state, ``None`` included. Raises as ``recv`` does.
        """
        # Quoted, here and below: an expression would be built at every call.
        return cast("tuple[R, _native.Timestamp | None]", _one(self, with_time=True))


class MultiPort[S = Never, R = Never](_Port[S, R]):
    """A connection point that sends messages of type ``S`` and receives
    messages of type ``R`` from any number of peers of one kind.

    It is declared and connected as a ``Port`` is, but ``recv()`` returns
    ``{channel: message}``, one entry per peer in connection order, and
    ``send_to`` sends to one peer. Use it to merge or to branch peers that play
    the same role; for inputs of different roles (a prediction and an error,
    say), use one ``Port`` each. The host does not write to a ``MultiPort``.

    Example::

        class Sum(Node):
            def __init__(self) -> None:
                super().__init__()
                self.prev = MultiPort(recv=TensorSpec((3,)))
                self.next = Port(send=TensorSpec((3,)))

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                self.next.send(sum(self.prev.recv().values(), torch.zeros(3)))
    """

    def recv(self) -> dict[Channel, R]:
        """Return ``{channel: message}``, the latest message of each peer in
        connection order (``{}`` while unconnected), as ``Port.recv`` returns the
        one message of its peer.

        Raises:
            RuntimeError: As ``Port.recv`` raises.
        """
        return cast("dict[Channel, R]", _many(self, with_time=False))

    def recv_with_time(self) -> dict[Channel, tuple[R, _native.Timestamp | None]]:
        """Like ``recv``, but each message comes as ``(message, time)``, as
        ``Port.recv_with_time`` gives it."""
        return cast(
            "dict[Channel, tuple[R, _native.Timestamp | None]]",
            _many(self, with_time=True),
        )

    def send_to(self, channel: Channel, msg: S) -> None:
        """Send ``msg`` only to the peer across ``channel``, one of ``channels``
        (a key of ``recv``).

        Unlike ``send``, it leaves what the host reads (``Run.read``) unchanged:
        before the first ``send``, that is this port's ``initial`` (or the
        ``zeros()`` of its send spec). Copies and raises as ``send`` does.
        """
        _wire(self).send_to(channel, msg)


@final
class Channel:
    """One connection between two ports, which ``connect`` makes.

    A channel is the handle of a peer: a ``MultiPort``'s ``recv`` is keyed by
    channel, and ``send_to`` takes one. A channel made otherwise connects
    nothing.

    Example::

        ch = a.next.connect(b.prev)
        ch.other(a.next) is b.prev  # True
    """

    def __init__(self, a: AnyPort, b: AnyPort) -> None:
        self._ends = (a, b)

    @property
    def ends(self) -> tuple[AnyPort, AnyPort]:
        """The two ports: the one ``connect`` was called on, then its argument."""
        return self._ends

    def other(self, port: AnyPort) -> AnyPort:
        """Return the end of this channel that is not ``port`` (``port`` itself
        for a port connected to itself)."""
        a, b = self._ends
        return b if port is a else a

    @override
    def __repr__(self) -> str:
        a, b = self._ends
        return f"Channel({a!r} <-> {b!r})"


@final
@dataclass(frozen=True, slots=True)
class Wire:
    """What one port of a running node reads and writes: the slots that the run
    laid out for it (``wiring.py``). The node's driver holds it."""

    # How errors name the port.
    label: str
    # Where the received messages arrive: the owning node's device.
    device: torch.device
    send_spec: MessageSpec[Any] | None
    recv_spec: MessageSpec[Any] | None
    # Where send writes: None for a port that does not send.
    outlet: _native.Outlet | None
    # What the port reads, (channel, slot) in connection order: an unconnected
    # Port that receives reads the slot the host writes, under None.
    sources: tuple[tuple[Channel | None, AnySlot], ...]
    # The index of each channel among the outlet's channel slots, for send_to.
    targets: Mapping[Channel, int]

    def read(self, slot: AnySlot, with_time: bool) -> object:
        message = slot.read()
        spec = cast("MessageSpec[Any]", self.recv_spec)  # a source has a recv spec
        value = from_core(spec, message.values, self.device)
        return (value, message.timestamp) if with_time else value

    def send(self, msg: object) -> None:
        self._outlet().send(self._tensors(msg))

    def send_to(self, channel: Channel, msg: object) -> None:
        self._outlet().send_to(self.targets[channel], self._tensors(msg))

    def _outlet(self) -> _native.Outlet:
        return cast(_native.Outlet, self.outlet)  # Never is the type of what it sends

    def _tensors(self, msg: object) -> list[torch.Tensor]:
        """The tensors of ``msg`` in host memory, for the outlet to copy."""
        return to_core(cast("MessageSpec[Any]", self.send_spec), msg, self.label)


class Running(Protocol):
    """What a hook sees of the node it runs as: the node's driver
    (``driver.py``)."""

    @property
    def node(self) -> object: ...

    @property
    def wires(self) -> Mapping[AnyPort, Wire]:
        """The wires of the node's ports: none in its ``state_dict``."""
        ...

    @property
    def clock(self) -> _native.ReadOnlyTimer: ...


def running() -> Running | None:
    """The driver of the node whose hook the calling thread runs, or ``None``
    outside the hooks."""
    return cast("Running | None", _native.current())


_OUTSIDE: Final = (
    "is used only in the hooks of the node that owns it, while it runs (a port not "
    "in state_dict); declare ports in __init__, and use run.read, run.write and "
    "run.timer from the host"
)


def outside(name: str) -> RuntimeError:
    """The error for a port or a node's timer, which ``name`` names, used
    outside the hooks of the node that owns it."""
    return RuntimeError(f"{name} {_OUTSIDE}")


def _wire(port: AnyPort) -> Wire:
    current = running()
    found = None if current is None else current.wires.get(port)
    if found is None:
        raise outside(repr(port))
    return found


def _one(port: AnyPort, with_time: bool) -> object:
    wire = _wire(port)
    ((_, slot),) = wire.sources
    return wire.read(slot, with_time)


def _many(port: AnyPort, with_time: bool) -> dict[Channel | None, object]:
    wire = _wire(port)
    return {channel: wire.read(slot, with_time) for channel, slot in wire.sources}


def _or_nothing(spec: object) -> str:
    """``repr(spec)``, or ``"nothing"`` for a direction without a spec."""
    return "nothing" if spec is None else repr(spec)
