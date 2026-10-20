"""Ports, the connection points of nodes and networks, and the channels between them.

A port has no direction: it holds the spec of the message it sends and the spec of
the message it receives, and of the data each carries. Two ports connect when their
specs are mutually reversed, and each connection is one ``Channel`` registered at
both ends (``specs/network.md``).
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Never,
    cast,
    final,
    overload,
    override,
)

from mujou._core.spec import DictSpec, MessageSpec, Spec, TensorSpec, host_tensors

if TYPE_CHECKING:
    import torch

    from mujou._core._native import Timestamp
    from mujou._core.binding import Binding

# The default of an omitted spec in the overloads below. Typed as a spec of Never so
# that pyright solves the message type of an omitted direction to Never, where an
# unsolved type variable would stay unknown. At runtime it is None.
_NO_SPEC: Final = cast(MessageSpec[Never], None)

# A port whose received carry another port passes on (``send_carry=port``): one
# peer, carrying C with what it receives.
type _CarrySource[C] = Port[Any, Any, Single, Any, C]


@final
class Single:
    """Type marker: the port connects to at most one peer."""


@final
class Multi:
    """Type marker: the port connects to any number of peers."""


# S: the messages a port sends. R: the messages it receives. M: the peer count,
# Single unless declared otherwise (Multi for NextPort). CS: the data carried with
# what it sends. CR: the data carried with what it receives.
class Port[S = Never, R = Never, M: (Single, Multi) = Single, CS = Never, CR = Never]:
    """A connection point that sends messages of type ``S`` and receives ``R``,
    carrying data of type ``CS`` with what it sends and ``CR`` with what it
    receives.

    Declare ports as attributes of a node (or expose a port of a child as an
    attribute of a network). ``send`` is the spec of what this port sends and
    ``recv`` the spec of what it receives; either may be omitted for a one-way port
    (omit it: type checkers reject ``None`` and infer ``Never`` for an omitted
    spec).
    ``multi=True`` lets the port take any number of peers (``Multi``) and
    ``multi=False`` at most one (``Single``). The default is one peer for ``Port``
    and ``PrevPort``, and many for ``NextPort``. ``multi`` fixes what ``recv``
    returns: the message for one peer, a dict keyed by ``Channel`` for many.

    Two ports connect when the ``send`` of each equals the ``recv`` of the other
    (an omitted spec matches an omitted spec). A port may stay unconnected.

    ``initial`` is what the port has sent before its first ``send``: what its
    peers receive and the host reads until then. It must match ``send`` exactly,
    and is copied when the port is made. Without it, that is the ``zeros()`` of
    ``send``. A restored run takes the values from the state instead.

    A message may carry data along a path, such as the target of a sample or
    its ID, in the same message as the value, so that a reader never pairs a
    value with the data of another sample. ``recv_carry`` is the spec of the
    data carried with what this port receives; ``carried`` reads it. A single
    peer only: a port with ``multi=True`` takes no ``recv_carry``.
    ``send_carry`` says what this port carries with what it sends: a spec, and
    ``send`` takes the data as ``carry``; or another port of the same node with
    a ``recv_carry``, and ``send`` passes on, without copying, the data of the
    message that port last received (``recv``). Until the first ``send``, and
    before the first ``recv`` of the port it passes on from, the carried data is
    the ``zeros()`` of its spec (in a restored run, the values under
    ``"carries"`` in the state). These zeros cannot be told from zeros that were
    sent: to tell them apart, carry a field that says whether the data is valid.

    Once assigned as an attribute of a node, the port is named
    ``"<node class>.<attribute>"`` (e.g. ``"Shift.next"``) in errors and its
    ``repr``; the first assignment names it.

    ``PrevPort`` and ``NextPort`` are ports with other defaults for the number of
    peers, for the previous and the next side of a node by convention.

    Example::

        class Sensor(Node):
            def __init__(self) -> None:
                super().__init__()
                self.out = Port(send=TensorSpec((3,)), initial=torch.ones(3))


        class Monitor(Node):
            def __init__(self) -> None:
                super().__init__()
                self.inp = Port(recv=TensorSpec((3,)))

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                x = self.inp.recv()  # torch.Tensor (one peer); ones until out sends


        sensor, monitor = Sensor(), Monitor()
        sensor.out.connect(monitor.inp)

    A node that passes the data its input carries on to its output::

        class Scale(Node):
            def __init__(self) -> None:
                super().__init__()
                self.prev = PrevPort(recv=TensorSpec((3,)), recv_carry=TensorSpec(()))
                self.next = NextPort(send=TensorSpec((3,)), send_carry=self.prev)

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                x = self.prev.recv()
                self.next.send(2 * x)  # carries what came with x
                sample_id = self.prev.carried()  # a copy, when the node needs it

    Raises:
        TypeError: ``send`` or ``recv`` is not a ``TensorSpec``, a ``DictSpec`` or
            ``None``; ``initial`` is given to a port without ``send``; or
            ``initial`` does not match ``send`` (as ``send`` raises).
            ``send_carry`` is not a spec, a port or ``None``, or is given to a port
            without ``send``; ``recv_carry`` is not a spec or ``None``, or is given
            to a port without ``recv`` or with ``multi=True``; or the port given
            as ``send_carry`` has no ``recv_carry``.
        ValueError: A shape, or the keys of a dict, in ``initial`` differ from
            ``send``.
    """

    @overload
    def __init__[S2, R2, CS2, CR2](
        self: Port[S2, R2, Single, CS2, CR2],
        send: MessageSpec[S2] = _NO_SPEC,
        recv: MessageSpec[R2] = _NO_SPEC,
        *,
        multi: Literal[False] = False,
        initial: S2 | None = None,
        send_carry: MessageSpec[CS2] | _CarrySource[CS2] = _NO_SPEC,
        recv_carry: MessageSpec[CR2] = _NO_SPEC,
    ) -> None: ...
    @overload
    def __init__[S2, R2, CS2](
        self: Port[S2, R2, Multi, CS2, Never],
        send: MessageSpec[S2] = _NO_SPEC,
        recv: MessageSpec[R2] = _NO_SPEC,
        *,
        multi: Literal[True],
        initial: S2 | None = None,
        send_carry: MessageSpec[CS2] | _CarrySource[CS2] = _NO_SPEC,
    ) -> None: ...
    def __init__(
        self,
        send: MessageSpec[S] | None = None,
        recv: MessageSpec[R] | None = None,
        *,
        multi: bool = False,
        initial: S | None = None,
        send_carry: MessageSpec[CS] | _CarrySource[CS] | None = None,
        recv_carry: MessageSpec[CR] | None = None,
    ) -> None:
        self._init(send, recv, multi, initial, send_carry, recv_carry)

    def _init(
        self,
        send: MessageSpec[S] | None,
        recv: MessageSpec[R] | None,
        multi: bool,
        initial: S | None,
        send_carry: MessageSpec[CS] | Port[Any, Any, Any, Any, CS] | None,
        recv_carry: MessageSpec[CR] | None,
    ) -> None:
        name = type(self).__name__
        for direction, spec in (
            ("send", send),
            ("recv", recv),
            ("recv_carry", recv_carry),
        ):
            if spec is not None and not isinstance(spec, TensorSpec | DictSpec):
                raise TypeError(
                    f"{name} {direction} must be a TensorSpec, a DictSpec or None, "
                    f"got {type(spec).__name__}"
                )
        if send_carry is not None and not isinstance(
            send_carry, TensorSpec | DictSpec | Port
        ):
            raise TypeError(
                f"{name} send_carry must be a TensorSpec, a DictSpec, a Port or "
                f"None, got {type(send_carry).__name__}"
            )
        if send_carry is not None and send is None:
            raise TypeError(f"{name} sends no messages, so it takes no send_carry")
        if recv_carry is not None and recv is None:
            raise TypeError(f"{name} receives no messages, so it takes no recv_carry")
        if recv_carry is not None and multi:
            raise TypeError(
                f"{name} takes many peers, so it takes no recv_carry (carried data "
                "comes from one peer)"
            )
        # The port whose last received carry this one passes on, or None.
        self._carry_source: Port[Any, Any, Any, Any, CS] | None = None
        self._send_carry_spec: MessageSpec[CS] | None = None
        if isinstance(send_carry, Port):
            if send_carry._recv_carry_spec is None:
                raise TypeError(
                    f"{name} send_carry: {send_carry._label()} has no recv_carry, so "
                    "there is nothing to pass on"
                )
            self._carry_source = send_carry
        else:
            self._send_carry_spec = cast(MessageSpec[CS] | None, send_carry)
        self._recv_carry_spec = recv_carry
        # The tensors of the initial value in field order, copied to the host now
        # so that changing the value later changes nothing.
        self._initial: list[torch.Tensor] | None = None
        if initial is not None:
            if send is None:
                raise TypeError(f"{name} sends no messages, so it takes no initial")
            self._initial = host_tensors(
                cast(Spec, send), initial, where=f"{name} initial", copy=True
            )
        self._send_spec = send
        self._recv_spec = recv
        self._multi = multi
        self._channels: list[Channel] = []
        # "<owner class>.<attribute>", set when the port is first assigned as an
        # attribute of a node; names the port in errors and its repr.
        self._name: str | None = None
        # Set by Network.build; a frozen port takes no more connections.
        self._frozen = False
        # Set by mujou.run: the slots recv and send use.
        self._binding: Binding | None = None

    @property
    def send_spec(self) -> MessageSpec[S] | None:
        """The spec of the messages this port sends, or ``None``."""
        return self._send_spec

    @property
    def recv_spec(self) -> MessageSpec[R] | None:
        """The spec of the messages this port receives, or ``None``."""
        return self._recv_spec

    @property
    def send_carry_spec(self) -> MessageSpec[CS] | None:
        """The spec of the data carried with what this port sends, or ``None``
        (for a port that passes it on, the ``recv_carry_spec`` of
        ``carry_source``)."""
        if self._carry_source is not None:
            return self._carry_source._recv_carry_spec
        return self._send_carry_spec

    @property
    def recv_carry_spec(self) -> MessageSpec[CR] | None:
        """The spec of the data carried with what this port receives, or
        ``None``."""
        return self._recv_carry_spec

    @property
    def carry_source(self) -> Port[Any, Any, Any, Any, CS] | None:
        """The port whose received carry this port passes on (its
        ``send_carry``), or ``None``."""
        return self._carry_source

    @property
    def multi(self) -> bool:
        """Whether this port takes any number of peers (else at most one)."""
        return self._multi

    @property
    def channels(self) -> tuple[Channel, ...]:
        """The channels connected to this port, in the order they were connected."""
        return tuple(self._channels)

    # Any: the peer may carry any message types as long as they match at runtime.
    def connect(self, other: Port[Any, Any, Any, Any, Any]) -> Channel:
        """Connect this port to ``other`` and return the new channel.

        The channel is registered at both ports, so each can send to and receive
        from the other. Connecting ``a`` to ``b`` and ``b`` to ``a`` are two
        different channels.

        Raises:
            TypeError: The specs are not mutually reversed (``self.send_spec !=
                other.recv_spec`` or ``self.recv_spec != other.send_spec``, and
                likewise for ``send_carry_spec`` and ``recv_carry_spec``), or
                neither direction carries a message.
            ValueError: ``other`` is this port, or a single-peer port is already
                connected.
            RuntimeError: Either port belongs to a tree that is already built (by
                ``Network.build`` or ``mujou.run``).
        """
        if self._frozen or other._frozen:
            raise RuntimeError(
                "cannot connect a port of a tree that is already built (by "
                "Network.build() or mujou.run())"
            )
        if other is self:
            raise ValueError(f"cannot connect a port to itself: {self._label()}")
        # dict.fromkeys: ports of two instances of one class can read the same.
        mismatches = dict.fromkeys(
            [
                f"{a._label()} sends {_or_nothing(a._send_spec)}, but {b._label()} "
                f"receives {_or_nothing(b._recv_spec)}"
                for a, b in ((self, other), (other, self))
                if a._send_spec != b._recv_spec
            ]
            + [
                f"{a._label()} carries {_or_nothing(a.send_carry_spec)} with what it "
                f"sends, but {b._label()} expects {_or_nothing(b._recv_carry_spec)}"
                for a, b in ((self, other), (other, self))
                if a.send_carry_spec != b._recv_carry_spec
            ]
        )
        if mismatches:
            raise TypeError("port specs do not match: " + "; ".join(mismatches))
        if self._send_spec is None and self._recv_spec is None:
            raise TypeError(
                f"neither {self._label()} nor {other._label()} has a spec, so nothing "
                "would flow"
            )
        for port in (self, other):
            if not port._multi and port._channels:
                raise ValueError(
                    f"{port._label()} takes one peer and is already connected; "
                    "declare it with multi=True to take more"
                )
        channel = _make_channel(self, other)
        self._channels.append(channel)
        other._channels.append(channel)
        return channel

    # The self-typed overloads make the return type depend on the peer count: one
    # peer gives the message, many give a dict keyed by channel in connection order.
    @overload
    def recv[R2](self: Port[Any, R2, Single, Any, Any]) -> R2: ...
    @overload
    def recv[R2](self: Port[Any, R2, Multi, Any, Any]) -> dict[Channel, R2]: ...
    def recv(self) -> Any:  # Any: the overloads above give the precise types
        """Return the latest message received, without waiting for a new one.

        A single-peer port returns the message itself. A multi-peer port returns
        ``{channel: message}``, one entry per channel in connection order (``{}``
        while unconnected).

        The message comes from the peer's ``send`` or ``send_to``; an unconnected
        single-peer port reads what the host writes with ``Run.write`` instead.
        Until that writer first writes, the message (or that channel's entry) is
        the initial value: the peer port's ``initial``, or else the recv spec's
        ``zeros()``. For a port the host writes, it is the value in ``inputs`` of
        ``mujou.run``, or else the ``zeros()``. A restored run takes it from
        ``state``. ``has_received`` tells whether the writer has written yet.

        Every call copies into new tensors on the owning node's device, without
        gradients, so changing them in place changes nothing else. The data the
        message carries is not copied: ``carried`` reads it, and a port with this
        one as its ``send_carry`` passes it on.

        Call it only from the owning node's hooks while the network runs.

        Raises:
            RuntimeError: The network is not running, or the caller is not a hook
                of the owning node (only the owner reads its port; ``state_dict``
                runs on another thread).
            TypeError: The port receives no messages.
        """
        return self._bound().recv(with_time=False)

    @overload
    def recv_with_time[R2](
        self: Port[Any, R2, Single, Any, Any],
    ) -> tuple[R2, Timestamp]: ...
    @overload
    def recv_with_time[R2](
        self: Port[Any, R2, Multi, Any, Any],
    ) -> dict[Channel, tuple[R2, Timestamp]]: ...
    def recv_with_time(self) -> Any:  # Any: as recv
        """Like ``recv``, but each message comes as ``(message, time)``.

        ``time`` is the ``Timestamp`` on the run's timer at which the message was
        written. Raises as ``recv`` does.
        """
        return self._bound().recv(with_time=True)

    @overload
    def has_received(self: Port[Any, Any, Single, Any, Any]) -> bool: ...
    @overload
    def has_received(self: Port[Any, Any, Multi, Any, Any]) -> dict[Channel, bool]: ...
    def has_received(self) -> Any:  # Any: as recv
        """Whether ``recv`` returns a message that its writer sent, rather than the
        initial value.

        The writer is the peer (its ``send`` or ``send_to``), or the host for an
        unconnected single-peer port (``inputs`` of ``mujou.run`` or
        ``Run.write``). Until the writer first writes, ``recv`` returns the
        initial value and this returns ``False``. A restored run keeps what was
        true when the state was taken.

        A single-peer port returns one ``bool``. A multi-peer port returns
        ``{channel: bool}``, keyed as ``recv`` is (``{}`` while unconnected).

        Call it only from the owning node's hooks while the network runs. Raises as
        ``recv`` does.
        """
        return self._bound().has_received()

    def send(self, msg: S, carry: CS | None = None) -> None:
        """Send ``msg`` to every peer and to what the host reads (``Run.read``).

        ``msg`` must match the send spec exactly; dtypes are never converted. The
        values are detached and copied, so changing ``msg`` afterwards changes
        nothing that was sent. A CUDA tensor is copied to the host on the node's
        stream. It never waits for the peers to read.

        A port whose ``send_carry`` is a spec takes the carried data as ``carry``,
        checked and copied as ``msg`` is. A port whose ``send_carry`` is another
        port takes no ``carry``: the message carries the data of the message
        that port last received, shared and not copied. Type checkers accept
        both mistakes (``carry`` given to a port that passes data on, or left
        out on a port that carries a spec); only the call rejects them.

        Call it only from the owning node's hooks while the network runs.

        Raises:
            RuntimeError: The network is not running, or the caller is not a hook
                of the owning node (only the owner writes its port).
            TypeError: The port sends no messages, a value is not a dense
                ``torch.Tensor`` of the spec's dtype, or a ``DictSpec`` message is
                not a dict; ``carry`` is missing for a port that carries a spec,
                or given to a port that carries nothing or passes data on.
            ValueError: A shape, or the keys of a dict message, differ from the spec.
        """
        self._bound().send(msg, carry)

    def send_to(self, channel: Channel, msg: S, carry: CS | None = None) -> None:
        """Send ``msg`` only to the peer across ``channel``.

        ``channel`` is one of ``channels`` (or a key of a multi-peer ``recv``).
        Unlike ``send``, it leaves what the host reads (``Run.read``) unchanged:
        before the first ``send``, that is this port's ``initial`` (or the
        ``zeros()`` of its send spec).
        Carries, copies and raises as ``send`` does, and raises ``ValueError``
        when ``channel`` is not one of this port's.
        """
        self._bound().send_to(channel, msg, carry)

    @overload
    def carried[CR2](self: Port[Any, Any, Any, Any, CR2]) -> CR2: ...
    @overload
    def carried(self, key: str) -> torch.Tensor: ...
    def carried(self, key: str | None = None) -> Any:  # Any: as recv
        """Return the data carried with the message last received by ``recv`` or
        ``recv_with_time``.

        Before the first ``recv`` it is the data of the message the port would
        have received then, the ``zeros()`` of the ``recv_carry`` spec until its
        writer first writes. With ``key``, only that field of a ``DictSpec`` is
        returned, and only it is copied.

        Every call copies into new tensors on the owning node's device, as
        ``recv`` does. Call it only from the owning node's hooks while the network
        runs.

        Raises:
            RuntimeError: As ``recv`` does.
            TypeError: The port has no ``recv_carry``, or ``key`` is given for a
                ``TensorSpec``.
            KeyError: ``key`` is not a key of the ``DictSpec``.
        """
        return self._bound().carried(key)

    def _bound(self) -> Binding:
        if self._binding is None:
            raise RuntimeError(
                f"{self._label()} can only send and receive in its owning node's "
                "hooks while the network runs (not in __init__)"
            )
        return self._binding

    def _label(self) -> str:
        """Name this port in an error: by its owner and attribute (e.g.
        ``"Shift.next"``) once assigned to a node, else by its repr."""
        return self._name if self._name is not None else repr(self)

    @override
    def __repr__(self) -> str:
        name = "" if self._name is None else f"{self._name}, "
        carry = ""
        if self._carry_source is not None:
            carry += f", send_carry={self._carry_source._label()}"
        elif self._send_carry_spec is not None:
            carry += f", send_carry={self._send_carry_spec!r}"
        if self._recv_carry_spec is not None:
            carry += f", recv_carry={self._recv_carry_spec!r}"
        return (
            f"{type(self).__name__}({name}send={self._send_spec!r}, "
            f"recv={self._recv_spec!r}, multi={self._multi}{carry})"
        )


class PrevPort[
    S = Never,
    R = Never,
    M: (Single, Multi) = Single,
    CS = Never,
    CR = Never,
](Port[S, R, M, CS, CR]):
    """A port for the previous side of a node or network: ``b.prev`` in
    ``a.next.connect(b.prev)``.

    It differs from ``Port`` only in its default: one peer, as ``Port``; pass
    ``multi=True`` to merge several. It is usually named ``prev``.

    Example::

        self.prev = PrevPort(send=TensorSpec((32, 784)), recv=TensorSpec((32, 784)))
        x = self.prev.recv()  # torch.Tensor
    """

    @overload
    def __init__[S2, R2, CS2, CR2](
        self: PrevPort[S2, R2, Single, CS2, CR2],
        send: MessageSpec[S2] = _NO_SPEC,
        recv: MessageSpec[R2] = _NO_SPEC,
        *,
        multi: Literal[False] = False,
        initial: S2 | None = None,
        send_carry: MessageSpec[CS2] | _CarrySource[CS2] = _NO_SPEC,
        recv_carry: MessageSpec[CR2] = _NO_SPEC,
    ) -> None: ...
    @overload
    def __init__[S2, R2, CS2](
        self: PrevPort[S2, R2, Multi, CS2, Never],
        send: MessageSpec[S2] = _NO_SPEC,
        recv: MessageSpec[R2] = _NO_SPEC,
        *,
        multi: Literal[True],
        initial: S2 | None = None,
        send_carry: MessageSpec[CS2] | _CarrySource[CS2] = _NO_SPEC,
    ) -> None: ...
    def __init__(
        self,
        send: MessageSpec[S] | None = None,
        recv: MessageSpec[R] | None = None,
        *,
        multi: bool = False,
        initial: S | None = None,
        send_carry: MessageSpec[CS] | _CarrySource[CS] | None = None,
        recv_carry: MessageSpec[CR] | None = None,
    ) -> None:
        self._init(send, recv, multi, initial, send_carry, recv_carry)


class NextPort[
    S = Never,
    R = Never,
    M: (Single, Multi) = Multi,
    CS = Never,
    CR = Never,
](Port[S, R, M, CS, CR]):
    """A port for the next side of a node or network: ``a.next`` in
    ``a.next.connect(b.prev)``.

    It differs from ``Port`` only in its default: any number of peers, since
    branches usually start here; pass ``multi=False`` for one. It is usually named
    ``next``.

    Example::

        self.next = NextPort(send=TensorSpec((32, 256)), recv=TensorSpec((32, 256)))
        feedback = self.next.recv()  # dict[Channel, torch.Tensor]
        self.next.send(h)  # to every peer
    """

    @overload
    def __init__[S2, R2, CS2](
        self: NextPort[S2, R2, Multi, CS2, Never],
        send: MessageSpec[S2] = _NO_SPEC,
        recv: MessageSpec[R2] = _NO_SPEC,
        *,
        multi: Literal[True] = True,
        initial: S2 | None = None,
        send_carry: MessageSpec[CS2] | _CarrySource[CS2] = _NO_SPEC,
    ) -> None: ...
    @overload
    def __init__[S2, R2, CS2, CR2](
        self: NextPort[S2, R2, Single, CS2, CR2],
        send: MessageSpec[S2] = _NO_SPEC,
        recv: MessageSpec[R2] = _NO_SPEC,
        *,
        multi: Literal[False],
        initial: S2 | None = None,
        send_carry: MessageSpec[CS2] | _CarrySource[CS2] = _NO_SPEC,
        recv_carry: MessageSpec[CR2] = _NO_SPEC,
    ) -> None: ...
    def __init__(
        self,
        send: MessageSpec[S] | None = None,
        recv: MessageSpec[R] | None = None,
        *,
        multi: bool = True,
        initial: S | None = None,
        send_carry: MessageSpec[CS] | _CarrySource[CS] | None = None,
        recv_carry: MessageSpec[CR] | None = None,
    ) -> None:
        self._init(send, recv, multi, initial, send_carry, recv_carry)


@final
class Channel:
    """One connection between two ports, made by ``Port.connect``.

    A channel is the handle of a peer: a multi-peer port's ``recv`` is keyed by
    channel and ``send_to`` takes one. Channels compare by identity, so two
    connections between the same ports are two different keys.

    It cannot be created directly: a channel that its ports do not know would
    carry nothing.

    Example::

        ch = a.next.connect(b.prev)
        ch.other(a.next) is b.prev  # True
    """

    # Any: a channel joins ports of any message types (connect checks they match).
    _ends: tuple[Port[Any, Any, Any, Any, Any], Port[Any, Any, Any, Any, Any]]

    def __init__(self) -> Never:
        raise TypeError("a Channel is made by Port.connect()")

    @property
    def ends(
        self,
    ) -> tuple[Port[Any, Any, Any, Any, Any], Port[Any, Any, Any, Any, Any]]:
        """The two ports: the one ``connect`` was called on, then its argument."""
        return self._ends

    def other(
        self, port: Port[Any, Any, Any, Any, Any]
    ) -> Port[Any, Any, Any, Any, Any]:
        """Return the end of this channel that is not ``port``.

        Raises:
            ValueError: ``port`` is not an end of this channel.
        """
        a, b = self._ends
        if port is a:
            return b
        if port is b:
            return a
        raise ValueError(f"{port!r} is not an end of this channel")

    @override
    def __repr__(self) -> str:
        a, b = self._ends
        return f"Channel({a!r} <-> {b!r})"


def _make_channel(
    a: Port[Any, Any, Any, Any, Any], b: Port[Any, Any, Any, Any, Any]
) -> Channel:
    channel = object.__new__(Channel)
    channel._ends = (a, b)  # pyright: ignore[reportPrivateUsage]  # Channel's own module
    return channel


def assign_owner(
    port: Port[Any, Any, Any, Any, Any], owner: str, attribute: str
) -> None:
    """Name ``port`` as ``"<owner>.<attribute>"`` unless it has a name already
    (a node calls this as the port is assigned; the first assignment wins, so a
    second name does not rename the port, and networks never do)."""
    if port._name is None:  # pyright: ignore[reportPrivateUsage]  # Port's own module
        port._name = f"{owner}.{attribute}"  # pyright: ignore[reportPrivateUsage]


def _or_nothing(spec: object) -> str:
    """``repr(spec)``, or ``"nothing"`` for a direction without a spec."""
    return "nothing" if spec is None else repr(spec)


def label(port: Port[Any, Any, Any, Any, Any]) -> str:
    """Name ``port`` in an error by its owner and attribute (``"Shift.next"``)."""
    return port._label()  # pyright: ignore[reportPrivateUsage]  # Port's own module


def initial(port: Port[Any, Any, Any, Any, Any]) -> list[torch.Tensor] | None:
    """The tensors of ``port``'s initial value in field order, or ``None`` (the
    run's ``Session`` writes them in place of the zeros)."""
    return port._initial  # pyright: ignore[reportPrivateUsage]  # Port's own module


def bind(port: Port[Any, Any, Any, Any, Any], binding: Binding) -> None:
    """Bind ``port`` to its slots (the run's ``Session`` calls this)."""
    port._binding = binding  # pyright: ignore[reportPrivateUsage]  # Port's own module


# Any (here and below): freezing does not depend on the message types.
def freeze(port: Port[Any, Any, Any, Any, Any]) -> None:
    """Make ``port`` reject further connections (``Network.build`` calls this)."""
    port._frozen = True  # pyright: ignore[reportPrivateUsage]  # Port's own module


def thaw(port: Port[Any, Any, Any, Any, Any]) -> None:
    """Let ``port`` take connections again (``mujou.run`` undoing a build whose run
    was refused)."""
    port._frozen = False  # pyright: ignore[reportPrivateUsage]  # Port's own module


def is_frozen(port: Port[Any, Any, Any, Any, Any]) -> bool:
    """Whether ``port`` belongs to a built network."""
    return port._frozen  # pyright: ignore[reportPrivateUsage]  # Port's own module
