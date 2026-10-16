"""Components (nodes and networks), how they connect, and how a network is built.

A network is the structure of a run: nodes and nested networks as attributes, and
ports connected into channels. ``Network.build`` walks the tree once, checks it,
freezes it and returns its ``Graph``, which ``mujou.run`` runs
(``specs/network.md``).
"""

import threading
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Never, cast, final

from mujou._core import _native
from mujou._core.port import Channel, NextPort, Port, PrevPort, freeze, is_frozen

if TYPE_CHECKING:
    from mujou._core.node import Node


class Component:
    """The common base of ``Node`` and ``Network``: something with ports that
    connects.

    A component has no state of its own, so subclasses need not call
    ``super().__init__()``.
    """

    def connect(self, other: Component) -> Channel:
        """Connect this component's ``NextPort`` to ``other``'s ``PrevPort``.

        ``a.connect(b)`` is ``a.next.connect(b.prev)`` with the ports found by their
        declarations, so ``a.connect(b)`` and ``b.connect(a)`` are different
        connections. Connect other ports with ``Port.connect``.

        Raises:
            ValueError: This component has no ``NextPort`` or ``other`` has no
                ``PrevPort`` as a direct attribute, or has more than one.
            TypeError, RuntimeError: As ``Port.connect``.
        """
        return _side(self, NextPort).connect(_side(other, PrevPort))


def _ports(component: Component) -> list[tuple[str, Port[Any, Any, Any]]]:
    """Return ``(attribute name, port)`` for the direct attributes that are ports."""
    # Any: ports of every message type are handled alike.
    return [
        (name, cast(Port[Any, Any, Any], value))
        for name, value in vars(component).items()
        if isinstance(value, Port)
    ]


def _sides(
    component: Component, kind: type[PrevPort[Any, Any, Any] | NextPort[Any, Any, Any]]
) -> list[Port[Any, Any, Any]]:
    """Return the distinct direct attributes of ``component`` that are ``kind``.

    The same port under two names counts once.
    """
    found: list[Port[Any, Any, Any]] = []
    for _, port in _ports(component):
        # Ports compare by identity, so `in` finds the same port.
        if isinstance(port, kind) and port not in found:
            found.append(port)
    return found


def _side(
    component: Component, kind: type[PrevPort[Any, Any, Any] | NextPort[Any, Any, Any]]
) -> Port[Any, Any, Any]:
    """Return the only direct attribute of ``component`` that is a ``kind``."""
    found = _sides(component, kind)
    if len(found) != 1:
        raise ValueError(
            f"{type(component).__name__} must have exactly one {kind.__name__} as a "
            f"direct attribute to connect, found {len(found)}"
        )
    return found[0]


@final
class Graph:
    """The structure of a built network, which ``mujou.run`` runs.

    ``Network.build`` makes it; it cannot be created directly. It is a different
    object from the network it was built from: the network declares the
    structure, and the graph is what runs. A graph runs once.
    """

    # Any: ports of every message type are handled alike.
    _nodes: tuple[tuple[str, Node], ...]
    _ports: Mapping[Port[Any, Any, Any], tuple[int, str]]
    # Whether a run has started (or tried to start and failed in a node), and the
    # lock that makes the check and the start one step for concurrent callers.
    _ran: bool
    _lock: threading.Lock

    def __init__(self) -> Never:
        raise TypeError("a Graph is made by Network.build()")

    @property
    def nodes(self) -> tuple[tuple[str, Node], ...]:
        """``(path, node)`` for every node in the tree, depth first in attribute
        definition order.

        A path is the dotted attribute path from the root, e.g. ``"body.l1"``.
        """
        return self._nodes

    def port_path(self, port: Port[Any, Any, Any]) -> str:
        """Return the path of ``port`` through its owner, e.g. ``"body.l1.next"``.

        Raises:
            ValueError: ``port`` is not owned by a node of this network.
        """
        if port not in self._ports:
            raise ValueError(f"{port!r} is not a port of this network")
        return _path(self._nodes, self._ports[port])


def _make_graph(
    nodes: tuple[tuple[str, Node], ...],
    ports: Mapping[Port[Any, Any, Any], tuple[int, str]],
) -> Graph:
    graph = object.__new__(Graph)
    graph._nodes = nodes  # pyright: ignore[reportPrivateUsage]  # Graph's own module
    graph._ports = ports  # pyright: ignore[reportPrivateUsage]
    graph._ran = False  # pyright: ignore[reportPrivateUsage]
    graph._lock = threading.Lock()  # pyright: ignore[reportPrivateUsage]
    return graph


def ports(graph: Graph) -> Mapping[Port[Any, Any, Any], tuple[int, str]]:
    """For every port owned by a node of ``graph``, the index of the owner in
    ``graph.nodes`` and the attribute name of the port."""
    return graph._ports  # pyright: ignore[reportPrivateUsage]  # Graph's own module


class Network(Component):
    """A composite of nodes and other networks, declared as attributes.

    Assign nodes and networks as attributes, connect their ports, and expose the
    ports the outside should reach as attributes of the network. An exposed
    ``PrevPort`` / ``NextPort`` becomes the previous / next side of the network, so
    networks connect just like nodes. Only direct attributes are walked (not lists
    or dicts). A network is a build-time structure: its nodes still run
    independently and asynchronously.

    Declare ports as instance attributes (in ``__init__``), not class attributes,
    and do not use the names of the methods below for attributes.

    ``mujou.run(net)`` builds the network (``build``) and runs it; the ``Run`` it
    returns is how the host talks to the running nodes.

    Example::

        class Stack(Network):
            def __init__(self) -> None:
                self.l1 = Hidden(784, 256)
                self.l2 = Hidden(256, 256)
                self.l1.connect(self.l2)  # l1.next <-> l2.prev
                self.prev = self.l1.prev  # expose the previous side
                self.next = self.l2.next  # expose the next side


        class Classifier(Network):
            def __init__(self) -> None:
                self.body = Stack()
                self.head = Head()
                self.body.connect(self.head)  # body.next (l2.next) <-> head.prev


        net = Classifier()
        with mujou.run(net, inputs={net.prev: x0, net.teacher: y0}) as run:
            run.write({net.prev: x, net.teacher: y})  # switch together
            mujou.time.sleep(0.1)  # present the input for 0.1 s of global time
            prediction, t = run.read(net.head.next)
            state = run.state_dict()
    """

    _graph: Graph | None = None

    def build(self) -> Graph:
        """Check the structure of this network as the root, freeze it and return
        its graph.

        After ``build``, no port in the tree takes new connections. Calling it
        again returns the same graph. ``mujou.run`` calls it.

        Raises:
            RuntimeError: Part of the tree belongs to another built network (or
                this network is part of a built one).
            ValueError: The structure is invalid: the tree has no node; a component
                appears at two paths; a port is owned by two nodes (or under two
                names); a port is declared as a class attribute; a node holds a
                node or network as an attribute; a network exposes a port that no
                node inside it owns; a channel leads to a port outside the tree; or
                a component has more than one ``PrevPort`` or ``NextPort`` as
                direct attributes.
        """
        # Imported here: _node imports Component from this module.
        from mujou._core.node import Node

        if self._graph is not None:
            return self._graph

        nodes: list[tuple[str, Node]] = []
        ports: dict[Port[Any, Any, Any], tuple[int, str]] = {}
        # (attribute path, port, indices of the nodes in the exposing subtree)
        exposed: list[tuple[str, Port[Any, Any, Any], range]] = []
        seen: dict[int, str] = {}  # id(component) -> path

        def visit(component: Component, path: str) -> None:
            if id(component) in seen:
                raise ValueError(
                    f"{type(component).__name__} appears at both "
                    f"{_where(seen[id(component)])} and {_where(path)}"
                )
            seen[id(component)] = path
            _check_declarations(component, _where(path))
            if isinstance(component, Node):
                index = len(nodes)
                nodes.append((path, component))
                for name, value in vars(component).items():
                    if isinstance(value, Component):
                        raise ValueError(
                            f"node {_where(path)} holds a {type(value).__name__} at "
                            f"{name!r}; only networks hold nodes and networks"
                        )
                for name, port in _ports(component):
                    owner = (index, name)
                    if port in ports:
                        raise ValueError(
                            f"the port {_path(nodes, owner)!r} is already owned as "
                            f"{_path(nodes, ports[port])!r}"
                        )
                    ports[port] = owner
                return
            first = len(nodes)
            for name, value in vars(component).items():
                if isinstance(value, Component):
                    visit(value, _child(path, name))
            for name, port in _ports(component):
                exposed.append((_child(path, name), port, range(first, len(nodes))))

        visit(self, "")
        if not nodes:
            raise ValueError(
                "the network has no node (only direct attributes are walked, not "
                "lists or dicts)"
            )
        for path, port, subtree in exposed:
            if port not in ports or ports[port][0] not in subtree:
                raise ValueError(
                    f"the port exposed at {path!r} is not owned by a node inside the "
                    "network that exposes it (only direct attributes are walked)"
                )
        for port, owner in ports.items():
            if is_frozen(port):
                raise RuntimeError(
                    f"the port {_path(nodes, owner)!r} belongs to a network that is "
                    "already built"
                )
            for channel in port.channels:
                if channel.other(port) not in ports:
                    raise ValueError(
                        f"the port {_path(nodes, owner)!r} is connected to a port "
                        "outside the network (only direct attributes are walked, "
                        "not lists or dicts)"
                    )
        for port in ports:
            freeze(port)
        self._graph = _make_graph(tuple(nodes), MappingProxyType(ports))
        return self._graph


class Compose(Network):
    """A network of the given networks side by side, unconnected unless they were
    connected before.

    A network given by position is at the path of its position (``"0"``,
    ``"1"``, ...) and one given by keyword at its keyword, so the node ``l1`` of
    ``Compose(a, b)`` is ``"1.l1"`` in ``b`` and that of ``Compose(enc=a)`` is
    ``"enc.l1"``. ``mujou.run(a, b)`` runs ``Compose(a, b)``.

    Example::

        enc, dec = Encoder(), Decoder()
        enc.connect(dec)
        with mujou.run(Compose(enc=enc, dec=dec)) as run:
            state = run.state_dict()  # nodes "enc.l1", ..., "dec.l1", ...

    Raises:
        TypeError: A value is not a ``Network``.
        ValueError: A keyword starts with ``_``, is the name of an attribute of
            ``Network`` (such as ``build``), or is the path of a position.
    """

    def __init__(self, *networks: Network, **named: Network) -> None:
        children = {str(i): network for i, network in enumerate(networks)}
        for name, network in named.items():
            if name.startswith("_") or hasattr(Compose, name) or name in children:
                raise ValueError(
                    f"cannot compose a network as {name!r}: the name is private, "
                    "an attribute of Network, or a position"
                )
            children[name] = network
        for name, network in children.items():
            # The annotation is not checked at runtime; a node here would be walked
            # as a child, not composed.
            if not isinstance(network, Network):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    f"Compose takes networks, got {type(network).__name__} at {name!r}"
                )
            setattr(self, name, network)


def start_once[T](graph: Graph, start: Callable[[], T]) -> T:
    """Call ``start`` to run ``graph`` unless it has run, and mark it as run.

    A start refused before any node started (any exception but ``NetworkError``)
    does not count, so the graph may run once the cause is gone.

    Raises:
        RuntimeError: ``graph`` has already run.
    """
    with graph._lock:  # pyright: ignore[reportPrivateUsage]  # Graph's own module
        if graph._ran:  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError(
                "the network has already run; build a new instance to run again"
            )
        try:
            started = start()
        except _native.NetworkError:
            graph._ran = True  # pyright: ignore[reportPrivateUsage]
            raise
        graph._ran = True  # pyright: ignore[reportPrivateUsage]
        return started


def _path(nodes: Sequence[tuple[str, Node]], owner: tuple[int, str]) -> str:
    """Return the path of the port ``owner`` (node index, attribute name) names."""
    index, name = owner
    return f"{nodes[index][0]}.{name}"


def _child(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _where(path: str) -> str:
    return repr(path) if path else "the root"


def _check_declarations(component: Component, where: str) -> None:
    """Reject ports declared on the class and more than one side of a kind."""
    for klass in type(component).__mro__:
        for name, value in vars(klass).items():
            if isinstance(value, Port):
                raise ValueError(
                    f"{type(component).__name__} at {where} declares the port "
                    f"{name!r} on the class {klass.__name__}; declare ports as "
                    "instance attributes in __init__"
                )
    for kind in (PrevPort, NextPort):
        n = len(_sides(component, kind))
        if n > 1:
            raise ValueError(
                f"{type(component).__name__} at {where} has {n} {kind.__name__} "
                "attributes; at most one is allowed"
            )
