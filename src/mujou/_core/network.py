"""Components (nodes and networks), how they connect, and how a network is built.

A network is the structure of a run: nodes and nested networks as attributes, and
ports connected into channels. ``Network.build`` walks the tree once, checks it,
freezes it and returns its ``Graph``, which ``mujou.run`` runs
(``specs/network.md``).
"""

import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, Never, cast, final

from mujou._core import _native
from mujou._core.port import (
    Channel,
    NextPort,
    Port,
    PrevPort,
    freeze,
    is_frozen,
    thaw,
)


class Component:
    """The common base of ``Node`` and ``Network``: something with ports that
    connects.

    A component has no state of its own: its ``__init__`` is ``object``'s.
    """

    def connect(self, other: Node | Network) -> Channel:
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
        hint = ""
        if not found and isinstance(component, Network):
            side = "next" if kind is NextPort else "prev"
            hint = f"; expose one, e.g. self.{side} = self.<child>.{side}"
        raise ValueError(
            f"{type(component).__name__} must have exactly one {kind.__name__} as a "
            f"direct attribute to connect, found {len(found)}{hint}"
        )
    return found[0]


@final
class Graph:
    """The structure of a built network (or of a node run alone): what runs when
    ``mujou.run`` runs it.

    ``Network.build`` makes it, and ``mujou.run`` for a node given alone; it
    cannot be created directly, and ``mujou.run`` takes nodes and networks, not
    graphs. It is a different object from the network it was built from: the
    network declares the structure, and the graph is what runs. A graph runs
    once.
    """

    # Any: ports of every message type are handled alike.
    _nodes: tuple[tuple[str, Node], ...]
    _ports: Mapping[Port[Any, Any, Any], tuple[int, str]]
    # Every node and network of the tree, marked as built with _tree.
    _members: tuple[Component, ...]
    _tree: _Tree
    # The lock that makes the check of _tree.ran and the start one step for
    # concurrent callers.
    _lock: threading.Lock

    def __init__(self) -> Never:
        raise TypeError("a Graph is made by Network.build() or mujou.run()")

    @property
    def nodes(self) -> tuple[tuple[str, Node], ...]:
        """``(path, node)`` for every node in the tree, depth first in attribute
        definition order.

        A path is the dotted attribute path from the root, e.g. ``"body.l1"``;
        a node that is itself the root (``mujou.run(node)``) is at ``""``.
        """
        return self._nodes

    def port_path(self, port: Port[Any, Any, Any]) -> str:
        """Return the path of ``port`` through its owner, e.g. ``"body.l1.next"``
        (``"next"`` when the owner is itself the root).

        Raises:
            ValueError: ``port`` is not owned by a node of this graph.
        """
        if port not in self._ports:
            raise ValueError(f"{port!r} is not a port of this graph")
        return _path(self._nodes, self._ports[port])


@final
class _Tree:
    """What the members of one built tree share: how it was built and whether it
    ran, for the error when one of them is built again."""

    __slots__ = ("by_run", "ran")

    def __init__(self, by_run: bool) -> None:
        # Built by mujou.run for its own Compose: built only to run.
        self.by_run = by_run
        # A run has started (or tried to start and failed in a node).
        self.ran = False


def _make_graph(
    nodes: tuple[tuple[str, Node], ...],
    ports: Mapping[Port[Any, Any, Any], tuple[int, str]],
    members: tuple[Component, ...],
    tree: _Tree,
) -> Graph:
    graph = object.__new__(Graph)
    graph._nodes = nodes  # pyright: ignore[reportPrivateUsage]  # Graph's own module
    graph._ports = ports  # pyright: ignore[reportPrivateUsage]
    graph._members = members  # pyright: ignore[reportPrivateUsage]
    graph._tree = tree  # pyright: ignore[reportPrivateUsage]
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
                super().__init__()
                self.l1 = Hidden(784, 256)
                self.l2 = Hidden(256, 256)
                self.l1.connect(self.l2)  # l1.next <-> l2.prev
                self.prev = self.l1.prev  # expose the previous side
                self.next = self.l2.next  # expose the next side


        class Classifier(Network):
            def __init__(self) -> None:
                super().__init__()
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
            RuntimeError: Part of the tree (a node, a network or a port) belongs
                to another built network, or this network is part of a built
                one.
            ValueError: The structure is invalid: the tree has no node; a component
                appears at two paths; a port is owned by two nodes (or under two
                names); a port is declared as a class attribute; a node holds a
                node or network as an attribute; a network exposes a port that no
                node inside it owns; a channel leads to a port outside the tree; or
                a component has more than one ``PrevPort`` or ``NextPort`` as
                direct attributes.
        """
        # One build at a time: two threads building one tree would each make a
        # graph, and both would run.
        with _BUILD_LOCK:
            if self._graph is None:
                self._graph = _build(self, by_run=isinstance(self, RunCompose))
            return self._graph


def _build(root: Component, by_run: bool) -> Graph:
    """Check the tree of ``root``, freeze it and mark it as built, as
    ``Network.build`` does; ``by_run`` when ``mujou.run`` builds it only to run.
    Call it under ``_BUILD_LOCK``."""
    members: list[Component] = []
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
        tree = _tree_of(component)
        if tree is not None:
            # What mujou.run was given is named without the path of its
            # own Compose.
            given = path == "" or (by_run and "." not in path)
            name = type(component).__name__ if given else _named(component, path)
            if tree.ran or tree.by_run:
                raise RuntimeError(
                    f"{name} has already run; create a new instance to run again"
                )
            raise RuntimeError(f"{name} belongs to a network that is already built")
        members.append(component)
        _check_declarations(component, path)
        if isinstance(component, Node):
            index = len(nodes)
            nodes.append((path, component))
            for name, value in vars(component).items():
                if isinstance(value, Component):
                    holder = (
                        f"node {path!r}"
                        if path
                        else f"the node {type(component).__name__}"
                    )
                    raise ValueError(
                        f"{holder} holds a {type(value).__name__} at {name!r}; "
                        "only networks hold nodes and networks"
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

    visit(root, "")
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
                    "outside the tree being built (only direct attributes are "
                    "walked, not lists or dicts)"
                )
    for port in ports:
        freeze(port)
    built = _Tree(by_run)
    for member in members:
        _mark_built(member, built)
    return _make_graph(tuple(nodes), MappingProxyType(ports), tuple(members), built)


def build_node(node: Node) -> Graph:
    """Build the tree of ``node`` alone, the node being the root at the path
    ``""``, for ``mujou.run`` to run once."""
    with _BUILD_LOCK:
        return _build(node, by_run=True)


def unbuild(graph: Graph) -> None:
    """Undo the build of ``graph``, which has not run, so that its parts can be
    built again (``mujou.run`` does this for a tree it built itself whose run was
    refused)."""
    with _BUILD_LOCK:
        for port in graph._ports:  # pyright: ignore[reportPrivateUsage]  # Graph's own module
            thaw(port)
        members = graph._members  # pyright: ignore[reportPrivateUsage]
        for member in members:
            _BUILT.pop(id(member), None)
        root = members[0]
        if isinstance(root, Network) and root._graph is graph:  # pyright: ignore[reportPrivateUsage]
            root._graph = None  # pyright: ignore[reportPrivateUsage]


# The nodes and networks of every built tree: a part of one cannot join another.
# Ports are marked too (frozen), but a node without ports has only this mark.
# Keyed by id and checked by identity, so that a component's own __eq__ / __hash__
# (e.g. a dataclass, which is unhashable) plays no part.
_BUILT: Final[dict[int, tuple[weakref.ref[Component], _Tree]]] = {}
_BUILD_LOCK: Final = threading.Lock()


def _tree_of(component: Component) -> _Tree | None:
    """The tree ``component`` is built in, or ``None`` if it is not built."""
    mark = _BUILT.get(id(component))
    if mark is None or mark[0]() is not component:
        return None
    return mark[1]


def _mark_built(component: Component, tree: _Tree) -> None:
    key = id(component)

    def forget(ref: weakref.ref[Component]) -> None:
        # Only this mark: a later component may have reused the id.
        mark = _BUILT.get(key)
        if mark is not None and mark[0] is ref:
            _BUILT.pop(key, None)

    _BUILT[key] = (weakref.ref(component, forget), tree)


class Compose(Network):
    """A network of the given nodes and networks side by side, unconnected unless
    they were connected before.

    One given by position is at the path of its position (``"0"``, ``"1"``, ...)
    and one given by keyword at its keyword, so the node ``l1`` of
    ``Compose(a, b)`` is ``"1.l1"`` in ``b`` and that of ``Compose(enc=a)`` is
    ``"enc.l1"``; a node ``n`` in ``Compose(n, a)`` is ``"0"``, as if it were an
    attribute ``"0"`` of a network. ``Compose(n)`` of one component still puts it
    at ``"0"``. ``mujou.run(a, b)`` runs ``Compose(a, b)``; ``mujou.run(a)`` runs
    ``a`` alone as the root.

    Example::

        enc, dec = Encoder(), Decoder()
        enc.connect(dec)
        with mujou.run(Compose(enc=enc, dec=dec)) as run:
            state = run.state_dict()  # nodes "enc.l1", ..., "dec.l1", ...

    Raises:
        TypeError: A value is neither a ``Node`` nor a ``Network``.
        ValueError: A keyword is empty, holds a ``.`` (a path would be
            ambiguous), starts with ``_``, is the name of an attribute of
            ``Network`` (such as ``build``), or is the path of a position.
    """

    def __init__(self, *components: Node | Network, **named: Node | Network) -> None:
        super().__init__()
        children = {str(i): component for i, component in enumerate(components)}
        for name, component in named.items():
            if (
                not name
                or "." in name
                or name.startswith("_")
                or hasattr(Compose, name)
                or name in children
            ):
                raise ValueError(
                    f"cannot compose a component as {name!r}: the name is empty, "
                    "holds a '.', is private, is an attribute of Network, or is a "
                    "position"
                )
            children[name] = component
        for name, component in children.items():
            # The annotation is not checked at runtime.
            if not isinstance(component, Node | Network):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    "Compose takes nodes and networks, got "
                    f"{type(component).__name__} at {name!r}"
                )
            setattr(self, name, component)


@final
class RunCompose(Compose):
    """The ``Compose`` that ``mujou.run`` makes of what it was given: built only to
    run, and named in errors by what was given rather than by its paths."""


def start_once[T](graph: Graph, start: Callable[[], T]) -> T:
    """Call ``start`` to run ``graph`` unless it has run, and mark it as run.

    A start refused before any node started (an ``Exception`` other than
    ``NetworkError``) does not count, so the graph may run once the cause is
    gone. An interruption (``KeyboardInterrupt``) counts, wherever it came: the
    start may go on in the background, and a run interrupted before it started
    the runtime is counted too, to be safe.

    Raises:
        RuntimeError: ``graph`` has already run.
    """
    tree = graph._tree  # pyright: ignore[reportPrivateUsage]  # Graph's own module
    with graph._lock:  # pyright: ignore[reportPrivateUsage]
        if tree.ran:
            root = graph._members[0]  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError(
                f"{type(root).__name__} has already run; create a new instance to "
                "run again"
            )
        try:
            started = start()
        except Exception as e:
            if isinstance(e, _native.NetworkError):
                tree.ran = True
            raise
        except BaseException:
            tree.ran = True
            raise
        tree.ran = True
        return started


def _path(nodes: Sequence[tuple[str, Node]], owner: tuple[int, str]) -> str:
    """Return the path of the port ``owner`` (node index, attribute name) names."""
    index, name = owner
    return _child(nodes[index][0], name)


def _child(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _where(path: str) -> str:
    return repr(path) if path else "the root"


def _named(component: Component, path: str) -> str:
    """Name ``component`` at ``path`` in an error: by its class alone at the root."""
    name = type(component).__name__
    return f"{name} at {path!r}" if path else name


def _check_declarations(component: Component, path: str) -> None:
    """Reject ports declared on the class and more than one side of a kind."""
    for klass in type(component).__mro__:
        for name, value in vars(klass).items():
            if isinstance(value, Port):
                raise ValueError(
                    f"{_named(component, path)} declares the port "
                    f"{name!r} on the class {klass.__name__}; declare ports as "
                    "instance attributes in __init__"
                )
    for kind in (PrevPort, NextPort):
        n = len(_sides(component, kind))
        if n > 1:
            raise ValueError(
                f"{_named(component, path)} has {n} {kind.__name__} "
                "attributes; at most one is allowed"
            )


# Last: node imports Component from this module, which is defined by now. Imported
# at run time, not only for the type checker, so that the annotations naming Node
# (e.g. of Compose.__init__, which inspect.signature evaluates) resolve. This
# assumes that this module is imported before node (mujou/__init__.py imports it
# before node; mujou.time, imported earlier, imports node only lazily): imported
# the other way round, node would be only partly initialized here, without Node,
# and the import would fail.
from mujou._core.node import Node  # noqa: E402
