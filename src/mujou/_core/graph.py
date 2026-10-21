"""How a network is built: the tree walked once, checked and marked as built,
and the ``Graph`` it gives, which ``mujou.run`` runs once (``specs/network.md``).
"""

import threading
import weakref
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Never, final

from mujou._core.component import Component, children_of, ports_of
from mujou._core.errors import NetworkError
from mujou._core.node import Node
from mujou._core.port import AnyPort, MultiPort, Port


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

    _nodes: tuple[tuple[str, Node], ...]
    _ports: Mapping[AnyPort, tuple[int, str]]
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

    def port_path(self, port: AnyPort) -> str:
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
        # Built by mujou.run for its own Group: built only to run.
        self.by_run = by_run
        # A run has started (or tried to start and failed in a node).
        self.ran = False


def _make_graph(
    nodes: tuple[tuple[str, Node], ...],
    ports: Mapping[AnyPort, tuple[int, str]],
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


def ports(graph: Graph) -> Mapping[AnyPort, tuple[int, str]]:
    """For every port owned by a node of ``graph``, the index of the owner in
    ``graph.nodes`` and the attribute name of the port."""
    return graph._ports  # pyright: ignore[reportPrivateUsage]  # Graph's own module


def build_tree(root: Component, by_run: bool) -> Graph:
    """Check the tree of ``root`` and mark it as built, as
    ``Network.build`` does; ``by_run`` when ``mujou.run`` builds it only to run.
    Call it under ``BUILD_LOCK``."""
    members: list[Component] = []
    nodes: list[tuple[str, Node]] = []
    ports: dict[AnyPort, tuple[int, str]] = {}
    # (attribute path, port, indices of the nodes in the exposing subtree)
    exposed: list[tuple[str, AnyPort, range]] = []
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
            # own Group.
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
            for name, port in ports_of(component).items():
                owner = (index, name)
                if port in ports:
                    raise ValueError(
                        f"the port {_path(nodes, owner)!r} is already owned as "
                        f"{_path(nodes, ports[port])!r}"
                    )
                ports[port] = owner
            return
        first = len(nodes)
        for name, child in children_of(component).items():
            visit(child, _child(path, name))
        for name, port in ports_of(component).items():
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
        for channel in port.channels:
            if channel.other(port) not in ports:
                raise ValueError(
                    f"the port {_path(nodes, owner)!r} is connected to a port "
                    "outside the tree being built (only direct attributes are "
                    "walked, not lists or dicts)"
                )
    built = _Tree(by_run)
    for member in members:
        _mark_built(member, built)
    return _make_graph(tuple(nodes), MappingProxyType(ports), tuple(members), built)


def build_node(node: Node) -> Graph:
    """Build the tree of ``node`` alone, the node being the root at the path
    ``""``, for ``mujou.run`` to run once."""
    with BUILD_LOCK:
        return build_tree(node, by_run=True)


def unbuild_tree(graph: Graph) -> Component:
    """Drop the marks of the tree of ``graph`` and return its root
    (``network.unbuild`` forgets the root's graph). Call it under
    ``BUILD_LOCK``."""
    members = graph._members  # pyright: ignore[reportPrivateUsage]  # Graph's own module
    for member in members:
        _BUILT.pop(id(member), None)
    return members[0]


# The nodes and networks of every built tree: a part of one cannot join another.
# Keyed by id and checked by identity, so that a component's own __eq__ / __hash__
# (e.g. a dataclass, which is unhashable) plays no part.
_BUILT: Final[dict[int, tuple[weakref.ref[Component], _Tree]]] = {}
# One build (or undoing of one) at a time: two threads building one tree would
# each make a graph, and both would run.
BUILD_LOCK: Final = threading.Lock()


def is_built(component: Component) -> bool:
    """Whether ``component`` is part of a built tree."""
    return _tree_of(component) is not None


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
            if isinstance(e, NetworkError):
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
    """Reject ports declared on the class."""
    for klass in type(component).__mro__:
        for name, value in vars(klass).items():
            if isinstance(value, Port | MultiPort):
                raise ValueError(
                    f"{_named(component, path)} declares the port "
                    f"{name!r} on the class {klass.__name__}; declare ports as "
                    "instance attributes in __init__"
                )
