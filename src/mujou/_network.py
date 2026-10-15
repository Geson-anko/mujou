"""Components (nodes and networks), how they connect, and how a network is built.

A network is the structure of a run: nodes and nested networks as attributes, and
ports connected into channels. ``Network.build`` walks the tree once, checks it,
and freezes it (``docs/network.md``).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from mujou._port import Channel, NextPort, Port, PrevPort, freeze, is_frozen

if TYPE_CHECKING:
    from mujou._node import Node


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


def _side(
    component: Component, kind: type[PrevPort[Any, Any, Any] | NextPort[Any, Any, Any]]
) -> Port[Any, Any, Any]:
    """Return the only direct attribute of ``component`` that is a ``kind``."""
    found = [p for _, p in _ports(component) if isinstance(p, kind)]
    if len(found) != 1:
        raise ValueError(
            f"{type(component).__name__} must have exactly one {kind.__name__} as a "
            f"direct attribute to connect, found {len(found)}"
        )
    return found[0]


@dataclass(frozen=True)
class Graph:
    """The structure of a built network.

    Attributes:
        nodes: ``(path, node)`` for every node in the tree, depth first in attribute
            definition order. A path is the dotted attribute path from the root,
            e.g. ``"body.l1"``.
        ports: For every port owned by a node, the index of the owner in ``nodes``
            and the attribute name of the port.
    """

    nodes: tuple[tuple[str, Node], ...]
    # Any: ports of every message type are handled alike.
    ports: Mapping[Port[Any, Any, Any], tuple[int, str]]

    def port_path(self, port: Port[Any, Any, Any]) -> str:
        """Return the path of ``port`` through its owner, e.g. ``"body.l1.next"``.

        Raises:
            ValueError: ``port`` is not owned by a node of this network.
        """
        if port not in self.ports:
            raise ValueError(f"{port!r} is not a port of this network")
        index, name = self.ports[port]
        return f"{self.nodes[index][0]}.{name}"


class Network(Component):
    """A composite of nodes and other networks, declared as attributes.

    Assign nodes and networks as attributes, connect their ports, and expose the
    ports the outside should reach as attributes of the network. An exposed
    ``PrevPort`` / ``NextPort`` becomes the previous / next side of the network, so
    networks connect just like nodes. Only direct attributes are walked (not lists
    or dicts). A network is a build-time structure: its nodes still run
    independently and asynchronously.

    Call ``build()`` once on the root network to check and freeze the structure.

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
        net.build()
    """

    _graph: Graph | None = None

    def build(self) -> None:
        """Check the structure of this network as the root and freeze it.

        After ``build``, no port in the tree takes new connections.

        Raises:
            RuntimeError: This network is already built, or part of the tree belongs
                to another built network.
            ValueError: The structure is invalid: the tree has no node; a component
                appears at two paths; a port is owned by two nodes (or under two
                names); a node holds a node or network as an attribute; a network
                exposes a port that no node of the tree owns; a channel leads to a
                port outside the tree; or a component has more than one
                ``PrevPort`` or ``NextPort`` as direct attributes.
        """
        # Imported here: _node imports Component from this module.
        from mujou._node import Node

        if self._graph is not None:
            raise RuntimeError("the network is already built")

        nodes: list[tuple[str, Node]] = []
        ports: dict[Port[Any, Any, Any], tuple[int, str]] = {}
        exposed: list[tuple[str, Port[Any, Any, Any]]] = []
        seen: dict[int, str] = {}  # id(component) -> path

        def visit(component: Component, path: str) -> None:
            where = repr(path) if path else "the root"
            if id(component) in seen:
                first = seen[id(component)]
                raise ValueError(
                    f"{type(component).__name__} appears at both "
                    f"{repr(first) if first else 'the root'} and {where}"
                )
            seen[id(component)] = path
            for kind in (PrevPort, NextPort):
                n = sum(isinstance(p, kind) for _, p in _ports(component))
                if n > 1:
                    raise ValueError(
                        f"{type(component).__name__} at {where} has {n} "
                        f"{kind.__name__} attributes; at most one is allowed"
                    )
            if isinstance(component, Node):
                index = len(nodes)
                nodes.append((path, component))
                for name, value in vars(component).items():
                    if isinstance(value, Component):
                        raise ValueError(
                            f"node {where} holds a {type(value).__name__} at "
                            f"{name!r}; only networks hold nodes and networks"
                        )
                for name, port in _ports(component):
                    if port in ports:
                        raise ValueError(
                            f"the port {_path(nodes, (index, name))!r} is already "
                            f"owned as {_path(nodes, ports[port])!r}"
                        )
                    ports[port] = (index, name)
                return
            for name, value in vars(component).items():
                if isinstance(value, Component):
                    visit(value, f"{path}.{name}" if path else name)
            for name, port in _ports(component):
                exposed.append((f"{path}.{name}" if path else name, port))

        visit(self, "")
        if not nodes:
            raise ValueError("the network has no node")
        for path, port in exposed:
            if port not in ports:
                raise ValueError(
                    f"the port exposed at {path!r} is not owned by a node of the "
                    "network"
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
                        "outside the network"
                    )
        for port in ports:
            freeze(port)
        self._graph = Graph(tuple(nodes), MappingProxyType(ports))


def _path(nodes: list[tuple[str, Node]], owner: tuple[int, str]) -> str:
    return f"{nodes[owner[0]][0]}.{owner[1]}"
