"""Components (nodes and networks), how they connect, and how a network is built.

A network is the structure of a run: nodes and nested networks as attributes, and
ports connected into channels. ``Network.build`` walks the tree once, checks it,
and freezes it (``docs/network.md``).
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import torch

from mujou._port import Channel, NextPort, Port, PrevPort, freeze, is_frozen

if TYPE_CHECKING:
    from mujou._node import Node
    from mujou._session import Session


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
        return _path(self.nodes, self.ports[port])


class Network(Component):
    """A composite of nodes and other networks, declared as attributes.

    Assign nodes and networks as attributes, connect their ports, and expose the
    ports the outside should reach as attributes of the network. An exposed
    ``PrevPort`` / ``NextPort`` becomes the previous / next side of the network, so
    networks connect just like nodes. Only direct attributes are walked (not lists
    or dicts). A network is a build-time structure: its nodes still run
    independently and asynchronously.

    Call ``build()`` once on the root network to check and freeze the structure,
    then ``run()`` it (or call ``mujou.run(network)``, which builds if needed). The
    root then talks with the running nodes: ``write`` to its unconnected ports,
    ``read`` / ``subscribe`` any port's output, take the ``state_dict``, and
    ``shutdown``. Declare ports as instance attributes (in ``__init__``), not class
    attributes, and do not use the names of these methods for attributes.

    If a node raises, the whole network stops, and every later call of ``write``,
    ``read``, ``subscribe``, ``state_dict``, ``shutdown`` and ``join`` raises
    ``NetworkError`` whose ``__cause__`` is the node's exception. Calling the
    methods that wait (``state_dict``, ``shutdown``, ``join``) from a node's hook
    raises ``RuntimeError``.

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
        net.write({net.prev: x0, net.teacher: y0})  # initial values
        net.run()  # returns once every node has started
        net.write({net.prev: x, net.teacher: y})  # switch together
        mujou.time.sleep(0.1)  # present the input for 0.1 s of global time
        prediction, t = net.read(net.head.next)
        net.shutdown()
    """

    _graph: Graph | None = None
    _session: Session | None = None

    def build(self) -> None:
        """Check the structure of this network as the root and freeze it.

        After ``build``, no port in the tree takes new connections.

        Raises:
            RuntimeError: This network is already built, or part of the tree belongs
                to another built network.
            ValueError: The structure is invalid: the tree has no node; a component
                appears at two paths; a port is owned by two nodes (or under two
                names); a port is declared as a class attribute; a node holds a
                node or network as an attribute; a network exposes a port that no
                node inside it owns; a channel leads to a port outside the tree; or
                a component has more than one ``PrevPort`` or ``NextPort`` as
                direct attributes.
        """
        # Imported here: _node imports Component from this module.
        from mujou._node import Node

        if self._graph is not None:
            raise RuntimeError("the network is already built")

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
        self._graph = Graph(tuple(nodes), MappingProxyType(ports))
        # Imported here: _session imports _node, which imports this module.
        from mujou._session import Session

        self._session = Session(self._graph)

    def run(
        self,
        *,
        capture_state_on_exception: bool = False,
        pause_timeout: float = 10.0,
        _manual: bool = False,
    ) -> None:
        """Start running the nodes in the background and return once they started.

        Every node's thread is set up, ``on_start`` (and, after
        ``load_state_dict``, the node's ``load_state_dict``) has returned, and the
        global timer runs when this returns. A network runs once.

        Args:
            capture_state_on_exception: When a node raises, take the state tree
                before stopping, for ``exception_state_dict``.
            pause_timeout: Real seconds that ``state_dict`` waits for every node
                to pause (the end of its current ``on_update``).
            _manual: For tests: the nodes update only in ``_step``.

        Raises:
            RuntimeError: The network is not built, or has already run.
            NetworkError: A node raised while starting; the network has stopped.
        """
        _session(self).run(
            manual=_manual,
            pause_timeout=pause_timeout,
            capture_state_on_exception=capture_state_on_exception,
        )

    def write(
        self,
        values: Mapping[Port[Any, Any, Any], torch.Tensor | Mapping[str, torch.Tensor]],
    ) -> None:
        """Write messages to unconnected single-peer ports that receive.

        The values are detached and copied. All of them switch together and carry
        the same time: at any one moment, a node sees either all of them or none.
        (A node that reads two of them one after the other may still see a later
        write in between; compare the times to tell.) Written before ``run``, they
        are the initial values; they take precedence over ``load_state_dict``.

        Raises:
            RuntimeError: The network is not built.
            ValueError: A port is outside the network, connected, takes many
                peers, or receives nothing; or a shape or the keys differ.
            TypeError: A message or dtype differs from the spec.
            NetworkError: The network stopped because a node raised.
        """
        _session(self).write(values)

    def read[S](self, port: Port[S, Any, Any]) -> tuple[S, float]:
        """Return the latest message ``port`` sent and its global time in seconds.

        Any port that sends can be read, so a connected port shows an inner
        node. Only ``send`` (to every peer) updates the value read here; before
        the first one it is zeros. The tensors are new copies on the CPU.

        Raises:
            RuntimeError: The network is not built.
            ValueError: ``port`` is outside the network or sends nothing.
            NetworkError: The network stopped because a node raised.
        """
        return cast(tuple[S, float], _session(self).read(port))

    def subscribe[S](self, port: Port[S, Any, Any]) -> Iterator[tuple[S, float]]:
        """Iterate over the messages ``port`` sends from now on, with their times.

        Each step waits until a newer message than the last one is sent, and
        yields the latest; messages sent in between are skipped. Only the
        consumer waits: the sending node never does. The iterator ends when the
        network shuts down normally.

        Raises:
            RuntimeError: The network is not built.
            ValueError: ``port`` is outside the network or sends nothing.
            NetworkError: The network stopped because a node raised (also from
                the iterator).
        """
        return cast(Iterator[tuple[S, float]], _session(self).subscribe(port))

    def state_dict(self) -> dict[str, Any]:
        """Pause the network, copy its state at one global time, and resume.

        The tree, which pickles, is ``{"nodes": {path: state}, "slots": {name:
        (message, time_ns)}, "last_update_ns": {path: int}, "timer": {...}}``.

        Raises:
            RuntimeError: The network is not running (before ``run`` or after
                ``shutdown``), or this is called from a node's hook.
            PauseTimeoutError: A node did not pause within ``pause_timeout``; the
                network keeps running.
            NetworkError: The network stopped because a node raised; see
                ``exception_state_dict``.
        """
        return _session(self).state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a tree from ``state_dict`` before ``run``.

        The slots are restored now; the nodes' states, their last update times
        and the global timer when ``run`` starts. The network must have the same
        node paths and slots as the one the tree came from.

        Raises:
            RuntimeError: The network is not built, or has already run.
            ValueError: The tree does not match the network.
        """
        _session(self).load_state_dict(state)

    def exception_state_dict(self) -> dict[str, Any] | None:
        """Return the tree taken when a node raised, or ``None``.

        Taken only with ``run(capture_state_on_exception=True)``. Besides the keys
        of ``state_dict`` it has ``"exception": {"node": path, "hook": name,
        "error": traceback}``. A node whose ``state_dict`` raised is missing.
        """
        session = self._session
        return None if session is None else session.exception_state_dict()

    def shutdown(self) -> None:
        """Stop every node and wait until they have stopped (``on_stop``).

        Does nothing before ``run`` or after the end.

        Raises:
            NetworkError: The network stopped because a node raised.
        """
        _session(self).shutdown()

    def join(self) -> None:
        """Wait until the network stops.

        Raises:
            RuntimeError: The network has not run.
            NetworkError: The network stopped because a node raised.
        """
        _session(self).join()

    def _step(self, order: Sequence[Node] | None = None) -> None:
        """For tests with ``run(_manual=True)``: update each node once, in
        ``order`` (by default the order of the tree), and return when done.

        A node whose global time has not advanced since its last update is not
        updated.
        """
        _session(self).step(order)


def run[N: Network](
    network: N,
    *,
    capture_state_on_exception: bool = False,
    pause_timeout: float = 10.0,
    _manual: bool = False,
) -> N:
    """Build ``network`` unless it is built, run it, and return it.

    Example::

        net = mujou.run(Classifier())
        net.write({net.prev: x, net.teacher: y})

    Raises as ``Network.build`` and ``Network.run`` do.
    """
    if network._graph is None:  # pyright: ignore[reportPrivateUsage]  # Network's own module
        network.build()
    network.run(
        capture_state_on_exception=capture_state_on_exception,
        pause_timeout=pause_timeout,
        _manual=_manual,
    )
    return network


def _session(network: Network) -> Session:
    if network._session is None:  # pyright: ignore[reportPrivateUsage]  # Network's own module
        raise RuntimeError("the network is not built; call build() first")
    return network._session  # pyright: ignore[reportPrivateUsage]  # Network's own module


def graph(network: Network) -> Graph:
    """Return the graph of a built ``network``.

    Raises:
        RuntimeError: ``network`` is not built.
    """
    if network._graph is None:  # pyright: ignore[reportPrivateUsage]  # Network's own module
        raise RuntimeError("the network is not built; call build() first")
    return network._graph  # pyright: ignore[reportPrivateUsage]  # Network's own module


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
