"""Components (nodes and networks), how they connect, and how a network is built.

A network is the structure of a run: nodes and nested networks as attributes, and
ports connected into channels. ``Network.build`` walks the tree once, checks it,
and freezes it (``docs/network.md``).
"""

import math
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

    Declare ports as instance attributes (in ``__init__``), not class attributes,
    and do not use the names of the methods below for attributes.

    The host (the Python code outside the nodes) uses the root network, the one
    it called ``build`` on, in this order:

    1. ``build()``: once, to check and freeze the structure.
    2. Optionally, before ``run``: ``write`` the initial inputs, or
       ``load_state_dict`` a saved tree.
    3. ``run()``: once; the nodes then run in the background. ``mujou.run(net)``
       does 1 and 3 in one call.
    4. While it runs: ``write`` inputs, ``read`` / ``subscribe`` outputs, take a
       ``state_dict``. ``read`` also works before ``run``.
    5. ``shutdown()`` to stop it, or ``join()`` to wait until it stops.
    6. Optionally, after a normal end: take the final ``state_dict``.

    These methods are for the host: calling them from a node's hook raises
    ``RuntimeError``.

    If a node raises, the whole network stops. If that happens while ``run``
    starts the nodes, ``run`` raises ``NetworkError``; otherwise every later call
    of ``write``, ``read``, ``subscribe`` (and its iterator), ``state_dict``,
    ``shutdown`` and ``join`` raises it. Its ``__cause__`` is the node's exception.
    The state at the exception is not kept.

    A running network keeps running when the host drops it, until ``shutdown``
    or, at the latest, the interpreter's exit (which waits for ``on_stop``).

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
        state = net.state_dict()  # the state at the end, to save
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

    def run(self, *, pause_timeout: float = 10.0, _manual: bool = False) -> None:
        """Start running the nodes in the background and return once they started.

        Every node's thread is set up, ``on_start`` (and, after
        ``load_state_dict``, the node's ``load_state_dict``) has returned, and the
        global timer runs when this returns. A network runs once.

        Args:
            pause_timeout: Real seconds that ``state_dict`` waits for every node
                to pause (the end of its current ``on_update``).
            _manual: For tests: the nodes update only in ``_step``.

        Raises:
            RuntimeError: The network is not built, or has already run.
            ValueError: The network was restored by ``load_state_dict`` while
                another network runs in this process (the global time cannot be
                set back under it). Nothing started; stop the other network and
                call ``run`` again.
            NetworkError: A node raised while starting; the network has stopped.
        """
        _session(self).run(manual=_manual, pause_timeout=pause_timeout)

    # P: a dict built beforehand has a precise key type, which a Mapping keyed by
    # Port[Any, Any, Any] would reject (Mapping keys are invariant).
    def write[P: Port[Any, Any, Any]](
        self, values: Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]]
    ) -> None:
        """Write messages to unconnected single-peer ports that receive.

        The values are detached and copied. All of them switch together and carry
        the same time: at any one moment, a node sees either all of them or none.
        (A node that reads two of them one after the other may still see a later
        write in between; compare the times to tell.) Written before ``run``, they
        are the initial values. ``load_state_dict`` replaces what was written
        before it; what is written after it replaces the tree's values and carries
        the tree's time.

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
        the first one it is zeros (or the value restored by ``load_state_dict``).
        The tensors are new copies on the CPU.

        Raises:
            RuntimeError: The network is not built.
            ValueError: ``port`` is outside the network or sends nothing.
            NetworkError: The network stopped because a node raised.
        """
        return cast(tuple[S, float], _session(self).read(port))

    def subscribe[S](self, port: Port[S, Any, Any]) -> Iterator[tuple[S, float]]:
        """Iterate over the messages ``port`` sends, with their times.

        The first step yields the latest message at the time of this call (zeros
        or the restored value before the first ``send``) without waiting. Each
        later step waits until a newer message is sent and yields the latest;
        messages sent in between are skipped. Only the consumer waits: the
        sending node never does. The iterator ends when the network shuts down
        normally. Shut the network down before the program ends: a thread
        iterating here ends only with the network.

        Raises:
            RuntimeError: The network is not built or not run yet.
            ValueError: ``port`` is outside the network or sends nothing.
            NetworkError: The network stopped because a node raised (also from
                the iterator).
        """
        return cast(Iterator[tuple[S, float]], _session(self).subscribe(port))

    def state_dict(self) -> dict[str, Any]:
        """Copy the network's state at one global time, while it runs or after
        it ended normally.

        While it runs, this pauses the network, copies the state and resumes, and
        returns once the network has resumed. After a normal end, this copies the
        state as the nodes left it, calling each node's ``state_dict`` on this
        thread; the tree's timer then holds the time of the latest update or
        write, so that a restored network goes on from the end without the time
        that passed since.

        The tree is a copy (later updates of the nodes do not change it) and
        pickles; saving it is up to the host::

            {
                "nodes": {"body.l1": ..., ...},  # each node's state_dict()
                "slots": {"body.l1.next/out": (value, time_ns), ...},
                "last_update_ns": {"body.l1": int, ...},
                "timer": {"time_ns": int, "scale": float},
            }

        A node is keyed by its attribute path from the root. A slot is keyed by
        the path of the port through its owning node (``"body.l1.prev/in"``, not
        an exposed name such as ``"prev/in"``) and holds its latest message:
        ``/out`` what ``read`` returns, ``/ch0``, ``/ch1``, ... each channel, and
        ``/in`` what the host wrote.

        Raises:
            RuntimeError: The network has not run (also after ``load_state_dict``
                until ``run``), or this is called from a node's hook.
            PauseTimeoutError: A node did not pause within ``pause_timeout``; the
                network keeps running.
            NetworkError: The network stopped because a node raised; its state is
                not kept.
        """
        return _session(self).state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a tree from ``state_dict`` before ``run``.

        The slots are restored now; the nodes' states, their last update times
        and the global time when ``run`` starts. ``run`` passes each node a copy
        of ``state["nodes"][path]`` through its ``load_state_dict``, after its
        ``on_start``, and the network continues from the time the tree was taken.
        Build the network as the one the tree came from: it must have the same
        node paths and slots.

        Raises:
            RuntimeError: The network is not built, or has already run.
            ValueError: The tree does not match the network.
        """
        _session(self).load_state_dict(state)

    def shutdown(self, timeout: float = math.inf) -> None:
        """Stop every node and return once every ``on_stop`` has returned.

        Does nothing before ``run`` or after a normal end.

        Args:
            timeout: Real seconds to wait for the end. ``math.inf`` waits as
                long as it takes.

        Raises:
            RuntimeError: The network is not built.
            ValueError: ``timeout`` is negative or NaN; nothing was asked.
            TimeoutError: The network did not stop within ``timeout``. The stop
                stays asked: the nodes take no more updates and the network goes
                on stopping in the background (an ``on_stop`` may still be
                running). ``join`` waits for the end.
            NetworkError: The network stopped because a node raised.
        """
        _session(self).shutdown(timeout)

    def join(self, timeout: float = math.inf) -> None:
        """Wait until the network stops, without asking it to stop.

        It returns once another thread called ``shutdown`` and every ``on_stop``
        has returned; if a node raises, it raises instead. If nothing stops the
        network and ``timeout`` is ``math.inf``, it never returns.

        Args:
            timeout: Real seconds to wait for the end. ``math.inf`` waits as
                long as it takes.

        Raises:
            RuntimeError: The network has not run.
            ValueError: ``timeout`` is negative or NaN.
            TimeoutError: The network did not stop within ``timeout``. Nothing
                changes: a running network keeps running.
            NetworkError: The network stopped because a node raised.
        """
        _session(self).join(timeout)

    def _step(self, order: Sequence[Node] | None = None) -> None:
        """For tests with ``run(_manual=True)``: update each node once, in
        ``order`` (by default the order of the tree), and return when done.

        A node whose global time has not advanced since its last update is not
        updated.
        """
        _session(self).step(order)


def run[N: Network](
    network: N, *, pause_timeout: float = 10.0, _manual: bool = False
) -> N:
    """Build ``network`` unless it is built, run it, and return it.

    Example::

        net = mujou.run(Classifier())
        net.write({net.prev: x, net.teacher: y})

    Raises as ``Network.build`` and ``Network.run`` do.
    """
    if network._graph is None:  # pyright: ignore[reportPrivateUsage]  # Network's own module
        network.build()
    network.run(pause_timeout=pause_timeout, _manual=_manual)
    return network


def _session(network: Network) -> Session:
    # Imported here: _node imports this module.
    from mujou._node import in_hook

    if in_hook():
        raise RuntimeError(
            "a node's hook cannot call the network's host methods; a node talks "
            "only through its own ports"
        )
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
