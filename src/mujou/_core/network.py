"""Networks: the composites of nodes and other networks (``specs/network.md``).

A network is the structure of a run: nodes and nested networks as attributes, and
ports connected into channels. ``Network.build`` makes its ``Graph`` (``graph.py``),
which ``mujou.run`` runs.
"""

from typing import final

from mujou._core import component as component_module
from mujou._core.component import Component
from mujou._core.graph import BUILD_LOCK, Graph, build_tree, unbuild_tree
from mujou._core.node import Node


class Network(Component):
    """A composite of nodes and other networks, declared as attributes.

    Assign nodes and networks as attributes, connect their ports with
    ``Port.connect``, and expose the ports the outside should reach as attributes
    of the network (by convention ``prev`` and ``next`` for the previous and the
    next side, e.g. ``self.prev = self.l1.prev``). The network registers what is
    assigned, as ``nn.Module`` registers submodules, so nodes and networks in
    lists or dicts are not part of it. A network is a build-time
    structure: its nodes still run independently and asynchronously.

    Call ``super().__init__()`` before assigning a node, a network or a port.
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
                self.l1.next.connect(self.l2.prev)
                self.prev = self.l1.prev  # expose the previous side
                self.next = self.l2.next  # expose the next side


        class Classifier(Network):
            def __init__(self) -> None:
                super().__init__()
                self.body = Stack()
                self.head = Head()  # a node with prev, next and teacher ports
                self.body.next.connect(self.head.prev)  # l2.next <-> head.prev
                self.prev = self.body.prev  # the input
                self.teacher = self.head.teacher  # the target, a port of head


        net = Classifier()
        with mujou.run(net, inputs={net.prev: x0, net.teacher: y0}) as run:
            run.write({net.prev: x, net.teacher: y})  # switch together
            mujou.time.sleep(0.1)  # present the input for 0.1 s of global time
            prediction = run.read(net.head.next)
            state = run.state_dict()
    """

    _graph: Graph | None = None

    def build(self) -> Graph:
        """Check the structure of this network as the root, freeze it and return
        its graph.

        After ``build``, no port in the tree takes new connections, and no part of
        the tree, this network included, can become part of a bigger network:
        build only the network you will run. Calling it again returns the same
        graph. ``mujou.run`` calls it.

        Raises:
            RuntimeError: Part of the tree (a node, a network or a port) belongs
                to another built network, or this network is part of a built
                one.
            ValueError: The structure is invalid: the tree has no node; a component
                appears at two paths; a port is owned by two nodes (or under two
                names); a port is declared as a class attribute; a network exposes
                a port that no node inside it owns; or a channel leads to a port
                outside the tree.
        """
        # One build at a time: two threads building one tree would each make a
        # graph, and both would run.
        with BUILD_LOCK:
            if self._graph is None:
                self._graph = build_tree(self, by_run=isinstance(self, RunGroup))
            return self._graph


def unbuild(graph: Graph) -> None:
    """Undo the build of ``graph``, which has not run, so that its parts can be
    built again (``mujou.run`` does this for a tree it built itself whose run was
    refused)."""
    with BUILD_LOCK:
        root = unbuild_tree(graph)
        if isinstance(root, Network) and root._graph is graph:  # pyright: ignore[reportPrivateUsage]  # Network's own module
            root._graph = None  # pyright: ignore[reportPrivateUsage]


# component.py names Node and Network in annotations but cannot import them (both
# import it), so it imports them only for the type checker. Bind them in its
# globals here, where both exist, so that inspect.signature and get_type_hints
# can resolve its annotations (evaluated lazily, on that request).
component_module.Node = Node
component_module.Network = Network


class Group(Network):
    """A network of the given nodes and networks side by side, unconnected unless
    they were connected before.

    One given by position is at the path of its position (``"0"``, ``"1"``, ...)
    and one given by keyword at its keyword, so the node ``l1`` of
    ``Group(a, b)`` is ``"1.l1"`` in ``b`` and that of ``Group(enc=a)`` is
    ``"enc.l1"``; a node ``n`` in ``Group(n, a)`` is ``"0"``, as if it were an
    attribute ``"0"`` of a network. ``Group(n)`` of one component still puts it
    at ``"0"``. ``mujou.run(a, b)`` runs ``Group(a, b)``; ``mujou.run(a)`` runs
    ``a`` alone as the root.

    Example::

        enc, dec = Encoder(), Decoder()
        enc.next.connect(dec.prev)
        with mujou.run(Group(enc=enc, dec=dec)) as run:
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
                or hasattr(Group, name)
                or name in children
            ):
                raise ValueError(
                    f"cannot group a component as {name!r}: the name is empty, "
                    "holds a '.', is private, is an attribute of Network, or is a "
                    "position"
                )
            children[name] = component
        for name, component in children.items():
            # The annotation is not checked at runtime.
            if not isinstance(component, Node | Network):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError(
                    "Group takes nodes and networks, got "
                    f"{type(component).__name__} at {name!r}"
                )
            setattr(self, name, component)


@final
class RunGroup(Group):
    """The ``Group`` that ``mujou.run`` makes of what it was given: built only to
    run, and named in errors by what was given rather than by its paths."""
