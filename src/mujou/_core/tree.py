"""The tree of a run: what ``mujou.run`` reads of its root, and checks, before
anything else (``specs/network.md``).

``walk`` reads the attributes of the root (its ``__dict__``) in the order of
assignment, and those of every network in it, down to the nodes: the children
of a network are the nodes and networks among its attributes, and the ports of
a node the ports among its attributes. A node is at the path of attributes
from the root (``"body.l1"``, and ``""`` for a node that is the root itself).
``walk`` changes nothing: a tree it refuses can be fixed and run again.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeIs, cast, final

import torch
from torch import nn

from mujou._core.network import Network
from mujou._core.node import Node, initialized, is_state
from mujou._core.port import AnyPort, MultiPort, Port

# What a tree is made of: the run sees each only as an attribute of its own.
_PARTS = (Port, MultiPort, Node, Network)


@final
@dataclass(frozen=True, slots=True)
class Member:
    """A node of a tree: its path, and its ports by attribute in the order of
    assignment."""

    path: str
    node: Node
    ports: Mapping[str, AnyPort]


@final
@dataclass(frozen=True, slots=True)
class Tree:
    """The nodes of a run, in the order of the tree (depth first, in the order
    of assignment), and the owner of each of their ports."""

    # How the root is named: its class.
    root: str
    members: tuple[Member, ...]
    # Each port and its owner: the path of the node and the attribute.
    owners: Mapping[AnyPort, tuple[str, str]]

    def name(self, path: str) -> str:
        """Name the node or network at ``path``: by its path, or by its class
        for the root."""
        return path or self.root

    def path(self, port: AnyPort) -> str:
        """The path of ``port``, a port of the tree (``"body.l1.next"``, or
        ``"next"`` for a port of a node that is the root)."""
        owner, attribute = self.owners[port]
        return _join(owner, attribute)

    def label(self, port: AnyPort) -> str:
        """Name ``port`` in an error: as its owner and attribute
        (``"body.l1.next"``, or ``"Leaky.next"`` for a port of a node that is
        the root), or by its ``repr`` if it is not a port of the tree."""
        found = self.owners.get(port)
        if found is None:
            return repr(port)
        owner, attribute = found
        return f"{self.name(owner)}.{attribute}"


def walk(root: Node | Network) -> Tree:
    """Read the tree of ``root`` and check it.

    Raises:
        TypeError: A node did not call ``super().__init__()``, holds a node or
            a network, or a node or a network holds a port, a node or a network
            in a list, a tuple or a dict.
        ValueError: A node, a network or a port is at two places of the tree;
            two nodes hold the same state (a tensor, an object with a state, or
            a parameter or a buffer of a module); or a port is connected to a
            port outside the tree.
    """
    walker = _Walker(type(root).__name__)
    walker.component(root, "")
    tree = Tree(walker.root, tuple(walker.members), walker.owners)
    for port in tree.owners:
        for channel in port.channels:
            peer = channel.other(port)
            if peer not in tree.owners:
                raise ValueError(
                    f"{tree.label(port)} is connected to {peer!r}, whose node is "
                    "not in the tree; add that node to the network"
                )
    return tree


@final
class _Walker:
    """What ``walk`` has found so far."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.members: list[Member] = []
        self.owners: dict[AnyPort, tuple[str, str]] = {}
        # Where each node, network and port is, by id.
        self._seen: dict[int, str] = {}
        # The path of the node that holds each value of a state, and where, by id.
        self._states: dict[int, tuple[str, str]] = {}

    def component(self, component: Node | Network, path: str) -> None:
        self._once(component, path or self.root)
        if isinstance(component, Node):
            self._node(component, path)
            return
        for attribute, value in vars(component).items():
            self._no_parts_in(value, self._at(path, attribute))
            if isinstance(value, Node | Network):
                self.component(value, _join(path, attribute))

    def _node(self, node: Node, path: str) -> None:
        if not initialized(node):
            raise TypeError(
                f"{type(node).__name__}.__init__ did not call super().__init__()"
            )
        ports: dict[str, AnyPort] = {}
        for attribute, value in vars(node).items():
            where = self._at(path, attribute)
            self._no_parts_in(value, where)
            if isinstance(value, Node | Network):
                raise TypeError(
                    f"{where} is a {type(value).__name__}: a node holds no nodes "
                    "or networks; put them side by side in a Network"
                )
            if _is_port(value):
                self._once(value, where)
                ports[attribute] = value
                self.owners[value] = (path, attribute)
            elif is_state(value):
                self._own(value, path, where)
        self.members.append(Member(path, node, ports))

    def _own(self, value: object, path: str, where: str) -> None:
        """Record that the node at ``path`` holds ``value`` in its state, at
        ``where``: the value itself and, for a module, each of its parameters
        and buffers, which another node may hold too (a tied weight)."""
        self._one_writer(value, path, where)
        if isinstance(value, nn.Module):
            for key, item in value.state_dict(keep_vars=True).items():
                # Not its extra state, which may be any value (an interned int).
                if isinstance(item, torch.Tensor):
                    self._one_writer(item, path, f"{where}.{key}")

    def _once(self, value: object, where: str) -> None:
        first = self._seen.setdefault(id(value), where)
        if first != where:
            raise ValueError(
                f"the {type(value).__name__} at {first!r} is also at {where!r}; a "
                "node, a network or a port is at one place of the tree"
            )

    def _one_writer(self, value: object, path: str, where: str) -> None:
        owner, first = self._states.setdefault(id(value), (path, where))
        if owner != path:
            raise ValueError(
                f"{first} and {where} hold the same {type(value).__name__}: a "
                "state has one writer, so give each node its own"
            )

    def _no_parts_in(self, value: object, where: str) -> None:
        kind = type(value).__name__
        if isinstance(value, dict):
            items: Iterable[object] = cast(dict[object, object], value).values()
        elif isinstance(value, list | tuple):
            items = cast(Iterable[object], value)
        else:
            return
        if any(isinstance(item, _PARTS) for item in items):
            raise TypeError(
                f"{where} holds ports, nodes or networks in a {kind}, which a run "
                "does not read; assign each to an attribute of its own (or use a "
                "MultiPort)"
            )

    def _at(self, path: str, attribute: str) -> str:
        """Name the attribute ``attribute`` of the component at ``path``."""
        return f"{path or self.root}.{attribute}"


def _is_port(value: object) -> TypeIs[AnyPort]:
    return isinstance(value, Port | MultiPort)


def _join(path: str, attribute: str) -> str:
    return f"{path}.{attribute}" if path else attribute
