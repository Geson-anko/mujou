"""The common base of nodes and networks: something with ports, which registers its
ports and its child nodes and networks as they are assigned
(``specs/network.md``)."""

import types
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from mujou._core import _native
from mujou._core.port import Port, assign_owner

if TYPE_CHECKING:
    from mujou._core.network import Network
    from mujou._core.node import Node

# The registries of a component before Component.__init__ has run: empty, and
# told apart from a component's own by identity.
_NOT_INITIALIZED: Final[Mapping[str, Any]] = MappingProxyType({})


class Component:
    """The common base of ``Node`` and ``Network``: something with ports.

    As ``nn.Module`` registers its parameters and submodules, a component
    registers what is assigned to its attributes: every ``Port`` and, on a
    network, every ``Node`` and ``Network``, by attribute name in the order of
    assignment. Assigning any other value under that name, or ``del``, removes
    it. ``Network.build`` walks only what is registered, so ports and components
    kept in lists or dicts are not part of the network.

    Call ``super().__init__()`` before assigning a port, a node or a network.
    Once the component is built (``Network.build`` or ``mujou.run``), its ports,
    nodes and networks cannot be assigned or removed.
    """

    # Any: ports of every message type are handled alike. Each registry is
    # replaced, never changed, so that a shallow copy (copy.copy) does not share
    # changes with the original.
    _ports: Mapping[str, Port[Any, Any, Any]] = _NOT_INITIALIZED
    _children: Mapping[str, Node | Network] = _NOT_INITIALIZED

    def __init__(self) -> None:
        object.__setattr__(self, "_ports", {})
        object.__setattr__(self, "_children", {})

    if not TYPE_CHECKING:
        # Hidden from the type checker, which would otherwise accept assignments
        # to undeclared attributes on every subclass.

        def __setattr__(self, name: str, value: object) -> None:
            # One isinstance for the common case, an ordinary value (say, a tensor
            # assigned in on_update): a port or a component is rare.
            if isinstance(value, _REGISTERED):
                _register(self, name, value)
            elif name in self._ports or name in self._children:
                _unregister(self, name, "replace")
            object.__setattr__(self, name, value)

        def __delattr__(self, name: str) -> None:
            if name in self._ports or name in self._children:
                _unregister(self, name, "delete")
            object.__delattr__(self, name)


# A tuple, not a union: isinstance checks a tuple several times faster, and
# Component.__setattr__ checks every assignment.
_REGISTERED: Final = (Port, Component)


def ports_of(component: Component) -> Mapping[str, Port[Any, Any, Any]]:
    """The ports registered on ``component``: attribute name to port, in the order
    of assignment."""
    return component._ports  # pyright: ignore[reportPrivateUsage]  # Component's own module


def children_of(component: Component) -> Mapping[str, Node | Network]:
    """The nodes and networks registered on ``component`` (only a network has
    any): attribute name to component, in the order of assignment."""
    return component._children  # pyright: ignore[reportPrivateUsage]


def is_data_descriptor(klass: type, name: str) -> bool:
    """Whether ``name`` on ``klass`` is a data descriptor other than a slot (a
    property, say), which stores the value somewhere else."""
    for base in klass.__mro__:
        if name in vars(base):
            attribute: object = vars(base)[name]
            return hasattr(type(attribute), "__set__") and not isinstance(
                attribute, types.MemberDescriptorType
            )
    return False


def _register(
    component: Component, name: str, value: Port[Any, Any, Any] | Component
) -> None:
    """Register ``value`` under ``name`` (or nothing for a property, whose setter
    assigns again).

    Raises:
        AttributeError: ``Component.__init__`` has not run.
        ValueError: ``component`` is a node and ``value`` a node or network.
        RuntimeError: ``component`` is built.
    """
    ports, children = ports_of(component), children_of(component)
    is_port = isinstance(value, Port)
    if ports is _NOT_INITIALIZED:
        what = "port" if is_port else type(value).__name__
        base = "Node" if isinstance(component, _native.Node) else "Network"
        raise AttributeError(
            f"cannot assign a {what} before {base}.__init__() call; call "
            "super().__init__() first"
        )
    if is_data_descriptor(type(component), name):
        return
    # A node: every component but a network is one (Node cannot be imported here).
    if not is_port and isinstance(component, _native.Node):
        raise ValueError(
            f"the node {type(component).__name__} holds a {type(value).__name__} at "
            f"{name!r}; only networks hold nodes and networks"
        )
    _check_not_built(component, name, "assign")
    if is_port and isinstance(component, _native.Node):
        assign_owner(value, type(component).__name__, name)
    # A name assigned again keeps its place in the order.
    ports, children = dict(ports), dict(children)
    if is_port:
        children.pop(name, None)
        ports[name] = value
    else:
        ports.pop(name, None)
        children[name] = value  # pyright: ignore[reportArgumentType]  # Node or Network
    object.__setattr__(component, "_ports", ports)
    object.__setattr__(component, "_children", children)


def _unregister(component: Component, name: str, how: str) -> None:
    """Remove ``name`` from the registries of ``component``.

    Raises:
        RuntimeError: ``component`` is built.
    """
    _check_not_built(component, name, how)
    ports = {k: v for k, v in ports_of(component).items() if k != name}
    children = {k: v for k, v in children_of(component).items() if k != name}
    object.__setattr__(component, "_ports", ports)
    object.__setattr__(component, "_children", children)


def _check_not_built(component: Component, name: str, how: str) -> None:
    # Imported here: graph.py imports this module.
    from mujou._core.graph import is_built

    if is_built(component):
        raise RuntimeError(
            f"cannot {how} {type(component).__name__}.{name}: a port, node or "
            "network is not assigned or removed once the tree is built (by "
            "Network.build or mujou.run); create a new instance instead"
        )
