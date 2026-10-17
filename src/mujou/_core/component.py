"""The common base of nodes and networks: something with ports that connects
(``specs/network.md``)."""

from typing import TYPE_CHECKING, Any, cast

from mujou._core import _native
from mujou._core.port import Channel, NextPort, Port, PrevPort

if TYPE_CHECKING:
    from mujou._core.network import Network
    from mujou._core.node import Node


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


def attribute_ports(component: Component) -> list[tuple[str, Port[Any, Any, Any]]]:
    """Return ``(attribute name, port)`` for the direct attributes that are ports."""
    # Any: ports of every message type are handled alike.
    return [
        (name, cast(Port[Any, Any, Any], value))
        for name, value in vars(component).items()
        if isinstance(value, Port)
    ]


def sides(
    component: Component, kind: type[PrevPort[Any, Any, Any] | NextPort[Any, Any, Any]]
) -> list[Port[Any, Any, Any]]:
    """Return the distinct direct attributes of ``component`` that are ``kind``.

    The same port under two names counts once.
    """
    found: list[Port[Any, Any, Any]] = []
    for _, port in attribute_ports(component):
        # Ports compare by identity, so `in` finds the same port.
        if isinstance(port, kind) and port not in found:
            found.append(port)
    return found


def _side(
    component: Component, kind: type[PrevPort[Any, Any, Any] | NextPort[Any, Any, Any]]
) -> Port[Any, Any, Any]:
    """Return the only direct attribute of ``component`` that is a ``kind``."""
    found = sides(component, kind)
    if len(found) != 1:
        hint = ""
        # A network: every component but a node (the core's Node) is one. The
        # annotation of connect's `other` is not checked at runtime.
        is_component = isinstance(component, Component)  # pyright: ignore[reportUnnecessaryIsInstance]
        if not found and is_component and not isinstance(component, _native.Node):
            side = "next" if kind is NextPort else "prev"
            hint = f"; expose one, e.g. self.{side} = self.<child>.{side}"
        raise ValueError(
            f"{type(component).__name__} must have exactly one {kind.__name__} as a "
            f"direct attribute to connect, found {len(found)}{hint}"
        )
    return found[0]
