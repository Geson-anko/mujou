"""The slots of a run, laid out for its tree, and the drivers of its nodes
(``specs/network.md``, ``specs/execution-model.md``).

``wire`` lays out, on the run's clock, the slots of every port of the tree: an
``Outlet`` for a port that sends (its outbox, which the host reads, and one
slot per channel), and a slot in the run's ``InputGroup`` for an unconnected
``Port`` that receives, which the host writes. It gives the driver of each
node the wire of each of its ports to those slots (``port.py``). It makes only
new objects: the nodes and the ports stay as they are.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, final

import torch

from mujou._core import _native
from mujou._core.codec import schema, to_core
from mujou._core.driver import Driver
from mujou._core.port import AnyPort, AnySlot, Channel, Port, Wire
from mujou._core.spec import MessageSpec
from mujou._core.tree import Tree


@final
@dataclass(frozen=True, slots=True)
class Held:
    """A slot of the run and the spec of the messages it holds."""

    slot: AnySlot
    spec: MessageSpec[Any]


# The slots of one port, as the state tree keys them: "out" (the outbox of a port
# that sends), "to" (its slot for each peer, by the peer's path) and "in" (the
# slot the host writes).
type PortSlots = dict[str, Held | dict[str, Held]]


@final
@dataclass(frozen=True, slots=True)
class Wiring:
    """The slots and the drivers of one run of ``tree``."""

    tree: Tree
    group: _native.InputGroup
    # The outlet of each port that sends, which the host reads.
    outlets: Mapping[AnyPort, _native.Outlet]
    # The slot of each port that the host writes.
    inputs: Mapping[AnyPort, _native.InputSlot]
    # The slots of each port of each node: node path -> attribute -> slots.
    slots: Mapping[str, Mapping[str, PortSlots]]
    # The driver of each node, in the order of tree.members.
    drivers: tuple[Driver, ...]

    def write(self, values: Mapping[Any, object]) -> _native.Timestamp:
        """Write ``values`` to the ports the host writes, all at once, and return
        the time they bear.

        Raises:
            ValueError: A port is not one that the host writes in this run, or
                a shape or a dtype differs from the spec.
            TypeError: A message is not a tensor, or a dict of tensors with the
                spec's keys.
        """
        return self.group.write(
            [self._item(port, value) for port, value in values.items()]
        )

    def _item(
        self, port: AnyPort, value: object
    ) -> tuple[_native.InputSlot, list[torch.Tensor]]:
        slot = self.inputs.get(port)
        label = self.tree.label(port)
        if slot is None:
            raise ValueError(
                f"the host does not write {label}: it writes an unconnected Port "
                "of this run that receives"
            )
        spec = cast("MessageSpec[Any]", port.recv_spec)  # it receives
        return slot, to_core(spec, value, label)


def wire(tree: Tree, clock: _native.ReadOnlyTimer) -> Wiring:
    """Lay out the slots of ``tree`` on ``clock``, with the ``initial`` value of
    each port that has one, and make the drivers of its nodes.

    Raises:
        TypeError, ValueError: An ``initial`` value does not match the send
            spec of its port (as ``Port.send`` raises).
    """
    group = _native.InputGroup(clock=clock)
    outlets = {
        port: _outlet(tree, port, clock)
        for port in tree.owners
        if port.send_spec is not None
    }
    inputs = {
        port: group.input(
            schema(cast("MessageSpec[Any]", port.recv_spec)), tree.label(port)
        )
        for port in tree.owners
        if _written_by_the_host(port)
    }
    slots = {
        member.path: {
            attribute: _slots(tree, port, outlets, inputs)
            for attribute, port in member.ports.items()
        }
        for member in tree.members
    }
    drivers = tuple(
        Driver(
            member.node,
            clock,
            {
                port: _wire(tree, port, member.node.device, outlets, inputs)
                for port in member.ports.values()
            },
        )
        for member in tree.members
    )
    return Wiring(tree, group, outlets, inputs, slots, drivers)


def _written_by_the_host(port: AnyPort) -> bool:
    """An unconnected ``Port`` that receives reads what the host writes."""
    return isinstance(port, Port) and port.recv_spec is not None and not port.channels


def _outlet(tree: Tree, port: AnyPort, clock: _native.ReadOnlyTimer) -> _native.Outlet:
    """The outlet of ``port``, a port that sends, holding its ``initial`` value
    (without a time: the port has not sent it) or zeros."""
    spec = cast("MessageSpec[Any]", port.send_spec)
    label = tree.label(port)
    outlet = _native.Outlet(schema(spec), len(port.channels), label, clock=clock)
    if port.initial is not None:
        values = to_core(spec, port.initial, label)
        for slot in (outlet.outbox, *_channels(outlet, port)):
            slot.restore(values, None)
    return outlet


def _channels(outlet: _native.Outlet, port: AnyPort) -> list[_native.Slot]:
    """The slots of ``outlet``, the outlet of ``port``, for each of its channels,
    in their order."""
    return [outlet.channel(k) for k in range(len(port.channels))]


def _slots(
    tree: Tree,
    port: AnyPort,
    outlets: Mapping[AnyPort, _native.Outlet],
    inputs: Mapping[AnyPort, _native.InputSlot],
) -> PortSlots:
    """The slots of ``port``, as the state tree keys them."""
    slots: PortSlots = {}
    outlet = outlets.get(port)
    if outlet is not None:
        spec = cast("MessageSpec[Any]", port.send_spec)
        slots["out"] = Held(outlet.outbox, spec)
        slots["to"] = {
            tree.path(channel.other(port)): Held(slot, spec)
            for channel, slot in zip(
                port.channels, _channels(outlet, port), strict=True
            )
        }
    if port in inputs:
        slots["in"] = Held(inputs[port], cast("MessageSpec[Any]", port.recv_spec))
    return slots


def _wire(
    tree: Tree,
    port: AnyPort,
    device: torch.device,
    outlets: Mapping[AnyPort, _native.Outlet],
    inputs: Mapping[AnyPort, _native.InputSlot],
) -> Wire:
    return Wire(
        label=tree.label(port),
        device=device,
        send_spec=port.send_spec,
        recv_spec=port.recv_spec,
        outlet=outlets.get(port),
        sources=_sources(port, outlets, inputs),
        targets={channel: k for k, channel in enumerate(port.channels)},
    )


def _sources(
    port: AnyPort,
    outlets: Mapping[AnyPort, _native.Outlet],
    inputs: Mapping[AnyPort, _native.InputSlot],
) -> tuple[tuple[Channel | None, AnySlot], ...]:
    """The slots ``port`` reads, as ``Wire.sources`` holds them: what the host
    writes, or the slot of each peer for its channel to ``port``."""
    if port in inputs:
        return ((None, inputs[port]),)
    if port.recv_spec is None:
        return ()
    sources: list[tuple[Channel | None, AnySlot]] = []
    for channel in port.channels:
        peer = channel.other(port)
        # Each end writes its k-th channel slot for its k-th channel.
        k = peer.channels.index(channel)
        sources.append((channel, outlets[peer].channel(k)))
    return tuple(sources)
