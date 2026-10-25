"""The slots of a run, laid out for its tree, and the host's way to them
(``specs/network.md``, ``specs/execution-model.md``).

``wire`` lays out, on the run's clock, the slots of every port of the tree: an
``Outlet`` for a port that sends (its outbox, which the host reads, and one
slot per channel), and a slot in the run's ``InputGroup`` for an unconnected
``Port`` that receives, which the host writes. It gives each port the wire to
those slots that its node reads and writes through (``port.py``), and checks
what the host writes and reads (``Wiring.write`` and ``Wiring.outbox``). It
makes only new objects: the nodes and the ports stay as they are.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast, final

import torch

from mujou import _native
from mujou.codec import schema, to_core
from mujou.port import AnyPort, AnySlot, Channel, Port, Wire
from mujou.spec import MessageSpec
from mujou.tree import Tree


@final
@dataclass(frozen=True, slots=True)
class Wiring:
    """The slots of one run of ``tree``."""

    tree: Tree
    group: _native.InputGroup
    # The outlet of each port that sends, which the host reads.
    outlets: Mapping[AnyPort, _native.Outlet]
    # The slot of each port that the host writes.
    inputs: Mapping[AnyPort, _native.InputSlot]
    # The wires of the ports of each node, in the order of tree.members.
    wires: tuple[Mapping[AnyPort, Wire], ...]
    # For each node, in the same order: the CUDA devices other than its own
    # where the core holds what its peers send (not what the host writes, on a
    # device known only when it writes).
    read_devices: tuple[frozenset[torch.device], ...]

    def write(self, values: Mapping[Any, object]) -> _native.Timestamp:
        """Write ``values`` to the ports the host writes, all at once, and return
        the time they bear.

        Raises:
            ValueError: A port is not one that the host writes in this run, or
                a shape or a dtype differs from the spec.
            TypeError: A value is not of its field's kind (a tensor, an
                ``int``, ...), or a dict message does not have the spec's keys.
            OverflowError: An ``int`` is out of the range of its ``IntSpec``.
        """
        return self.group.write(
            [self._item(port, value) for port, value in values.items()]
        )

    def outbox(self, port: AnyPort) -> tuple[_native.Slot, MessageSpec[Any]]:
        """The slot of ``port`` that the host reads, and the spec of its
        messages.

        Raises:
            ValueError: ``port`` is not a port of this run that sends.
        """
        outlet = self.outlets.get(port)
        if outlet is None:
            raise ValueError(
                f"the host does not read {self.tree.label(port)}: it reads a port "
                "of this run that sends"
            )
        return outlet.outbox, cast("MessageSpec[Any]", port.send_spec)  # it sends

    def channels(self, port: AnyPort) -> list[_native.Slot]:
        """The slots of ``port``, a port that sends, for each of its channels,
        in their order."""
        return _channels(self.outlets[port], port)

    def _item(
        self, port: AnyPort, value: object
    ) -> tuple[_native.InputSlot, list[object]]:
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
    each port that has one, and the wires of its ports.

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
        if written_by_the_host(port)
    }
    # The index of each channel of each port: each end writes its k-th channel
    # slot for its k-th channel.
    indices = {
        port: {channel: k for k, channel in enumerate(port.channels)}
        for port in tree.owners
    }
    device = {
        port: member.node.device
        for member in tree.members
        for port in member.ports.values()
    }
    # Where what each port that sends stays: its node's CUDA device, if a node
    # on CUDA reads it. Nodes on the CPU and the host read a copy in host
    # memory sooner, which the sender makes once.
    held = {
        port: device[port] if _held(port, device) else None
        for port in tree.owners
        if port.send_spec is not None
    }
    wires = tuple(
        {
            port: Wire(
                port=port,
                label=tree.label(port),
                device=member.node.device,
                cuda=held.get(port),
                outlet=outlets.get(port),
                sources=_sources(port, outlets, inputs, indices),
                targets=indices[port],
            )
            for port in member.ports.values()
        }
        for member in tree.members
    )
    read_devices = tuple(
        frozenset(
            peer
            for port in member.ports.values()
            if port.recv_spec is not None
            for channel in port.channels
            if (peer := held[channel.other(port)]) is not None
            and peer != member.node.device
        )
        for member in tree.members
    )
    return Wiring(tree, group, outlets, inputs, wires, read_devices)


def _held(port: AnyPort, device: Mapping[AnyPort, torch.device]) -> bool:
    """Whether the core holds what ``port`` sends on the CUDA device of its node:
    the node is on CUDA, and so is the node of a peer (``device`` is that of
    each port's node)."""
    return device[port].type == "cuda" and any(
        device[channel.other(port)].type == "cuda" for channel in port.channels
    )


def written_by_the_host(port: AnyPort) -> bool:
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


def _sources(
    port: AnyPort,
    outlets: Mapping[AnyPort, _native.Outlet],
    inputs: Mapping[AnyPort, _native.InputSlot],
    indices: Mapping[AnyPort, Mapping[Channel, int]],
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
        sources.append((channel, outlets[peer].channel(indices[peer][channel])))
    return tuple(sources)
