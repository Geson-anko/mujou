"""What a port of a running network reads and writes: its ``Binding`` to the
slots of the run (``specs/network.md``).

Values cross the boundary to the core as copies: a received message is a new
tensor, and a sent one is copied into the core.
"""

import threading
from collections.abc import Sequence
from typing import Any, cast, final

import torch

from mujou._core import _native
from mujou._core.node import Node, thread_ident
from mujou._core.port import Channel, Port
from mujou._core.spec import Spec, host_tensors, join

# Any (here and below): ports and messages of every type are handled alike.
type AnyPort = Port[Any, Any, Any]
type AnySlot = _native.Slot | _native.InputSlot


def as_spec(spec: object) -> Spec | None:
    """Type the ``send_spec`` or ``recv_spec`` of a port as a ``Spec``."""
    # Port accepts only TensorSpec, DictSpec or None (checked when it is made).
    return cast(Spec | None, spec)


def message_value(spec: Spec, message: _native.Message, device: torch.device) -> object:
    """Copy ``message`` into new tensors on ``device`` and build the message."""
    return join(
        spec, [torch.from_dlpack(b).to(device, copy=True) for b in message.buffers]
    )


@final
class Binding:
    """What one port of a running network reads and writes.

    ``Port.recv`` / ``send`` delegate here once the network is built.
    """

    def __init__(
        self,
        owner: Node,
        name: str,
        port: AnyPort,
        outlet: _native.Outlet | None,
        sources: Sequence[tuple[Channel | None, AnySlot]],
    ) -> None:
        self._owner = owner
        # How errors name the port.
        self._name = name
        self._send_spec = as_spec(port.send_spec)
        self._recv_spec = as_spec(port.recv_spec)
        self._multi = port.multi
        self._outlet = outlet
        # (channel, slot) in connection order; a single-peer port has exactly one,
        # whose channel is None while unconnected (the host writes it).
        self._sources = list(sources)
        self._indices = {channel: k for k, channel in enumerate(port.channels)}

    def recv(self, with_time: bool) -> object:
        self._check_owner("receives on")
        spec = self._recv_spec
        if spec is None:
            raise TypeError(f"the port {self._name!r} receives no messages")
        device = self._owner.device
        received: dict[Channel | None, object] = {}
        for channel, slot in self._sources:
            message = slot.read()
            value = message_value(spec, message, device)
            received[channel] = (value, message.timestamp) if with_time else value
        if self._multi:
            return received
        (value,) = received.values()
        return value

    def has_received(self) -> object:
        self._check_owner("receives on")
        if self._recv_spec is None:
            raise TypeError(f"the port {self._name!r} receives no messages")
        received = {channel: slot.read().sent for channel, slot in self._sources}
        if self._multi:
            return received
        (sent,) = received.values()
        return sent

    def send(self, msg: object) -> None:
        outlet = self._sending()
        outlet.send(self._tensors(msg))

    def send_to(self, channel: Channel, msg: object) -> None:
        outlet = self._sending()
        index = self._indices.get(channel)
        if index is None:
            raise ValueError(f"{channel!r} is not a channel of {self._name!r}")
        outlet.send_to(index, self._tensors(msg))

    def _check_owner(self, verb: str) -> None:
        # A port is read and written only by its owner: this keeps each slot to
        # one writer, and a node to the slots of its own ports (locality).
        if threading.get_ident() != thread_ident(self._owner):
            raise RuntimeError(
                f"only the node that owns the port {self._name!r} {verb} it, from its "
                "own hooks while the network runs (state_dict runs on another thread)"
            )

    def _sending(self) -> _native.Outlet:
        self._check_owner("sends on")
        if self._outlet is None:
            raise TypeError(f"the port {self._name!r} sends no messages")
        return self._outlet

    def _tensors(self, msg: object) -> list[torch.Tensor]:
        spec = cast(Spec, self._send_spec)  # an outlet exists only with a send spec
        return host_tensors(spec, msg, where=self._name)
