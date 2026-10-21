"""What a port of a running network reads and writes: its ``Binding`` to the
slots of the run (``specs/network.md``).

Values cross the boundary to the core as copies: a received message is a new
tensor, and a sent one is copied into the core. The data a message carries stays
in the core: a port passes it on as the ``Carry`` it last received, and copies it
only when ``carried`` reads it.
"""

from collections.abc import Sequence
from typing import Any, cast, final

import torch

from mujou._core import _native
from mujou._core.hooks import current_node
from mujou._core.node import Node
from mujou._core.port import Channel, Port
from mujou._core.spec import Spec, TensorSpec, host_tensors, join

# Any (here and below): ports and messages of every type are handled alike.
type AnyPort = Port[Any, Any, Any, Any, Any]
type AnySlot = _native.Slot | _native.InputSlot


def as_spec(spec: object) -> Spec | None:
    """Type the ``send_spec`` or ``recv_spec`` of a port as a ``Spec``."""
    # Port accepts only TensorSpec, DictSpec or None (checked when it is made).
    return cast(Spec | None, spec)


def views_value(
    spec: Spec, views: Sequence[_native.BufferView], device: torch.device
) -> object:
    """Copy ``views`` into new tensors on ``device`` and build the message."""
    return join(spec, [torch.from_dlpack(v).to(device, copy=True) for v in views])


def message_value(spec: Spec, message: _native.Message, device: torch.device) -> object:
    """Copy the value of ``message`` into new tensors on ``device`` and build the
    message."""
    return views_value(spec, message.buffers, device)


@final
class Binding:
    """What one port of a running network reads and writes.

    ``Port.recv`` / ``send`` / ``carried`` delegate here once the network is
    built.
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
        self._send_carry_spec = as_spec(port.send_carry_spec)
        self._recv_carry_spec = as_spec(port.recv_carry_spec)
        self._multi = port.multi
        self._outlet = outlet
        # (channel, slot) in connection order; a single-peer port has exactly one,
        # whose channel is None while unconnected (the host writes it).
        self._sources = list(sources)
        self._indices = {channel: k for k, channel in enumerate(port.channels)}
        # The carry of the message last received, for a port with a recv_carry
        # (which has one peer). Until the first recv, that of the message it
        # would have received at the start: the source's initial value.
        self._carry: _native.Carry | None = None
        if self._recv_carry_spec is not None:
            ((_, source),) = self._sources
            self._carry = source.read().carry
        # The binding of the port whose carry a send passes on, set by relay_from.
        self._relay: Binding | None = None

    def relay_from(self, source: Binding) -> None:
        """Pass on the carry ``source`` last received with every send (the
        ``carry_source`` of the port; the run's ``Session`` binds it once every
        port is bound)."""
        self._relay = source

    @property
    def carry(self) -> _native.Carry | None:
        """The carry of the message last received, or ``None`` for a port without
        a ``recv_carry`` (for the state tree)."""
        return self._carry

    @carry.setter
    def carry(self, carry: _native.Carry) -> None:
        # Only a restore sets it, while no node runs.
        self._carry = carry

    def recv(self, with_time: bool) -> object:
        spec = self._receiving()
        device = self._owner.device
        received: dict[Channel | None, object] = {}
        for channel, slot in self._sources:
            message = slot.read()
            value = message_value(spec, message, device)
            received[channel] = (value, message.timestamp) if with_time else value
            if self._carry is not None:
                # The very message of the value: a port with a recv_carry has one
                # source, so this is the only message read.
                self._carry = message.carry
        if self._multi:
            return received
        (value,) = received.values()
        return value

    def has_received(self) -> object:
        self._receiving()
        received = {channel: slot.read().sent for channel, slot in self._sources}
        if self._multi:
            return received
        (sent,) = received.values()
        return sent

    def carried(self, key: str | None) -> object:
        self._check_owner("reads the carried data of")
        spec = self._recv_carry_spec
        carry = self._carry
        if spec is None or carry is None:
            raise TypeError(f"the port {self._name!r} has no recv_carry")
        device = self._owner.device
        if key is None:
            return views_value(spec, carry.buffers, device)
        if isinstance(spec, TensorSpec):
            raise TypeError(
                f"the port {self._name!r} carries a tensor, which has no key {key!r}"
            )
        keys = list(spec.specs)
        if key not in keys:
            raise KeyError(
                f"the port {self._name!r} carries no key {key!r}; its keys are {keys}"
            )
        view = carry.buffers[keys.index(key)]
        return torch.from_dlpack(view).to(device, copy=True)

    def send(self, msg: object, carry: object) -> None:
        outlet = self._sending()
        tensors, relay = self._message(msg, carry)
        outlet.send(tensors, relay=relay)

    def send_to(self, channel: Channel, msg: object, carry: object) -> None:
        outlet = self._sending()
        index = self._indices.get(channel)
        if index is None:
            raise ValueError(f"{channel!r} is not a channel of {self._name!r}")
        tensors, relay = self._message(msg, carry)
        outlet.send_to(index, tensors, relay=relay)

    def _check_owner(self, verb: str) -> None:
        # A port is read and written only by its owner: this keeps each slot to
        # one writer, and a node to the slots of its own ports (locality). The
        # node, not the thread: a worker runs other nodes too.
        if current_node() is not self._owner:
            raise RuntimeError(
                f"only the node that owns the port {self._name!r} {verb} it, from its "
                "own hooks while the network runs (state_dict runs on another thread)"
            )

    def _receiving(self) -> Spec:
        self._check_owner("receives on")
        if self._recv_spec is None:
            raise TypeError(f"the port {self._name!r} receives no messages")
        return self._recv_spec

    def _sending(self) -> _native.Outlet:
        self._check_owner("sends on")
        if self._outlet is None:
            raise TypeError(f"the port {self._name!r} sends no messages")
        return self._outlet

    def _message(
        self, msg: object, carry: object
    ) -> tuple[list[torch.Tensor], _native.Carry | None]:
        """What the outlet sends for ``msg`` and ``carry``: the tensors in host
        memory (the value's, then the carried data's unless it is passed on) and
        the carry to pass on, if any."""
        spec = cast(Spec, self._send_spec)  # an outlet exists only with a send spec
        tensors = host_tensors(spec, msg, where=self._name)
        if self._relay is not None:
            if carry is not None:
                raise TypeError(
                    f"the port {self._name!r} passes on the data its send_carry "
                    "port received, so it takes no carry"
                )
            # The relay's port has a recv_carry, so it holds a carry.
            return tensors, self._relay._carry
        if self._send_carry_spec is None:
            if carry is not None:
                raise TypeError(f"the port {self._name!r} carries no data")
            return tensors, None
        if carry is None:
            raise TypeError(
                f"the port {self._name!r} carries {self._send_carry_spec!r}; pass "
                "it as carry="
            )
        where = f"{self._name} carry"
        return tensors + host_tensors(self._send_carry_spec, carry, where=where), None
