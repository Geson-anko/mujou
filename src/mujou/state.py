"""The state tree of a run: what ``Run.state_dict`` returns and
``mujou.run(..., state=...)`` restores (``specs/execution-model.md``).

The tree is a plain ``dict`` of copies on the CPU, which share no memory with
the run; ``Run.state_dict`` documents its shape, which this module alone makes.
Its ports mirror the slots that ``wiring.wire`` lays out (``_slots``):
``capture`` reads them and ``restore`` writes them.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, final

import torch

from mujou.codec import peek, times_from_ns, times_to_ns, to_core
from mujou.node import map_tensors
from mujou.port import AnyPort, AnySlot
from mujou.spec import MessageSpec
from mujou.wiring import Wiring


@final
@dataclass(frozen=True, slots=True)
class _Held:
    """A slot of the run and the spec of the messages it holds."""

    slot: AnySlot
    spec: MessageSpec[Any]


# The slots of one port, as the tree keys them: "out" (the outbox of a port that
# sends), "to" (its slot for each peer, by the path of the peer's port) and "in"
# (the slot the host writes).
type _PortSlots = dict[str, _Held | dict[str, _Held]]


def capture(
    wiring: Wiring,
    states: Sequence[object],
    last_update_ns: Sequence[int],
    time_ns: int,
    time_scale: float,
) -> dict[str, Any]:
    """The tree of a paused or ended run at ``time_ns`` on its timer, from the
    copies of the nodes' states (``Driver.capture``) and their last update
    times, in the order of the tree's members. The values of the slots that
    the core holds on a CUDA device come uncopied, so as not to lengthen the
    pause: ``cpu`` moves them and the states to the CPU afterwards."""
    return {
        "timer": {"time_ns": time_ns, "time_scale": time_scale},
        "nodes": {
            member.path: {
                "state": state,
                "last_update_ns": updated_ns,
                "ports": {
                    attribute: _read(_slots(wiring, port))
                    for attribute, port in member.ports.items()
                },
            }
            for member, state, updated_ns in zip(
                wiring.tree.members, states, last_update_ns, strict=True
            )
        },
    }


def cpu(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with the tensors of every node's state and of its ports on the
    CPU, as copies (the rest of the tree is on the CPU already)."""
    for node in tree["nodes"].values():
        node["state"] = map_tensors(node["state"], torch.Tensor.cpu)
        node["ports"] = map_tensors(node["ports"], torch.Tensor.cpu)
    return tree


def saved_time(tree: Mapping[str, Any]) -> tuple[int, float]:
    """The time on the timer and the time scale at which ``tree`` was taken,
    after checking the top of the tree (``restore`` checks its nodes).

    Raises:
        ValueError: The tree does not have the entries of its timer, or has
            other entries than ``timer`` and ``nodes``.
    """
    expected = {"timer": {"time_ns": None, "time_scale": None}, "nodes": None}
    _match(_mismatches(expected, tree, ""))
    saved = tree["timer"]
    return saved["time_ns"], saved["time_scale"]


def restore(wiring: Wiring, tree: Mapping[str, Any]) -> list[tuple[object, int]]:
    """Check the nodes of ``tree`` against ``wiring`` and restore its slots;
    return the state and the last update time of each node, in the order of the
    tree's members. ``saved_time`` has checked the top of the tree.

    Raises:
        ValueError: The tree does not have the nodes, the ports or the slots of
            this run, or a value restored to a slot does not match its spec.
        TypeError: A value restored to a slot is not of its structure.
    """
    nodes: Mapping[str, Any] = tree["nodes"]
    slots = {
        member.path: {
            attribute: _slots(wiring, port) for attribute, port in member.ports.items()
        }
        for member in wiring.tree.members
    }
    expected = {
        path: {"state": None, "last_update_ns": None, "ports": ports}
        for path, ports in slots.items()
    }
    _match(_mismatches(expected, nodes, "nodes"))
    for path, ports in slots.items():
        for attribute, port_slots in ports.items():
            _write(port_slots, nodes[path]["ports"][attribute])
    return [
        (nodes[member.path]["state"], nodes[member.path]["last_update_ns"])
        for member in wiring.tree.members
    ]


def _slots(wiring: Wiring, port: AnyPort) -> _PortSlots:
    """The slots of ``port``, as the tree keys them."""
    slots: _PortSlots = {}
    outlet = wiring.outlets.get(port)
    if outlet is not None:
        spec = cast("MessageSpec[Any]", port.send_spec)
        slots["out"] = _Held(outlet.outbox, spec)
        slots["to"] = {
            wiring.tree.path(channel.other(port)): _Held(slot, spec)
            for channel, slot in zip(port.channels, wiring.channels(port), strict=True)
        }
    slot = wiring.inputs.get(port)
    if slot is not None:
        slots["in"] = _Held(slot, cast("MessageSpec[Any]", port.recv_spec))
    return slots


def _read(slots: _PortSlots | dict[str, _Held]) -> dict[str, Any]:
    return {
        key: _entry(held) if isinstance(held, _Held) else _read(held)
        for key, held in slots.items()
    }


def _entry(held: _Held) -> dict[str, Any]:
    message = held.slot.read()
    time = message.timestamp
    return {
        "value": times_to_ns(held.spec, peek(held.spec, message.values)),
        "time_ns": None if time is None else time.ns,
    }


def _write(slots: _PortSlots | dict[str, _Held], entries: Mapping[str, Any]) -> None:
    for key, held in slots.items():
        if isinstance(held, _Held):
            entry = entries[key]
            value = times_from_ns(held.spec, entry["value"])
            values = to_core(held.spec, value, held.slot.name)
            held.slot.restore(values, entry["time_ns"])
        else:
            _write(held, entries[key])


def _match(mismatches: Iterator[str]) -> None:
    """Raise the ``ValueError`` of a tree that does not match, if ``mismatches``
    lists anything."""
    found = list(mismatches)
    if found:
        raise ValueError("the state tree does not match this run: " + ", ".join(found))


def _mismatches(
    expected: Mapping[str, object], given: Mapping[str, object], at: str
) -> Iterator[str]:
    """The keys missing from ``given`` and those it has in excess, at any depth
    where ``expected`` holds a mapping; ``at`` is where ``given`` is in the tree
    (``""`` for its top)."""
    for key, value in expected.items():
        if key not in given:
            yield f"missing {_where(at, key)}"
        elif isinstance(value, Mapping):
            inner = cast(Mapping[str, object], value)
            given_inner = cast(Mapping[str, object], given[key])
            yield from _mismatches(inner, given_inner, _where(at, key))
    for key in given:
        if key not in expected:
            yield f"unexpected {_where(at, key)}"


def _where(at: str, key: str) -> str:
    return f"{at}[{key!r}]" if at else key
