"""The state tree of a run: what ``Run.state_dict`` returns and
``mujou.run(..., state=...)`` restores (``specs/execution-model.md``).

The tree is a plain ``dict`` of copies on the CPU, which share no memory with
the run; ``Run.state_dict`` documents its shape. Its ports mirror the slots that
``wiring.wire`` lays out: ``capture`` reads them and ``restore`` writes them.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final, cast

import torch

from mujou._core.codec import from_core, to_core
from mujou._core.wiring import Held, PortSlots, Wiring

# Where the tree holds its tensors.
_CPU: Final = torch.device("cpu")


def capture(
    wiring: Wiring,
    states: Sequence[object],
    last_update_ns: Sequence[int],
    time_ns: int,
    time_scale: float,
) -> dict[str, Any]:
    """The tree of a paused or ended run at ``time_ns`` on its timer, from the
    copies of the nodes' states (``Driver.capture``) and their last update
    times, in the order of the tree's members. ``cpu`` moves the states to the
    CPU afterwards."""
    return {
        "timer": {"time_ns": time_ns, "time_scale": time_scale},
        "nodes": {
            member.path: {
                "state": state,
                "last_update_ns": updated_ns,
                "ports": {
                    attribute: _read(slots)
                    for attribute, slots in wiring.slots[member.path].items()
                },
            }
            for member, state, updated_ns in zip(
                wiring.tree.members, states, last_update_ns, strict=True
            )
        },
    }


def cpu(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` with the tensors of every node's state on the CPU, moved in the
    tree's own containers (the rest of the tree is on the CPU already)."""
    for node in tree["nodes"].values():
        node["state"] = _cpu(node["state"])
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
    expected = {
        path: {"state": None, "last_update_ns": None, "ports": ports}
        for path, ports in wiring.slots.items()
    }
    _match(_mismatches(expected, nodes, "nodes"))
    for path, ports in wiring.slots.items():
        for attribute, slots in ports.items():
            _write(slots, nodes[path]["ports"][attribute])
    return [
        (nodes[member.path]["state"], nodes[member.path]["last_update_ns"])
        for member in wiring.tree.members
    ]


def _read(slots: PortSlots | dict[str, Held]) -> dict[str, Any]:
    return {
        key: _entry(held) if isinstance(held, Held) else _read(held)
        for key, held in slots.items()
    }


def _entry(held: Held) -> dict[str, Any]:
    message = held.slot.read()
    time = message.timestamp
    return {
        "value": from_core(held.spec, message.values, _CPU),
        "time_ns": None if time is None else time.ns,
    }


def _write(slots: PortSlots | dict[str, Held], entries: Mapping[str, Any]) -> None:
    for key, held in slots.items():
        if isinstance(held, Held):
            entry = entries[key]
            values = to_core(held.spec, entry["value"], held.slot.name)
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


def _cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        for key, item in items.items():
            items[key] = _cpu(item)
        return items
    if isinstance(value, list):
        elements = cast(list[object], value)
        elements[:] = [_cpu(element) for element in elements]
        return elements
    if type(value) is tuple:
        return tuple(_cpu(element) for element in cast(tuple[object, ...], value))
    return value
