"""The state tree of a run: what ``Run.state_dict`` returns and
``mujou.run(..., state=...)`` restores (``specs/execution-model.md``).

This module is the one place that defines the tree::

    {
        "nodes": {path: the node's state_dict(), ...},
        "slots": {slot name: (value, time_ns, sent), ...},
        "last_update_ns": {path: int, ...},
        "timer": {"time_ns": int, "scale": float},
    }

Values are copies: the tree shares no memory with the run, in either direction.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Final, cast

import torch

from mujou._core.binding import AnySlot, message_value
from mujou._core.spec import Spec, host_tensors

# Where the tree holds the values of the slots.
_CPU: Final = torch.device("cpu")


def make_tree(
    paths: Sequence[str],
    states: Sequence[object],
    slots: Mapping[str, tuple[AnySlot, Spec]],
    last_update_ns: Sequence[int],
    timer: Mapping[str, int | float],
) -> dict[str, Any]:
    """Copy the state of a paused or ended run into a tree.

    ``states`` and ``last_update_ns`` are per node, in the order of ``paths``. The
    copies queue on the calling thread's current streams; synchronize them before
    the nodes update in place again.
    """
    values: dict[str, tuple[object, int, bool]] = {}
    for name, (slot, spec) in slots.items():
        message = slot.read()
        values[name] = (
            message_value(spec, message, _CPU),
            message.timestamp.ns,
            message.sent,
        )
    return {
        "nodes": {
            path: copy.deepcopy(state)
            for path, state in zip(paths, states, strict=True)
        },
        "slots": values,
        "last_update_ns": dict(zip(paths, last_update_ns, strict=True)),
        "timer": dict(timer),
    }


def read_tree(
    state: Mapping[str, Any],
    paths: Sequence[str],
    slots: Mapping[str, tuple[AnySlot, Spec]],
) -> tuple[list[tuple[AnySlot, list[torch.Tensor], int, bool]], dict[str, Any]]:
    """Check ``state`` against a run of the nodes at ``paths`` with ``slots``, and
    copy it, changing nothing.

    Returns ``(slot, tensors in host memory, time_ns, sent)`` for each slot to
    restore,
    and what ``Runtime.start`` restores: the nodes' states (copies, so that the
    nodes' updates in place never reach ``state``), their last update times and
    the timer. The copies queue on the calling thread's current streams.

    Raises:
        ValueError: The tree does not match the run.
    """
    for key in ("nodes", "slots", "last_update_ns", "timer"):
        if key not in state:
            raise ValueError(f"the state tree has no {key!r}")
    nodes = cast(Mapping[str, object], state["nodes"])
    entries = cast(Mapping[str, object], state["slots"])
    last_update_ns = cast(Mapping[str, object], state["last_update_ns"])
    timer = cast(Mapping[str, object], state["timer"])
    _same_keys("nodes", nodes, paths)
    _same_keys("slots", entries, slots)
    _same_keys("last_update_ns", last_update_ns, paths)
    _same_keys("timer", timer, ("time_ns", "scale"))
    for path in paths:
        if not _is_int(last_update_ns[path]):
            raise ValueError(f"the state tree's last_update_ns[{path!r}] is not an int")
    if not _is_int(timer["time_ns"]) or not isinstance(timer["scale"], int | float):
        raise ValueError("the state tree's timer must have an int time_ns and a scale")
    restores = _slot_restores(entries, slots)
    restored = {
        "states": [copy.deepcopy(nodes[path]) for path in paths],
        "last_update_ns": [cast(int, last_update_ns[path]) for path in paths],
        "timer_state": {
            "time_ns": cast(int, timer["time_ns"]),
            "scale": cast(float, timer["scale"]),
        },
    }
    return restores, restored


def _slot_restores(
    entries: Mapping[str, object], slots: Mapping[str, tuple[AnySlot, Spec]]
) -> list[tuple[AnySlot, list[torch.Tensor], int, bool]]:
    """Check each slot's ``(value, time_ns, sent)`` in the tree and copy the value
    to the host, without restoring anything yet."""
    restores: list[tuple[AnySlot, list[torch.Tensor], int, bool]] = []
    for name, (slot, spec) in slots.items():
        entry = entries[name]
        if not isinstance(entry, tuple) or len(cast(tuple[object, ...], entry)) != 3:
            raise ValueError(
                f"the state tree's slot {name!r} is not (value, time_ns, sent)"
            )
        value, time_ns, sent = cast(tuple[object, object, object], entry)
        if not _is_int(time_ns):
            raise ValueError(
                f"the state tree's slot {name!r} has a time that is not an int"
            )
        if not isinstance(sent, bool):
            raise ValueError(
                f"the state tree's slot {name!r} has a sent flag that is not a bool"
            )
        restores.append(
            (slot, host_tensors(spec, value, where=name), cast(int, time_ns), sent)
        )
    return restores


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _same_keys(
    what: str, given: Mapping[str, object], expected: Sequence[str] | Mapping[str, Any]
) -> None:
    missing = [k for k in expected if k not in given]
    unknown = [k for k in given if k not in expected]
    if missing or unknown:
        raise ValueError(
            f"the state tree's {what} do not match this network: missing {missing}, "
            f"unknown {unknown}"
        )
