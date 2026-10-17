"""The state tree of a run: what ``Run.state_dict`` returns and
``mujou.run(..., state=...)`` restores (``specs/execution-model.md``).

This module is the one place that defines the tree::

    {
        "nodes": {path: the node's state_dict(), ...},
        "slots": {slot name: (value, time_ns), ...},
        "last_update_ns": {path: int, ...},
        "timer": {"time_ns": int, "scale": float},
    }

Values are copies: the tree shares no memory with the run, in either direction.
The tree is a ``StateDict``, a ``dict`` that also moves all its tensors to a device.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, Final, Self, cast, override

import torch

from mujou._core.binding import AnySlot, message_value
from mujou._core.spec import Spec, host_tensors

# Where the tree holds the values of the slots.
_CPU: Final = torch.device("cpu")


class StateDict(dict[str, Any]):
    """The state tree of a run, as ``Run.state_dict`` returns it.

    A ``dict``: index it, pickle it, save it with ``torch.save`` and pass it to
    ``mujou.run(..., state=tree)`` as the dict it is. It adds ``to`` and ``cpu``,
    which move every tensor of the tree to one device, for example to save the
    state of nodes on a GPU as CPU tensors::

        torch.save(run.state_dict().cpu(), "state.pt")

    It pickles as ``mujou.StateDict``, and ``import mujou`` allows it in
    ``torch.load`` with the default ``weights_only=True``: a tree saved with
    ``torch.save`` loads back as a ``StateDict``.
    """

    def to(self, device: torch.device | str | int) -> Self:
        """Return a new tree with every tensor of this one on ``device``.

        Tensors are found in the dicts (``OrderedDict`` and other subclasses
        too, such as the ``state_dict`` of an ``nn.Module``), lists and tuples of
        the tree, at any depth. Those containers are new, of the same type; any
        other value is shared with this tree. As with ``torch.Tensor.to``, a
        tensor already on ``device`` is shared, not copied. A tensor that
        appears twice in this tree appears as one tensor in the new tree.

        This tree is not changed.

        Args:
            device: The device to move the tensors to, as ``torch.device``
                takes it (``"cpu"``, ``"cuda:0"``, ...).
        """
        return _moved(self, torch.device(device), {})

    def cpu(self) -> Self:
        """Return a new tree with every tensor of this one on the CPU (``to``)."""
        return self.to(_CPU)

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[dict[str, Any]]]:
        # A dict subclass pickles its items with SETITEMS, which the
        # weights_only unpickler of torch.load allows only on a plain dict. Pickle
        # the items as a plain dict and rebuild from it instead.
        return (type(self), (dict(self),))


# Pickled by its public name, so that saved trees keep loading when the module of
# the implementation moves.
StateDict.__module__ = "mujou"
torch.serialization.add_safe_globals([StateDict])  # pyright: ignore[reportUnknownMemberType]


def _moved[T](value: T, device: torch.device, memo: dict[int, object]) -> T:
    """``value`` with every tensor in it moved to ``device`` (``StateDict.to``).

    ``memo`` maps the id of each tensor or container already moved to its result.
    """
    found = id(value)  # before the checks below narrow the type of value
    if found in memo:
        return cast(T, memo[found])
    result: object
    if isinstance(value, torch.Tensor):
        result = value.to(device)
    elif isinstance(value, dict):
        items = cast(dict[object, object], value)
        moved = copy.copy(items)  # the same type, with its attributes
        for key, item in items.items():
            moved[key] = _moved(item, device, memo)
        result = moved
    elif type(value) is list:
        result = [_moved(item, device, memo) for item in cast(list[object], value)]
    elif type(value) is tuple:
        items = cast(tuple[object, ...], value)
        result = tuple(_moved(item, device, memo) for item in items)
    else:
        return value
    memo[found] = result
    return cast(T, result)


def make_tree(
    paths: Sequence[str],
    states: Sequence[object],
    slots: Mapping[str, tuple[AnySlot, Spec]],
    last_update_ns: Sequence[int],
    timer: Mapping[str, int | float],
) -> StateDict:
    """Copy the state of a paused or ended run into a tree.

    ``states`` and ``last_update_ns`` are per node, in the order of ``paths``. The
    copies queue on the calling thread's current streams; synchronize them before
    the nodes update in place again.
    """
    values: dict[str, tuple[object, int]] = {}
    for name, (slot, spec) in slots.items():
        message = slot.read()
        values[name] = (message_value(spec, message, _CPU), message.timestamp.ns)
    return StateDict(
        {
            "nodes": {
                path: copy.deepcopy(state)
                for path, state in zip(paths, states, strict=True)
            },
            "slots": values,
            "last_update_ns": dict(zip(paths, last_update_ns, strict=True)),
            "timer": dict(timer),
        }
    )


def read_tree(
    state: Mapping[str, Any],
    paths: Sequence[str],
    slots: Mapping[str, tuple[AnySlot, Spec]],
) -> tuple[list[tuple[AnySlot, list[torch.Tensor], int]], dict[str, Any]]:
    """Check ``state`` against a run of the nodes at ``paths`` with ``slots``, and
    copy it, changing nothing.

    Returns ``(slot, tensors in host memory, time_ns)`` for each slot to restore,
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
) -> list[tuple[AnySlot, list[torch.Tensor], int]]:
    """Check each slot's ``(value, time_ns)`` in the tree and copy the value to
    the host, without restoring anything yet."""
    restores: list[tuple[AnySlot, list[torch.Tensor], int]] = []
    for name, (slot, spec) in slots.items():
        entry = entries[name]
        if not isinstance(entry, tuple) or len(cast(tuple[object, ...], entry)) != 2:
            raise ValueError(f"the state tree's slot {name!r} is not (value, time_ns)")
        value, time_ns = cast(tuple[object, object], entry)
        if not _is_int(time_ns):
            raise ValueError(
                f"the state tree's slot {name!r} has a time that is not an int"
            )
        restores.append(
            (slot, host_tensors(spec, value, where=name), cast(int, time_ns))
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
