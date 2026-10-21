"""The node whose hook the calling thread runs: ``running()`` is the one entry
to it, over the bindings' thread-local ``_native.current()``
(``specs/execution-model.md``).

The ports of a node (``port.py``), its timer (``Node.timer``) and the host's
methods (``execution.py``) ask it whether they are called where they may be: a port
and the timer only in the hooks of their node, the host's methods only outside
every hook.
"""

from collections.abc import Mapping
from typing import Any, Protocol, cast

from mujou import _native


class Running(Protocol):
    """What a hook sees of the node it runs as: the node's driver
    (``driver.py``)."""

    @property
    def node(self) -> object: ...

    @property
    def wires(self) -> Mapping[object, Any]:
        """The wire (``port.Wire``) of each port of the node: none in its
        ``state_dict``."""
        ...

    @property
    def clock(self) -> _native.ReadOnlyTimer: ...


def running() -> Running | None:
    """The driver of the node whose hook the calling thread runs, or ``None``
    outside the hooks."""
    return cast("Running | None", _native.current())


def outside(name: str, hint: str) -> RuntimeError:
    """The error for ``name``, a port or a node's timer, used outside the hooks
    of its node; ``hint`` says what to do instead."""
    return RuntimeError(
        f"{name} is used only in the hooks of its node while it runs; {hint}"
    )
