"""Whether the calling thread runs a node's hook, and which node, for the
host-only operations to refuse a node and for a port to tell its owner
(``specs/execution-model.md``, ``specs/network.md``).

A thread that runs nodes (a worker of a run's pool, or a dedicated node's own
thread) is inside a hook from the first time it enters a node on; it runs one
node at a time, the one it entered last. The control thread, or after a normal
end the host's thread that called ``Run.state_dict``, is inside a hook only
while it calls a node's ``state_dict``, and runs no node.
"""

import contextlib
import threading
from collections.abc import Generator
from typing import Final


class _Hooks(threading.local):
    # Whether this thread is running a node's hook (a thread that runs nodes
    # always is).
    inside = False
    # The node this thread runs (the one it entered last), or None. ``object``:
    # this module is below the node's.
    node: object = None


_hooks: Final = _Hooks()


def in_hook() -> bool:
    """Whether the calling thread is running a node's hook."""
    return _hooks.inside


def enter_node(node: object) -> None:
    """Mark the calling thread, one that runs nodes, as running ``node`` from now
    on, and as inside a hook for good."""
    _hooks.inside = True
    _hooks.node = node


def current_node() -> object:
    """The node the calling thread runs, or ``None`` for a thread that runs
    none."""
    return _hooks.node


@contextlib.contextmanager
def running_hook() -> Generator[None]:
    """Mark the calling thread as inside a hook for the ``with`` block."""
    outer = _hooks.inside
    _hooks.inside = True
    try:
        yield
    finally:
        _hooks.inside = outer
