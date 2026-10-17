"""Whether the calling thread runs a node's hook, for the host-only operations to
refuse a node (``specs/execution-model.md``).

A node's own thread is inside a hook from its first hook on; the control thread
is inside one only while it calls a node's ``state_dict``.
"""

import contextlib
import threading
from collections.abc import Generator
from typing import Final


class _Hooks(threading.local):
    # Whether this thread is running a node's hook (a node's own thread always is).
    inside = False


_hooks: Final = _Hooks()


def in_hook() -> bool:
    """Whether the calling thread is running a node's hook."""
    return _hooks.inside


def enter_node_thread() -> None:
    """Mark the calling thread, a node's own, as inside a hook from now on."""
    _hooks.inside = True


@contextlib.contextmanager
def running_hook() -> Generator[None]:
    """Mark the calling thread as inside a hook for the ``with`` block."""
    outer = _hooks.inside
    _hooks.inside = True
    try:
        yield
    finally:
        _hooks.inside = outer
