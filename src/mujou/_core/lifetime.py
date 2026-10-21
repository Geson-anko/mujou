"""The runs that are running, and their stop when the interpreter exits
(``specs/network.md``).

A running network keeps running when the host drops it, until the interpreter
exits: the runtime's threads are not Python threads, and must not call into a
finalizing interpreter.
"""

import atexit
import contextlib
import math
import threading
from typing import Final, Protocol

from mujou._core.errors import NetworkError


class Stoppable(Protocol):
    """A run as this module keeps it."""

    def shutdown(self, timeout: float) -> None:
        """Stop the run, waiting up to ``timeout`` seconds."""
        ...


# The runs that have started and not yet been seen to end. Held strongly: a
# running network keeps running when the host drops it, until it is shut down
# here at the latest.
_RUNNING: Final[set[Stoppable]] = set()


def add(run: Stoppable) -> None:
    """Keep ``run`` until ``discard``, and stop it at the exit if still kept."""
    _RUNNING.add(run)
    _watch_exit()


def discard(run: Stoppable) -> None:
    """Stop keeping ``run`` (it has ended, or never started)."""
    _RUNNING.discard(run)


def _shutdown_all() -> None:
    # Stop every run even if one fails (e.g. Ctrl-C while waiting for its
    # on_stop), then raise the first failure.
    failure: BaseException | None = None
    for run in list(_RUNNING):
        try:
            run.shutdown(math.inf)
        except NetworkError:
            pass  # already logged, and the host has gone
        except BaseException as e:  # noqa: BLE001 - re-raised below
            failure = failure or e
    if failure is not None:
        raise failure


def _stop_at_exit() -> None:
    """Stop every running network once the main thread has ended.

    At the interpreter's exit, ``threading`` marks the main thread as ended
    before it joins the non-daemon threads, so this starts stopping the networks
    before that join and a thread iterating over ``subscribe`` ends with its
    network. That order is CPython's implementation, not its documented
    behavior; ``TestInterpreterExit`` in ``tests/mujou/_core/test_run.py`` guards
    it.
    """
    threading.main_thread().join()
    # On a failure the atexit call below tries again, and raises.
    with contextlib.suppress(Exception):
        _shutdown_all()


_watching = False
_watching_lock: Final = threading.Lock()


def _watch_exit() -> None:
    """Start the daemon thread of ``_stop_at_exit`` once, at the first run."""
    global _watching
    with _watching_lock:
        if _watching:
            return
        _watching = True
    threading.Thread(target=_stop_at_exit, name="mujou-exit", daemon=True).start()


# After the join of the non-daemon threads, wait until the networks have stopped
# (the daemon thread may still be stopping them) and stop those that started
# since.
atexit.register(_shutdown_all)
