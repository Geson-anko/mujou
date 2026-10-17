"""The time of mujou.

Every run measures its nodes' ``dt`` on one timer, a time axis that can be scaled
and paused. By default every run shares the global timer, which the functions
below read; give a run a ``Timer`` of its own with ``mujou.run(..., timer=...)``.
Use these functions instead of the standard ``time`` module wherever a duration
should follow the global time, e.g. to present an input for a while.
The global time starts at 0 when the process first uses it.

- ``now()``: the global time as a ``Timestamp``, as ``Timer.now`` gives a timer's
  (``now().float()`` in seconds, ``now().ns`` in nanoseconds)
- ``sleep(seconds)``: block for ``seconds`` of global time, following the scale and
  pauses
- ``pause()`` / ``resume()`` / ``is_paused()``: stop the global time and move it
  again, e.g. to take the states of several runs on it at one time
- ``get_time_scale()`` / ``set_time_scale(scale)``: how many global seconds pass per
  real second (finite and > 0)
- ``Timer``: a time axis of its own for a run
- ``ReadOnlyTimer``: the reading side of a timer, as a node sees its run's timer
  (``Node.timer``)
- ``Timestamp``: a time on a timer, as ``Run.read_with_time`` and
  ``Port.recv_with_time`` give it

Only the host controls the time: ``sleep``, ``set_time_scale``, ``pause`` and
``resume`` raise ``RuntimeError`` in a node's hook.
"""

from mujou._core._native import Timestamp as Timestamp, timer as _timer
from mujou._core.timer import (
    ReadOnlyTimer as ReadOnlyTimer,
    Timer as Timer,
    host_only as _host_only,
)


def get_time_scale() -> float:
    """Return how many global seconds pass per real second."""
    return _timer.time_scale()


def now() -> Timestamp:
    """Return the global time now, as ``Timer.now`` returns a timer's.

    Compare it with the times of runs on the global timer, such as those that
    ``Run.read_with_time`` and ``Run.write`` return.
    """
    return _timer.now()


def sleep(seconds: float) -> None:
    """Block for ``seconds`` of global time, following the scale and the pauses.

    Raises:
        ValueError: ``seconds`` is negative or not finite.
        RuntimeError: This is called from a node's hook.
    """
    _host_only("mujou.time.sleep")
    _timer.sleep(seconds)


def set_time_scale(scale: float) -> None:
    """Set how many global seconds pass per real second.

    Raises:
        ValueError: ``scale`` is not finite and > 0.
        RuntimeError: This is called from a node's hook.
    """
    _host_only("mujou.time.set_time_scale")
    _timer.set_time_scale(scale)


def pause() -> None:
    """Stop the global time until ``resume``, as ``Timer.pause`` stops a timer's.

    Every run on the global timer stops updating at one time T, so states taken
    with ``Run.state_dict`` until ``resume`` are all of T.

    Raises:
        RuntimeError: This is called from a node's hook.
    """
    _host_only("mujou.time.pause")
    _timer.pause()


def resume() -> None:
    """Turn off the switch of ``pause``, as ``Timer.resume`` does for a timer.

    Raises:
        RuntimeError: This is called from a node's hook.
    """
    _host_only("mujou.time.resume")
    _timer.resume()


def is_paused() -> bool:
    """Return whether the global time is stopped, as ``Timer.is_paused`` does."""
    return _timer.is_paused()
