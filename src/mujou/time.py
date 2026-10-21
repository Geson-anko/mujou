"""The time of mujou.

Every run measures its nodes' ``dt`` on one timer, a time axis that can be scaled
and paused. By default every run shares ``global_timer``; give a run a ``Timer``
of its own with ``mujou.run(..., timer=...)``. The host controls the time of a
run through ``run.timer``, and a node reads it with ``self.timer``.

- ``Timer``: a time axis, running with the real clock at a scale
- ``VirtualTimer``: a ``Timer`` that moves only when advanced, for tests
- ``ReadOnlyTimer``: the reading side of a timer, as a node sees its run's
  timer (``Node.timer``)
- ``Timestamp``: a time on a timer, as ``Run.read_with_time`` and
  ``Port.recv_with_time`` give it
- ``global_timer``: the timer of every run not given one of its own

Durations are in seconds of the timer (``sleep``, ``advance``, ``dt``,
``period``), except timeouts, which are in real seconds. Only the host controls
the time: ``sleep``, ``pause``, ``resume``, ``set_time_scale`` and
``VirtualTimer.advance`` raise ``RuntimeError`` in a node's hook.
"""

from mujou._core._native import (
    ReadOnlyTimer as ReadOnlyTimer,
    Timer as Timer,
    Timestamp as Timestamp,
    VirtualTimer as VirtualTimer,
    global_timer as global_timer,
)
