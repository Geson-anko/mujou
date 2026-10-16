"""The time of mujou.

Every run measures its nodes' ``dt`` on one timer, a time axis that can be scaled
and paused. By default every run shares the global timer, which the functions
below read; give a run a ``Timer`` of its own with ``mujou.run(..., timer=...)``.
Use these functions instead of the standard ``time`` module wherever a duration
should follow the global time, e.g. to present an input for a while.
The global time starts at 0 when the process first uses it.

- ``time()``: the global time in seconds
- ``time_ns()``: the global time in nanoseconds
- ``sleep(seconds)``: block for ``seconds`` of global time, following the scale and
  pauses
- ``get_time_scale()`` / ``set_time_scale(scale)``: how many global seconds pass per
  real second (finite and > 0)
- ``Timer``: a time axis of its own for a run
- ``Timestamp``: a time on a timer, as ``Run.read`` and ``Port.recv_with_time``
  give it
"""

from mujou._core._native import Timestamp as Timestamp, timer as _timer
from mujou._core.timer import Timer as Timer

time = _timer.time
time_ns = _timer.time_ns
sleep = _timer.sleep
get_time_scale = _timer.time_scale
set_time_scale = _timer.set_time_scale
