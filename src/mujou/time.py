"""The global time of mujou.

All nodes share one time axis, which can be scaled and paused. Use these functions
instead of the standard ``time`` module wherever a duration should follow the
global time, e.g. to present an input for a while.
The time starts at 0 when the process first uses it.

- ``time()``: the global time in seconds
- ``time_ns()``: the global time in nanoseconds
- ``sleep(seconds)``: block for ``seconds`` of global time, following the scale and
  pauses
- ``get_time_scale()`` / ``set_time_scale(scale)``: how many global seconds pass per
  real second (finite and > 0)
"""

from mujou._core import timer as _timer

time = _timer.time
time_ns = _timer.time_ns
sleep = _timer.sleep
get_time_scale = _timer.time_scale
set_time_scale = _timer.set_time_scale
