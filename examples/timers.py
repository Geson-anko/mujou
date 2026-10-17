"""Time in mujou: the global time, a timer of a run's own, scales and timestamps.

Every run measures ``dt`` on one timer: the global timer by default, or the
``Timer`` given as ``mujou.run(..., timer=...)``. A timer's time can run faster or
slower than the real time. Times are ``Timestamp`` values.
"""

import time
from typing import override

import torch

import mujou
from mujou import NextPort, Node, TensorSpec
from mujou.time import Timer, Timestamp

SCALAR = TensorSpec((), torch.float64)


class Clock(Node):
    """Sends the time of its run's timer, in seconds."""

    def __init__(self) -> None:
        self.next = NextPort(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        self.next.send(torch.tensor(float(self.timer.now()), dtype=torch.float64))


# A Timestamp: float() is seconds, .ns the exact nanoseconds; differences of two
# are Timestamps too.
t0, t1 = Timestamp(ns=1_500_000_000), Timestamp(ns=2_000_000_000)
assert float(t0) == 1.5 and int(t0) == 1 and (t1 - t0).ns == 500_000_000

# A run on a timer of its own, ten times faster than the real time. Pausing this
# run (to take its state, say) stops only this timer, not the global one.
timer = Timer()  # starts at 0
timer.set_time_scale(10.0)
clock = Clock()
start_real, start_global = time.perf_counter(), mujou.time.time()
with mujou.run(clock, timer=timer) as run:
    timer.sleep(2.0)  # 2 s of the run's time: about 0.2 s of real time
    # The clock sends once per update, and when it updates depends on how the
    # threads get scheduled: read until it has sent from 2 s on, giving up after
    # 5 s of real time.
    seen, t = run.read(clock.next)  # t is on the run's timer
    give_up = time.perf_counter() + 5.0
    while float(t) < 2.0 and time.perf_counter() < give_up:
        timer.sleep(0.1)  # 0.01 s of real time
        seen, t = run.read(clock.next)
elapsed_real = time.perf_counter() - start_real
elapsed_global = mujou.time.time() - start_global
print(
    f"run's time {float(t):.2f} s, real {elapsed_real:.2f} s, "
    f"global {elapsed_global:.2f} s"
)
# The run's 2 s took less real time (about 0.2 s, but a busy machine takes longer).
assert float(t) >= 2.0 and elapsed_real < 2.0
# The clock read its time just before it sent: compare times of the same timer
# only, since a Timestamp does not say which timer it is on.
assert seen.item() <= float(t)

# The global timer can be scaled too; it then applies to every run on it.
mujou.time.set_time_scale(5.0)
start_real = time.perf_counter()
mujou.time.sleep(0.5)  # 0.5 s of the global time: about 0.1 s of real time
print(f"slept 0.5 s of global time in {time.perf_counter() - start_real:.2f} s real")
mujou.time.set_time_scale(1.0)
