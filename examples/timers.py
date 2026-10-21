"""Time in mujou: timestamps, a timer of a run's own, time scales, and pausing
the timer of several runs to take their states at one time.

Every run measures ``dt`` on one timer: ``mujou.time.global_timer`` by default,
or the ``Timer`` given as ``mujou.run(..., timer=...)``, which the host reaches
as ``run.timer``. A timer's time can run faster or slower than the real time.
"""

import pickle
import time
from typing import override

import torch

import mujou
from mujou import Node, Port, TensorSpec
from mujou.time import Timer, Timestamp, global_timer

SCALAR = TensorSpec((), torch.float64)


class Clock(Node):
    """Sends the time of its run's timer, in seconds, and counts its ``dt``."""

    def __init__(self) -> None:
        super().__init__()
        self.elapsed = SCALAR.zeros()  # a tensor: part of the state
        self.next = Port(send=SCALAR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.elapsed += dt
        self.next.send(torch.tensor(float(self.timer.now()), dtype=torch.float64))


# --8<-- [start:timestamp]
# A Timestamp: float(t) is seconds, .ns the exact nanoseconds; the difference of
# two is the seconds between them, a float (negative if the first is earlier).
t0, t1 = Timestamp(ns=1_500_000_000), Timestamp(ns=2_000_000_000)
assert float(t0) == 1.5 and t0.ns == 1_500_000_000
assert t1 - t0 == 0.5 and t0 - t1 == -0.5
# --8<-- [end:timestamp]

# --8<-- [start:own-timer]
# A run on a timer of its own, ten times faster than the real time. Pausing this
# timer (or taking this run's state) stops only this run, not the global timer.
timer = Timer(time_scale=10.0)  # starts at 0
clock = Clock()
start_real, start_global = time.perf_counter(), global_timer.now()
with mujou.run(clock, timer=timer) as run:
    run.timer.sleep(2.0)  # 2 s of the run's time: about 0.2 s of real time
    # When the clock updates depends on how the threads get scheduled: wait until
    # it has sent a time from 2 s on. The timeout is in real seconds.
    seen, t = run.wait_for(clock.next, lambda now: now.item() >= 2.0, timeout=5.0)
    assert t is not None  # None only for the zeros before the first send
elapsed_real = time.perf_counter() - start_real
elapsed_global = global_timer.now() - start_global
print(
    f"run's time {float(t):.2f} s, real {elapsed_real:.2f} s, "
    f"global {elapsed_global:.2f} s"
)
# Compare times of the same timer only: a Timestamp does not say which it is on.
assert seen.item() <= float(t)
# --8<-- [end:own-timer]

# --8<-- [start:global-scale]
# The global timer can be scaled too; it then applies to every run on it.
global_timer.set_time_scale(5.0)
start_real = time.perf_counter()
global_timer.sleep(0.5)  # 0.5 s of the global time: about 0.1 s of real time
print(f"slept 0.5 s of global time in {time.perf_counter() - start_real:.2f} s real")
global_timer.set_time_scale(1.0)
# --8<-- [end:global-scale]

# --8<-- [start:pause]
# Two runs on the global timer, such as two agents that should be saved together.
# Taken one after the other, the second state would be of a later time; pausing
# the timer they share stops both at one time T.
a, b = Clock(), Clock()
with mujou.run(a) as run_a, mujou.run(b) as run_b:
    global_timer.sleep(0.1)
    global_timer.pause()  # the global time stops at one time T
    try:
        tree_a = run_a.state_dict()  # still works while paused
        tree_b = run_b.state_dict()
    finally:
        global_timer.resume()  # the time goes on from T
t_ns = tree_a["timer"]["time_ns"]
assert tree_b["timer"]["time_ns"] == t_ns  # both are of T
data = pickle.dumps({"a": tree_a, "b": tree_b})  # save them together
# --8<-- [end:pause]
print(f"both states are of t = {t_ns / 1e9:.3f} s")

# --8<-- [start:restore]
# A run restored from one of them goes on from T on a timer of its own: the
# global timer does not go back.
saved = pickle.loads(data)
restored = Clock()
with mujou.run(restored, state=saved["a"]) as run:
    assert run.timer is not global_timer
    now, _ = run.wait_for(
        restored.next, lambda now: now.item() > t_ns / 1e9, timeout=5.0
    )
# After the run, its nodes' attributes are the host's to read.
at_save = saved["a"]["nodes"][""]["state"]["elapsed"].item()
print(
    f"run a went on from t = {t_ns / 1e9:.3f} s to {now.item():.3f} s; "
    f"its dt summed to {at_save:.3f} s at the save, {restored.elapsed.item():.3f} s now"
)
# --8<-- [end:restore]
