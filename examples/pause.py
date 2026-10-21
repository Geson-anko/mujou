"""Pause the time to take the states of two runs at one time.

``Run.state_dict`` takes the state of one run at one time. Taken one run after the
other, the second state is of a later time: the time goes on in between. Pausing
the timer that both runs are on stops it at one time T, so that both states are
of T.
"""

import pickle
from typing import override

import torch

import mujou
import mujou.time
from mujou import Node, Port, TensorSpec

SCALAR = TensorSpec((), torch.float64)


class Clock(Node):
    """Accumulates its ``dt`` (the time of its run since the start) and sends it."""

    def __init__(self) -> None:
        super().__init__(period=0.01)
        self.next = Port(send=SCALAR)

    @override
    def on_start(self) -> None:
        super().on_start()
        self.elapsed = SCALAR.zeros()  # a tensor: part of the state

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.elapsed += dt
        self.next.send(self.elapsed)


# --8<-- [start:pause]
# Two runs on the global timer, such as two agents that should be saved together.
a, b = Clock(), Clock()
with mujou.run(a) as run_a, mujou.run(b) as run_b:
    mujou.time.global_timer.sleep(0.1)
    mujou.time.global_timer.pause()  # the global time stops at one time T
    try:
        tree_a = run_a.state_dict()  # still works while paused
        tree_b = run_b.state_dict()
    finally:
        mujou.time.global_timer.resume()  # the time goes on from T
    mujou.time.global_timer.sleep(0.05)

# Both trees are of T, and no update in them is later than T.
t_ns = tree_a["timer"]["time_ns"]
assert tree_b["timer"]["time_ns"] == t_ns
assert tree_a["last_update_ns"][""] <= t_ns and tree_b["last_update_ns"][""] <= t_ns
data = pickle.dumps({"a": tree_a, "b": tree_b})  # save them together
# --8<-- [end:pause]
print(f"both states are of t = {t_ns / 1e9:.3f} s")

# Restore one of them in a new run, which goes on from T on a timer of its own
# (the global timer does not go back to T).
saved = pickle.loads(data)
at_save = saved["a"]["nodes"][""]["elapsed"].item()
restored = Clock()
with mujou.run(restored, state=saved["a"]) as run:
    # Wait until it has updated after the restore.
    went_on, _ = run.wait_for(restored.next, lambda e: e.item() > at_save, timeout=5.0)
print(f"run a went on from {at_save:.3f} s to {went_on.item():.3f} s")
