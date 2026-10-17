import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

import pytest
import torch

import mujou.testing
import mujou.time
from mujou import Compose, NextPort, Node, TensorSpec
from mujou.testing import VirtualTimer
from mujou.time import Timer, Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual

H = 0.001


class TestTimer:
    def test_starts_at_zero_and_follows_the_real_clock(self):
        timer = Timer()
        first = timer.now()
        time.sleep(0.02)
        later = timer.now()
        assert isinstance(first, Timestamp)
        assert first < Timestamp(ns=1_000_000_000)
        assert later - first >= 0.02

    def test_scales_its_own_time_only(self):
        timer = Timer()
        timer.set_time_scale(4.0)
        assert timer.get_time_scale() == 4.0
        assert mujou.time.get_time_scale() == 1.0
        assert Timer().get_time_scale() == 1.0

    def test_sleeps_on_its_own_time(self):
        timer = Timer()
        timer.set_time_scale(10.0)
        started = time.monotonic()
        timer.sleep(0.5)  # 0.05 s of real time
        assert time.monotonic() - started < 0.4

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
    def test_rejects_an_invalid_scale(self, scale: float):
        with pytest.raises(ValueError, match="scale"):
            Timer().set_time_scale(scale)

    def test_shows_only_what_a_user_needs(self):
        public = {name for name in dir(Timer) if not name.startswith("_")}
        assert public == {
            "now",
            "sleep",
            "get_time_scale",
            "set_time_scale",
            "pause",
            "resume",
            "is_paused",
        }


class TestVirtualTimer:
    def test_moves_only_by_advance(self):
        timer = VirtualTimer()
        assert timer.now() == Timestamp(ns=0)
        time.sleep(0.01)
        assert timer.now() == Timestamp(ns=0)
        timer.advance(1.5)
        timer.advance(1e-9)
        assert timer.now() == Timestamp(ns=1_500_000_001)
        assert isinstance(timer, Timer)

    def test_the_scale_does_not_apply(self):
        timer = VirtualTimer()
        timer.set_time_scale(3.0)
        timer.advance(1.0)
        assert timer.now() == Timestamp(ns=1_000_000_000)

    def test_sleep_returns_once_advanced_far_enough(self):
        timer = VirtualTimer()
        done = threading.Event()

        def sleeper() -> None:
            timer.sleep(1.0)
            done.set()

        thread = threading.Thread(target=sleeper)
        thread.start()
        try:
            assert not done.wait(0.05)
            timer.advance(0.5)
            assert not done.wait(0.05)
            timer.advance(0.5)
            assert done.wait(5.0)
        finally:
            timer.advance(10.0)
            thread.join(5.0)

    @pytest.mark.parametrize("seconds", [-1.0, float("nan")])
    def test_rejects_an_invalid_advance(self, seconds: float):
        with pytest.raises(ValueError):
            VirtualTimer().advance(seconds)

    def test_is_independent_of_the_global_timer(self):
        before = mujou.time.now()
        VirtualTimer().advance(100.0)
        assert mujou.time.now() - before < 50.0


class Ticks(Node):
    """Counts its updates, in a tensor of its state, sends the count, and records
    each ``dt``."""

    def __init__(self) -> None:
        super().__init__(period=0.001)
        self.next = NextPort(send=TensorSpec((), torch.int64))
        self.dts: list[float] = []

    @override
    def on_start(self) -> None:
        super().on_start()
        self.count = torch.zeros((), dtype=torch.int64)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.dts.append(dt)
        self.count += 1
        self.next.send(self.count)


def wait_for_updates(run: mujou.Run, node: Ticks, more_than: int) -> None:
    run.wait_for(node.next, lambda count: count.item() > more_than, timeout=10.0)


class TestPause:
    """The host stops a timer to do something at one time T, such as taking the
    states of every run on it."""

    def test_is_a_switch_apart_from_the_runs(self):
        timer = VirtualTimer()
        assert not timer.is_paused()
        timer.pause()
        timer.pause()
        assert timer.is_paused()
        with pytest.raises(RuntimeError, match="paused"):
            timer.advance(1.0)
        timer.resume()  # one resume undoes both pauses
        assert not timer.is_paused()
        timer.advance(1.0)
        assert timer.now() == Timestamp(ns=1_000_000_000)

    def test_the_states_of_two_runs_on_one_timer_are_of_one_time(self):
        timer = Timer()  # the real clock: one state after the other would differ
        a, b = Ticks(), Ticks()
        with mujou.run(a, timer=timer) as run_a, mujou.run(b, timer=timer) as run_b:
            wait_for_updates(run_a, a, 2)
            wait_for_updates(run_b, b, 2)
            timer.pause()
            try:
                paused_at = timer.now()
                trees = [run_a.state_dict(), run_b.state_dict()]
                assert timer.now() == paused_at
                assert timer.is_paused()  # the runs' own holds left the switch on
            finally:
                timer.resume()
            assert not timer.is_paused()
        for tree in trees:
            assert tree["timer"]["time_ns"] == paused_at.ns
            assert tree["last_update_ns"][""] <= paused_at.ns
            assert tree["nodes"][""]["count"] > 2

    def test_nothing_moves_while_paused(self):
        timer = Timer()
        node = Ticks()
        with mujou.run(node, timer=timer) as run:
            wait_for_updates(run, node, 2)
            timer.pause()
            try:
                before = run.state_dict()
                slept = threading.Event()
                sleeper = threading.Thread(
                    target=lambda: (timer.sleep(0.001), slept.set())
                )
                sleeper.start()
                # 0.05 real seconds pass, and none of the timer's.
                assert not slept.wait(0.05)
                after = run.state_dict()
            finally:
                timer.resume()
            sleeper.join(5.0)
        assert slept.is_set()
        assert after["timer"] == before["timer"]
        assert after["last_update_ns"] == before["last_update_ns"]
        assert after["nodes"][""]["count"] == before["nodes"][""]["count"]

    def test_the_first_dt_after_resume_leaves_out_the_pause(self):
        timer = Timer()
        node = Ticks()
        with mujou.testing.run(node, timer=timer) as run:
            real_start, start = time.monotonic(), timer.now()
            run.step()
            timer.pause()
            time.sleep(0.05)
            timer.resume()
            run.step()
            end, real_end = timer.now(), time.monotonic()
        # Both updates fall between start and end, on the timer, which stood
        # still for the 0.05 real seconds of the pause.
        assert node.dts[-1] <= end - start
        assert end - start <= real_end - real_start - 0.05 + 1e-6

    def test_runs_go_on_from_states_taken_while_paused(self):
        timer = Timer()
        a, b = Ticks(), Ticks()
        with mujou.run(a, timer=timer) as run_a, mujou.run(b, timer=timer) as run_b:
            wait_for_updates(run_a, a, 2)
            wait_for_updates(run_b, b, 2)
            timer.pause()
            try:
                trees = [run_a.state_dict(), run_b.state_dict()]
            finally:
                timer.resume()
        paused_at_ns = trees[0]["timer"]["time_ns"]
        for tree in trees:
            # A timer of its own each: a run cannot restore while another runs
            # on its timer. Both start again from the same time.
            restored, own = Ticks(), Timer()
            with mujou.run(restored, timer=own, state=tree) as run:
                assert own.now().ns >= paused_at_ns
                saved = int(tree["nodes"][""]["count"])
                wait_for_updates(run, restored, saved)
            assert run.state_dict()["last_update_ns"][""] > paused_at_ns


class Calls(Node):
    """Calls ``action`` in each ``on_update``, keeps what it raised, and records
    each ``dt``."""

    def __init__(self, action: Callable[[], object] = lambda: None) -> None:
        self.action = action
        self.errors: list[Exception] = []
        self.dts: list[float] = []

    @override
    def on_update(self, dt: float) -> None:
        self.dts.append(dt)
        try:
            self.action()
        except Exception as e:  # noqa: BLE001 - kept for the test to inspect
            self.errors.append(e)


class TestOnlyTheHostControlsTheTime:
    """A node reads the time and changes nothing: a node changing the scale, the
    virtual clock or waiting on the time would reach the other nodes past the
    ports."""

    @pytest.mark.parametrize("scaled", ["timer", "global"])
    def test_a_node_cannot_scale_the_time(
        self, virtual: VirtualTimer, start_manual: StartManual, scaled: str
    ):
        target = virtual if scaled == "timer" else mujou.time
        node = Calls(lambda: target.set_time_scale(1000.0))
        run = start_manual(Compose(node, Calls()), timer=virtual)
        try:
            virtual.advance(H)
            run.step()
            assert isinstance(node.errors[0], RuntimeError)
            assert "only the host" in str(node.errors[0])
            assert virtual.get_time_scale() == 1.0
            assert mujou.time.get_time_scale() == 1.0
        finally:
            # Should the check regress, leave no scale to the later tests.
            mujou.time.set_time_scale(1.0)

    def test_a_node_cannot_advance_the_virtual_clock(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        first = Calls(lambda: virtual.advance(5.0))
        second = Calls()
        run = start_manual(Compose(first, second), timer=virtual)
        virtual.advance(H)
        run.step(first, second)
        assert isinstance(first.errors[0], RuntimeError)
        # Every dt is the time the test advanced.
        assert second.dts == [H]
        assert virtual.now() == Timestamp(ns=1_000_000)

    @pytest.mark.parametrize("sleep", ["timer", "global"])
    def test_a_node_cannot_wait_on_the_time(
        self, virtual: VirtualTimer, start_manual: StartManual, sleep: str
    ):
        wait = virtual.sleep if sleep == "timer" else mujou.time.sleep
        node = Calls(lambda: wait(0.001))
        run = start_manual(Compose(node, Calls()), timer=virtual)
        virtual.advance(H)
        # Nothing in the step waits for the clock. Should it regress, the step
        # would wait for an advance: fail instead of hanging.
        stepper = threading.Thread(target=run.step)
        stepper.start()
        stepper.join(5.0)
        if stepper.is_alive():
            virtual.advance(1.0)  # release the sleeping node
            stepper.join()
            pytest.fail("a node's sleep held the step")
        assert isinstance(node.errors[0], RuntimeError)
        run.state_dict()  # the pause does not wait for a sleeping node

    @pytest.mark.parametrize("target", ["timer", "global"])
    @pytest.mark.parametrize("switch", ["pause", "resume"])
    def test_a_node_cannot_pause_or_resume_the_time(
        self, virtual: VirtualTimer, start_manual: StartManual, target: str, switch: str
    ):
        owner = virtual if target == "timer" else mujou.time
        node = Calls(getattr(owner, switch))
        run = start_manual(Compose(node, Calls()), timer=virtual)
        try:
            virtual.advance(H)
            run.step()
            assert isinstance(node.errors[0], RuntimeError)
            assert "only the host" in str(node.errors[0])
            assert not virtual.is_paused()
            assert not mujou.time.is_paused()
        finally:
            # Should the check regress, leave no pause to the later tests.
            virtual.resume()
            mujou.time.resume()
