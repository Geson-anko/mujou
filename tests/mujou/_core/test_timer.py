import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, override

import pytest

import mujou.testing
import mujou.time
from mujou import Group, Node
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
        assert float(later - first) >= 0.02

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
        assert public == {"now", "sleep", "get_time_scale", "set_time_scale"}


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
        before = mujou.time.time_ns()
        VirtualTimer().advance(100.0)
        assert mujou.time.time_ns() - before < 50_000_000_000


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
        run = start_manual(Group(node, Calls()), timer=virtual)
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
        run = start_manual(Group(first, second), timer=virtual)
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
        run = start_manual(Group(node, Calls()), timer=virtual)
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
