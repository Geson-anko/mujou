import threading
import time

import pytest

import mujou.testing
import mujou.time
from mujou.testing import VirtualTimer
from mujou.time import Timer, Timestamp


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
