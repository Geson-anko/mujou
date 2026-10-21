import copy
import pickle
import threading
import time

import pytest

import mujou.time
from mujou.time import ReadOnlyTimer, Timer, Timestamp, VirtualTimer, global_timer


class TestPublicNames:
    def test_exports(self):
        # The names of mujou.time in specs/network.md (it has no __all__).
        public = sorted(n for n in vars(mujou.time) if not n.startswith("_"))
        assert public == [
            "ReadOnlyTimer",
            "Timer",
            "Timestamp",
            "VirtualTimer",
            "global_timer",
        ]

    def test_the_types_are_named_as_in_mujou_time(self):
        for kind in (Timer, VirtualTimer, ReadOnlyTimer, Timestamp):
            assert kind.__module__ == "mujou.time"


class TestTimer:
    def test_starts_at_zero_and_follows_the_real_clock(self):
        timer = Timer()
        first = timer.now()
        time.sleep(0.02)
        later = timer.now()
        assert isinstance(first, Timestamp)
        assert first < Timestamp(ns=1_000_000_000)
        assert later - first >= 0.02

    def test_starts_at_the_given_time_and_scale(self):
        timer = Timer(start=Timestamp(ns=5_000_000_000), time_scale=4.0)
        assert timer.now() >= Timestamp(ns=5_000_000_000)
        assert timer.time_scale() == 4.0

    def test_scales_its_own_time_only(self):
        timer = Timer()
        timer.set_time_scale(4.0)
        assert timer.time_scale() == 4.0
        assert global_timer.time_scale() == 1.0
        assert Timer().time_scale() == 1.0

    def test_sleeps_on_its_own_time(self):
        timer = Timer(time_scale=10.0)
        started = time.monotonic()
        timer.sleep(0.5)  # 0.05 s of real time
        assert time.monotonic() - started < 0.4

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
    def test_rejects_an_invalid_scale(self, scale: float):
        with pytest.raises(ValueError, match="scale"):
            Timer().set_time_scale(scale)
        with pytest.raises(ValueError, match="scale"):
            Timer(time_scale=scale)

    @pytest.mark.parametrize("seconds", [-1.0, float("nan")])
    def test_rejects_an_invalid_sleep(self, seconds: float):
        with pytest.raises(ValueError, match="duration"):
            Timer().sleep(seconds)

    def test_shows_only_what_a_user_needs(self):
        public = {name for name in dir(Timer) if not name.startswith("_")}
        assert public == {
            "now",
            "sleep",
            "time_scale",
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

    def test_starts_at_the_given_time(self):
        assert VirtualTimer(start=Timestamp(ns=7)).now() == Timestamp(ns=7)

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
        before = global_timer.now()
        VirtualTimer().advance(100.0)
        assert global_timer.now() - before < 50.0


class TestPause:
    def test_is_a_switch(self):
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

    def test_sleep_does_not_return_until_resumed(self):
        timer = Timer(time_scale=1000.0)
        timer.pause()
        slept = threading.Event()
        sleeper = threading.Thread(target=lambda: (timer.sleep(0.001), slept.set()))
        sleeper.start()
        # 0.05 real seconds pass, and none of the timer's.
        assert not slept.wait(0.05)
        timer.resume()
        sleeper.join(5.0)
        assert slept.is_set()


class TestReadOnlyTimer:
    def test_reads_its_timer_and_has_nothing_that_changes_it(self):
        timer = VirtualTimer()
        view = ReadOnlyTimer(timer)
        timer.advance(1.5)
        timer.set_time_scale(3.0)
        assert view.now() == Timestamp(ns=1_500_000_000)
        assert view.time_scale() == 3.0
        public = {n for n in dir(view) if not n.startswith("_")}
        assert public == {"now", "time_scale"}


class TestGlobalTimer:
    def test_is_one_timer(self):
        assert isinstance(global_timer, Timer)

    def test_follows_the_real_clock(self):
        first = global_timer.now()
        time.sleep(0.01)
        assert global_timer.now() - first >= 0.01


class TestTimestamp:
    def test_float_is_seconds_and_int_is_whole_seconds(self):
        ts = Timestamp(ns=2_500_000_001)
        assert ts.ns == 2_500_000_001
        assert float(ts) == 2_500_000_001 / 1e9
        assert int(ts) == 2

    def test_compares_and_hashes_by_time(self):
        early, late = Timestamp(ns=1), Timestamp(ns=2)
        assert early < late <= Timestamp(ns=2)
        assert late > early >= Timestamp(ns=1)
        assert early == Timestamp(ns=1)
        assert early != late
        assert len({early, Timestamp(ns=1), late}) == 2

    @pytest.mark.parametrize(
        ("a", "b"), [(10, 3), (3, 10), (7, 7), (2**64 - 1, 0), (0, 2**64 - 1)]
    )
    def test_difference_is_the_signed_seconds_between(self, a: int, b: int):
        difference = Timestamp(ns=a) - Timestamp(ns=b)
        assert type(difference) is float
        # Rounded as the difference of the exact nanoseconds is.
        assert difference == (a - b) / 1e9

    def test_repr_rebuilds_it(self):
        ts = Timestamp(ns=123)
        assert repr(ts) == "Timestamp(ns=123)"
        assert eval(repr(ts), {"Timestamp": Timestamp}) == ts

    def test_pickles_and_copies(self):
        ts = Timestamp(ns=2**64 - 1)
        for again in (
            pickle.loads(pickle.dumps(ts)),
            copy.deepcopy(ts),
            copy.copy(ts),
        ):
            assert type(again) is Timestamp and again == ts

    def test_takes_nanoseconds_by_keyword_only(self):
        # A bare number could mean seconds or nanoseconds.
        with pytest.raises(TypeError):
            Timestamp(5)  # type: ignore[misc]
