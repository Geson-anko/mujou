import threading
from collections.abc import Callable

import pytest

import mujou.time
from mujou._core._native import GlobalTimer
from mujou.time import Timestamp


def measure_sleep(
    timer: GlobalTimer, sleep: Callable[[float], None], seconds: float
) -> int:
    """Run ``sleep(seconds)`` in a thread while advancing the virtual clock in small
    steps, and return the global nanoseconds slept.

    The thread measures its own sleep, so the result does not depend on when it
    starts relative to the advances.
    """
    slept: list[int] = []

    def sleeper() -> None:
        start = timer.time_ns()
        sleep(seconds)
        slept.append(timer.time_ns() - start)

    thread = threading.Thread(target=sleeper)
    thread.start()
    for _ in range(10_000):
        thread.join(timeout=0.001)
        if not thread.is_alive():
            break
        timer.advance(seconds / 10)
    thread.join(timeout=5.0)
    assert slept, "sleep did not return"
    return slept[0]


class TestPublicNames:
    def test_exports(self):
        # The names of mujou.time in specs/network.md (it has no __all__).
        public = sorted(n for n in vars(mujou.time) if not n.startswith("_"))
        assert public == [
            "ReadOnlyTimer",
            "Timer",
            "Timestamp",
            "get_time_scale",
            "is_paused",
            "now",
            "pause",
            "resume",
            "set_time_scale",
            "sleep",
        ]


class TestTime:
    def test_now_is_the_global_time_as_a_timestamp(self, virtual_timer: GlobalTimer):
        virtual_timer.advance(1.5)
        assert mujou.time.now() == Timestamp(ns=1_500_000_000)
        virtual_timer.advance(1e-9)
        assert mujou.time.now().float() == 1_500_000_001 / 1e9

    def test_sleep_waits_in_global_time_without_holding_the_gil(
        self, virtual_timer: GlobalTimer
    ):
        # If sleep held the GIL (on a GIL build), the main thread could not advance.
        slept = measure_sleep(virtual_timer, mujou.time.sleep, 2.0)
        assert slept >= 2_000_000_000

    @pytest.mark.parametrize("seconds", [-1.0, float("nan")])
    def test_sleep_rejects_invalid_duration(self, seconds: float):
        with pytest.raises(ValueError, match="duration"):
            mujou.time.sleep(seconds)

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
    def test_set_time_scale_rejects_invalid_scale(self, scale: float):
        with pytest.raises(ValueError, match="scale"):
            mujou.time.set_time_scale(scale)
        assert mujou.time.get_time_scale() == 1.0


class TestPause:
    def test_stops_the_global_time_until_resume(self, virtual_timer: GlobalTimer):
        assert not mujou.time.is_paused()
        mujou.time.pause()
        mujou.time.pause()  # a switch: one resume undoes both
        assert mujou.time.is_paused()
        with pytest.raises(RuntimeError, match="paused"):
            virtual_timer.advance(1.0)
        mujou.time.resume()
        assert not mujou.time.is_paused()
        virtual_timer.advance(1.0)
        assert mujou.time.now() == Timestamp(ns=1_000_000_000)
