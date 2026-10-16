import threading
from collections.abc import Callable

import pytest

import mujou.time
from mujou._core._native import GlobalTimer


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


class TestTime:
    def test_reads_the_global_timer(self, virtual_timer: GlobalTimer):
        virtual_timer.advance(1.5)
        assert mujou.time.time_ns() == 1_500_000_000
        virtual_timer.advance(1e-9)
        assert mujou.time.time() == 1_500_000_001 / 1e9

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
