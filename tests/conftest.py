import threading
from collections.abc import Callable, Iterator

import pytest

from mujou._core import GlobalTimer, timer

type MeasureSleep = Callable[[Callable[[float], None], float], int]


@pytest.fixture
def virtual_timer() -> Iterator[GlobalTimer]:
    """Switch the global timer to the virtual clock for one test."""
    timer.use_virtual_clock(True)
    try:
        yield timer
    finally:
        timer.resume()
        timer.set_time_scale(1.0)
        timer.use_virtual_clock(False)


@pytest.fixture
def measure_sleep(virtual_timer: GlobalTimer) -> MeasureSleep:
    """Return a function that runs ``sleep(seconds)`` in a thread while advancing
    the virtual clock in small steps, and returns the global nanoseconds slept.

    The result does not depend on when the thread starts.
    """

    def measure(sleep: Callable[[float], None], seconds: float) -> int:
        slept: list[int] = []

        def sleeper() -> None:
            start = virtual_timer.time_ns()
            sleep(seconds)
            slept.append(virtual_timer.time_ns() - start)

        thread = threading.Thread(target=sleeper)
        thread.start()
        for _ in range(10_000):
            thread.join(timeout=0.001)
            if not thread.is_alive():
                break
            virtual_timer.advance(seconds / 10)
        thread.join(timeout=5.0)
        assert slept, "sleep did not return"
        return slept[0]

    return measure
