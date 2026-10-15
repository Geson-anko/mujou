from collections.abc import Iterator

import pytest

from mujou._core import GlobalTimer, timer


@pytest.fixture
def virtual_timer() -> Iterator[GlobalTimer]:
    """Switch the global timer to the virtual clock at time 0 for one test."""
    timer.use_virtual_clock(True)
    timer.load_state_dict({"time_ns": 0, "scale": 1.0})
    try:
        yield timer
    finally:
        timer.resume()
        timer.set_time_scale(1.0)
        timer.use_virtual_clock(False)
