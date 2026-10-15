from typing import TYPE_CHECKING

import pytest

import mujou.time
from mujou._core import GlobalTimer

if TYPE_CHECKING:
    from tests.conftest import MeasureSleep


class TestTime:
    def test_reads_the_global_timer(self, virtual_timer: GlobalTimer):
        start_ns = mujou.time.time_ns()
        virtual_timer.advance(1.5)
        assert mujou.time.time_ns() == start_ns + 1_500_000_000
        assert mujou.time.time() == mujou.time.time_ns() * 1e-9

    def test_sleep_waits_in_global_time(self, measure_sleep: MeasureSleep):
        slept = measure_sleep(mujou.time.sleep, 2.0)
        assert slept >= 2_000_000_000

    def test_time_scale_is_shared_with_the_global_timer(
        self, virtual_timer: GlobalTimer
    ):
        mujou.time.set_time_scale(8.0)
        assert mujou.time.get_time_scale() == 8.0
        assert virtual_timer.time_scale() == 8.0
        with pytest.raises(ValueError, match="scale"):
            mujou.time.set_time_scale(0.0)
