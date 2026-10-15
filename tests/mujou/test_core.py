import os
import pickle
import subprocess
import sys
import sysconfig
import threading
from typing import TYPE_CHECKING

import pytest

from mujou._core import GlobalTimer

if TYPE_CHECKING:
    from tests.conftest import MeasureSleep


class TestCoreModule:
    @pytest.mark.skipif(
        not sysconfig.get_config_var("Py_GIL_DISABLED"),
        reason="requires a free-threaded build",
    )
    def test_import_keeps_gil_disabled(self):
        # A fresh interpreter, so that no other extension has re-enabled the GIL.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, mujou._core; print(sys._is_gil_enabled())",
            ],
            capture_output=True,
            text=True,
            check=True,
            # PYTHON_GIL=0 would keep the GIL off whatever the module declares.
            env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
        )
        assert result.stdout.strip() == "False"


def _start_sleeper(timer: GlobalTimer, seconds: float) -> threading.Thread:
    sleeper = threading.Thread(target=timer.sleep, args=(seconds,))
    sleeper.start()
    return sleeper


class TestGlobalTimer:
    def test_advance_converts_seconds_to_nanoseconds(self, virtual_timer: GlobalTimer):
        start = virtual_timer.time_ns()
        virtual_timer.advance(0.25)
        virtual_timer.advance(1e-9)
        assert virtual_timer.time_ns() == start + 250_000_001
        assert virtual_timer.time() == virtual_timer.time_ns() * 1e-9

    def test_sleep_releases_the_gil_until_advanced(
        self, virtual_timer: GlobalTimer, measure_sleep: MeasureSleep
    ):
        # If sleep held the GIL (on a GIL build), the main thread could not advance.
        slept = measure_sleep(virtual_timer.sleep, 1.0)
        assert slept >= 1_000_000_000

    def test_sleep_waits_while_paused(self, virtual_timer: GlobalTimer):
        virtual_timer.use_virtual_clock(False)
        virtual_timer.set_time_scale(1000.0)
        virtual_timer.pause()
        assert virtual_timer.is_paused()
        sleeper = _start_sleeper(virtual_timer, 1.0)
        sleeper.join(timeout=0.05)
        assert sleeper.is_alive()
        virtual_timer.resume()
        sleeper.join(timeout=5.0)
        assert not sleeper.is_alive()

    def test_state_dict_round_trips_through_pickle(self, virtual_timer: GlobalTimer):
        virtual_timer.set_time_scale(2.0)
        virtual_timer.advance(3.0)
        state = pickle.loads(pickle.dumps(virtual_timer.state_dict()))
        expected_ns = virtual_timer.time_ns()

        virtual_timer.set_time_scale(1.0)
        virtual_timer.advance(5.0)
        virtual_timer.load_state_dict(state)
        assert virtual_timer.time_ns() == expected_ns
        assert virtual_timer.time_scale() == 2.0

    @pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
    def test_rejects_invalid_scale(self, virtual_timer: GlobalTimer, scale: float):
        with pytest.raises(ValueError, match="scale"):
            virtual_timer.set_time_scale(scale)

    @pytest.mark.parametrize("seconds", [-1.0, float("nan")])
    def test_rejects_invalid_duration(self, virtual_timer: GlobalTimer, seconds: float):
        with pytest.raises(ValueError, match="duration"):
            virtual_timer.sleep(seconds)
        with pytest.raises(ValueError, match="duration"):
            virtual_timer.advance(seconds)

    def test_rejects_advance_on_the_real_clock(self, virtual_timer: GlobalTimer):
        virtual_timer.use_virtual_clock(False)
        with pytest.raises(RuntimeError, match="virtual"):
            virtual_timer.advance(1.0)
