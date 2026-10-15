import os
import pickle
import subprocess
import sys
import sysconfig
import threading

import pytest

from mujou._core import GlobalTimer


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


class TestGlobalTimer:
    """The private controls; the public functions are tested via mujou.time."""

    def test_advance_converts_seconds_to_nanoseconds(self, virtual_timer: GlobalTimer):
        virtual_timer.advance(0.25)
        virtual_timer.advance(1e-9)
        assert virtual_timer.time_ns() == 250_000_001

    def test_sleep_waits_while_paused(self, virtual_timer: GlobalTimer):
        virtual_timer.use_virtual_clock(False)
        virtual_timer.set_time_scale(1000.0)
        virtual_timer.pause()
        assert virtual_timer.is_paused()
        sleeper = threading.Thread(target=virtual_timer.sleep, args=(1.0,))
        sleeper.start()
        sleeper.join(timeout=0.05)
        assert sleeper.is_alive()
        virtual_timer.resume()
        sleeper.join(timeout=5.0)
        assert not sleeper.is_alive()

    def test_state_dict_round_trips_through_pickle(self, virtual_timer: GlobalTimer):
        virtual_timer.set_time_scale(2.0)
        virtual_timer.advance(3.0)
        state = pickle.loads(pickle.dumps(virtual_timer.state_dict()))

        virtual_timer.set_time_scale(1.0)
        virtual_timer.advance(5.0)
        virtual_timer.load_state_dict(state)
        assert virtual_timer.time_ns() == 3_000_000_000
        assert virtual_timer.time_scale() == 2.0

    @pytest.mark.parametrize("seconds", [-1.0, float("nan")])
    def test_advance_rejects_invalid_duration(
        self, virtual_timer: GlobalTimer, seconds: float
    ):
        with pytest.raises(ValueError, match="duration"):
            virtual_timer.advance(seconds)

    def test_rejects_advance_on_the_real_clock(self, virtual_timer: GlobalTimer):
        virtual_timer.use_virtual_clock(False)
        with pytest.raises(RuntimeError, match="virtual"):
            virtual_timer.advance(1.0)
