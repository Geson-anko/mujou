import copy
import os
import pickle
import subprocess
import sys
import sysconfig
import threading
import time

import pytest

from mujou._core import _native
from mujou._core._native import GlobalTimer, Timer, Timestamp, timer


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
                "import sys, mujou._core._native; print(sys._is_gil_enabled())",
            ],
            capture_output=True,
            text=True,
            check=True,
            # PYTHON_GIL=0 would keep the GIL off whatever the module declares.
            env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
        )
        assert result.stdout.strip() == "False"

    @pytest.mark.parametrize(
        "name",
        [
            "Timer",
            "GlobalTimer",
            "TimerReader",
            "Timestamp",
            "Schema",
            "BufferView",
            "Message",
            "Slot",
            "Outlet",
            "InputGroup",
            "InputSlot",
            "Node",
            "Runtime",
            "NetworkError",
            "PauseTimeoutError",
        ],
    )
    def test_every_class_belongs_to_this_module(self, name: str):
        assert getattr(_native, name).__module__ == _native.__name__

    def test_exceptions_keep_their_builtin_bases(self):
        assert issubclass(_native.NetworkError, RuntimeError)
        assert issubclass(_native.PauseTimeoutError, TimeoutError)
        assert str(_native.NetworkError("message")) == "message"


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


def virtual(t: Timer) -> Timer:
    """Put a timer on the virtual clock at 0 (the real clock ran until now)."""
    t.use_virtual_clock(True)
    t.load_state_dict({"time_ns": 0, "scale": 1.0})
    return t


class TestTimer:
    def test_a_new_timer_starts_at_zero_and_runs(self):
        t = Timer()
        start = t.time_ns()
        assert 0 <= start < 1_000_000_000
        time.sleep(0.01)
        assert t.time_ns() > start

    def test_timers_advance_independently(self):
        a, b = virtual(Timer()), virtual(Timer())
        a.advance(2.0)
        b.advance(0.5)
        assert (a.time_ns(), b.time_ns()) == (2_000_000_000, 500_000_000)

    def test_pausing_one_timer_leaves_the_global_timer_running(
        self, virtual_timer: GlobalTimer
    ):
        own = virtual(Timer())
        own.pause()
        assert own.is_paused()
        assert not virtual_timer.is_paused()
        virtual_timer.advance(1.0)
        assert virtual_timer.time_ns() == 1_000_000_000
        with pytest.raises(RuntimeError, match="paused"):
            own.advance(1.0)

    def test_the_state_of_one_timer_loads_into_another(self):
        source = virtual(Timer())
        source.set_time_scale(2.0)
        source.advance(3.0)
        target = virtual(Timer())
        target.load_state_dict(source.state_dict())
        assert target.time_ns() == 3_000_000_000
        assert target.time_scale() == 2.0

    def test_now_is_a_timestamp_of_the_time(self):
        t = virtual(Timer())
        t.advance(1.25)
        assert t.now() == Timestamp(ns=1_250_000_000)

    def test_the_global_timer_is_a_timer_without_a_constructor(self):
        assert isinstance(timer, Timer)
        assert isinstance(timer, GlobalTimer)
        with pytest.raises(TypeError):
            GlobalTimer()


class TestTimerReader:
    def test_reads_the_time_and_the_scale_of_its_timer(self):
        t = virtual(Timer())
        reader = t.reader()
        t.advance(1.5)
        t.set_time_scale(3.0)
        assert reader.now() == Timestamp(ns=1_500_000_000)
        assert reader.time_scale() == 3.0

    def test_has_nothing_that_changes_the_timer(self):
        public = {n for n in dir(Timer().reader()) if not n.startswith("_")}
        assert public == {"now", "time_scale"}
        with pytest.raises(TypeError):
            _native.TimerReader()


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

    def test_difference_is_the_time_between(self):
        assert Timestamp(ns=10) - Timestamp(ns=3) == Timestamp(ns=7)
        with pytest.raises(ValueError, match="later"):
            _ = Timestamp(ns=3) - Timestamp(ns=10)

    @pytest.mark.parametrize("ns", [-1, 2**64])
    def test_rejects_nanoseconds_out_of_range(self, ns: int):
        with pytest.raises(ValueError, match="2\\*\\*64"):
            Timestamp(ns=ns)

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
