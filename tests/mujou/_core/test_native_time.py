import copy
import pickle
from collections.abc import Callable

import pytest

from mujou._core import _native
from mujou._core._native import GlobalTimer, Timer, Timestamp, timer


def virtual(t: Timer) -> Timer:
    """Put a timer on the virtual clock at 0 (the real clock ran until now)."""
    t.use_virtual_clock(True)
    t.load_state_dict({"time_ns": 0, "scale": 1.0})
    return t


class TestTimer:
    def test_the_state_round_trips_through_pickle_into_another_timer(self):
        source = virtual(Timer())
        source.set_time_scale(2.0)
        source.advance(3.0)
        state = pickle.loads(pickle.dumps(source.state_dict()))
        target = virtual(Timer())
        target.load_state_dict(state)
        assert target.time_ns() == 3_000_000_000
        assert target.time_scale() == 2.0

    @pytest.mark.parametrize(
        ("stop", "message"),
        [(lambda t: t.use_virtual_clock(False), "virtual"), (Timer.pause, "paused")],
        ids=["real-clock", "paused"],
    )
    def test_advance_raises_runtime_error_when_the_clock_cannot_move(
        self, stop: Callable[[Timer], None], message: str
    ):
        t = virtual(Timer())
        stop(t)
        with pytest.raises(RuntimeError, match=message):
            t.advance(1.0)

    def test_the_global_timer_is_a_timer_without_a_constructor(self):
        assert isinstance(timer, Timer)
        assert isinstance(timer, GlobalTimer)
        with pytest.raises(TypeError):
            GlobalTimer()


class TestTimerReader:
    def test_reads_its_timer_and_has_nothing_that_changes_it(self):
        t = virtual(Timer())
        reader = t.reader()
        t.advance(1.5)
        t.set_time_scale(3.0)
        assert reader.now() == Timestamp(ns=1_500_000_000)
        assert reader.time_scale() == 3.0
        assert {n for n in dir(reader) if not n.startswith("_")} == {
            "now",
            "time_scale",
        }
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

    def test_float_is_the_time_in_seconds(self):
        ts = Timestamp(ns=2_500_000_001)
        assert ts.float() == float(ts) == 2_500_000_001 / 1e9

    @pytest.mark.parametrize(
        ("a", "b"), [(10, 3), (3, 10), (7, 7), (2**64 - 1, 0), (0, 2**64 - 1)]
    )
    def test_difference_is_the_signed_seconds_between(self, a: int, b: int):
        difference = Timestamp(ns=a) - Timestamp(ns=b)
        assert type(difference) is float
        # Rounded as the difference of the exact nanoseconds is.
        assert difference == (a - b) / 1e9

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
