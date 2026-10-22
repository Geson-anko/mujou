import io
import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch

import mujou.pc as pc
from mujou.time import Timestamp

MS = 1_000_000  # ns
DT = 1e-3  # seconds per update


def at(ms: int) -> Timestamp:
    """The time ``ms`` milliseconds into the run."""
    return Timestamp(ns=ms * MS)


def clock(learn_interval: float | None, max_dt: float = 1.0) -> pc.LearnClock:
    return pc.LearnClock(period=DT, learn_interval=learn_interval, max_dt=max_dt)


def run(learn: pc.LearnClock, updates: Sequence[tuple[int, int]]) -> dict[int, pc.Tick]:
    """Tick ``learn`` once per ``(now, time)`` in ms, each update ``DT`` long;
    return the ticks by ``now``."""
    return {now: learn.tick(DT, at(time), at(now)) for now, time in updates}


def learned(ticks: dict[int, pc.Tick]) -> dict[int, float]:
    """The updates that learned, with their ``elapsed`` in ms."""
    return {now: round(t.elapsed / DT, 9) for now, t in ticks.items() if t.learn}


class TestGrid:
    def test_learns_at_the_first_update_on_or_after_each_point_of_the_write(self):
        # Written at 10 ms, learn_interval 3 ms: the points are 13, 16, 19 ms.
        # The update at 10 ms sees the input; each learns from the time since
        # the one before, that update included.
        ticks = run(clock(3e-3), [(now, 10) for now in range(10, 21)])
        assert learned(ticks) == {13: 4, 16: 3, 19: 3}

    def test_never_learns_in_the_update_that_first_sees_an_input(self):
        # Seen first at 5 ms, past the point 0 + 2 ms: it learns in the next
        # update, and then from the next point after it, 8 ms.
        ticks = run(clock(2e-3), [(now, 0) for now in range(5, 10)])
        assert not ticks[5].learn and ticks[5].new
        assert learned(ticks) == {6: 2, 8: 2}

    def test_the_next_point_after_a_late_update_is_on_the_grid(self):
        # Points at 2, 4, 6, 8, 10 ms: learned late at 7 ms, the next point is 8
        # ms (not 7 + 2); learned at 8 ms, it is 10 ms, past the update at 9 ms.
        ticks = run(clock(2e-3), [(1, 0), (7, 0), (8, 0), (9, 0), (10, 0)])
        assert [now for now in ticks if ticks[now].learn] == [7, 8, 10]

    def test_the_first_input_is_new_also_when_written_at_time_zero(self):
        ticks = run(clock(2e-3), [(1, 0), (2, 0), (3, 0)])
        assert [t.new for t in ticks.values()] == [True, False, False]
        assert learned(ticks) == {2: 2}


class TestElapsed:
    def test_an_input_that_changes_before_it_is_learned_drops_its_time(self):
        # The input written at 0 ms changes at 2 ms (seen at 3 ms) before the
        # clock learned from it: only the time from 3 ms counts at 7 ms.
        ticks = run(clock(5e-3), [(1, 0), (2, 0), *((now, 2) for now in range(3, 8))])
        assert learned(ticks) == {7: 5}

    def test_an_input_that_changes_after_it_is_learned_keeps_the_time(self):
        # Learned at 2 ms; the input written at 3 ms is seen at 4 ms and learned
        # at 5 ms, from the time of 3, 4 and 5 ms.
        ticks = run(clock(2e-3), [(1, 0), (2, 0), (3, 0), (4, 3), (5, 3)])
        assert learned(ticks) == {2: 2, 5: 3}

    def test_dt_is_capped_at_max_dt_in_the_tick_and_the_elapsed_time(self):
        learn = clock(None, max_dt=2e-3)
        assert learn.tick(5e-3, at(0), at(1)) == pc.Tick(2e-3, True, True, 2e-3)
        assert learn.tick(1e-3, at(0), at(2)) == pc.Tick(1e-3, False, True, 1e-3)

    def test_the_time_beyond_max_dt_is_dropped_from_the_weight(self):
        learn = clock(3e-3, max_dt=2e-3)
        learn.tick(1e-3, at(0), at(1))
        tick = learn.tick(5e-3, at(0), at(6))
        assert tick.learn and tick.elapsed == pytest.approx(3e-3)  # 1 + 2 ms


class TestWithoutInterval:
    def test_none_learns_in_every_update_by_its_own_dt(self):
        ticks = run(clock(None), [(1, 0), (2, 0), (3, 2), (4, 2)])
        assert learned(ticks) == {1: 1, 2: 1, 3: 1, 4: 1}
        assert [t.new for t in ticks.values()] == [True, False, True, False]

    def test_inf_never_learns(self):
        ticks = run(clock(math.inf), [(now, now // 10) for now in range(1, 100)])
        assert learned(ticks) == {}
        assert sum(t.new for t in ticks.values()) == 10

    @pytest.mark.parametrize(
        ("learn_interval", "discrete"), [(None, False), (2e-3, True), (math.inf, True)]
    )
    def test_learns_in_jumps_only_at_intervals(
        self, learn_interval: float | None, discrete: bool
    ):
        assert clock(learn_interval).discrete is discrete


class TestFromTimeConstants:
    def test_takes_the_default_max_dt_of_tau(self):
        tau = pc.TimeConstants(h=4e-3, lam=0.5e-3, theta=1.0)
        learn = pc.LearnClock.of(tau, period=DT, learn_interval=None)
        # max_dt is min(h / 2, 3 * lam) = 1.5 ms.
        assert learn.tick(10 * DT, at(0), at(1)).dt == pytest.approx(1.5e-3)
        given = pc.LearnClock.of(tau, period=DT, learn_interval=None, max_dt=1.2e-3)
        assert given.tick(10 * DT, at(0), at(1)).dt == pytest.approx(1.2e-3)

    def test_never_learns_with_an_infinite_theta(self):
        tau = pc.TimeConstants(h=4e-3, lam=8e-3, theta=math.inf)
        learn = pc.LearnClock.of(tau, period=DT, learn_interval=None)
        ticks = run(learn, [(now, 0) for now in range(1, 20)])
        assert learned(ticks) == {}


class TestChecks:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"learn_interval": 0.0}, "learn_interval"),
            ({"learn_interval": -1.0}, "learn_interval"),
            ({"learn_interval": math.nan}, "learn_interval"),
            ({"max_dt": 0.0}, "max_dt"),
            ({"max_dt": math.nan}, "max_dt"),
            ({"period": 2.0}, "max_dt"),  # longer than max_dt
        ],
    )
    def test_value_errors(self, kwargs: dict[str, float], match: str):
        args: dict[str, Any] = {"period": DT, "learn_interval": None, "max_dt": 1.0}
        with pytest.raises(ValueError, match=match):
            pc.LearnClock(**(args | kwargs))

    def test_takes_infinite_times(self):
        pc.LearnClock(period=DT, learn_interval=math.inf, max_dt=math.inf)


class TestState:
    @pytest.mark.parametrize("learn_interval", [None, 3e-3, math.inf])
    def test_a_restored_clock_goes_on_as_the_same_clock(
        self, learn_interval: float | None
    ):
        # Taken between two points, after a switch of the input.
        updates = [(now, 0 if now < 5 else 4) for now in range(1, 15)]
        whole = clock(learn_interval)
        expected = run(whole, updates)
        first = clock(learn_interval)
        run(first, updates[:6])
        buffer = io.BytesIO()
        torch.save(first.state_dict(), buffer)
        buffer.seek(0)
        restored = clock(learn_interval)
        restored.load_state_dict(torch.load(buffer))  # weights_only=True
        assert run(restored, updates[6:]) == {
            now: expected[now] for now, _ in updates[6:]
        }
