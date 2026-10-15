import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import override

import pytest

from mujou._core import (
    GlobalTimer,
    NetworkError,
    Node,
    PauseTimeoutError,
    Runtime,
    timer,
)

H_NS = 1_000_000
H = H_NS / 1e9
WAIT = 5.0  # Real seconds to wait for a condition that must come; never reached.


class Log:
    """Hook calls of every node, in the order they happened."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self.events: list[tuple[str, str]] = []

    def add(self, name: str, event: str) -> None:
        with self._changed:
            self.events.append((name, event))
            self._changed.notify_all()

    def of(self, name: str) -> list[str]:
        with self._changed:
            return [e for n, e in self.events if n == name]

    def count(self, name: str, event: str) -> int:
        return self.of(name).count(event)

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self._changed:
            assert self._changed.wait_for(predicate, timeout=WAIT), "timed out"


class Recorder(Node):
    """Records its hook calls; raises in ``raise_in`` on its ``at``-th call."""

    def __init__(
        self, name: str, log: Log, raise_in: str | None = None, at: int = 1
    ) -> None:
        self.name = name
        self.log = log
        self.raise_in = raise_in
        self.at = at
        self.dts: list[float] = []
        self.threads: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.error: Exception | None = None
        self.loaded: object = None

    def hook(self, event: str) -> None:
        self.threads[event] = threading.get_ident()
        self.calls[event] = self.calls.get(event, 0) + 1
        self.log.add(self.name, event)
        if event == self.raise_in and self.calls[event] == self.at:
            self.error = ValueError(f"{self.name} raised in {event}")
            raise self.error

    @override
    def _on_thread_start(self) -> None:
        self.hook("_on_thread_start")

    @override
    def on_start(self) -> None:
        self.hook("on_start")

    @override
    def on_update(self, dt: float) -> None:
        self.dts.append(dt)
        self.hook("on_update")

    @override
    def on_pause(self) -> None:
        self.hook("on_pause")

    @override
    def on_resume(self) -> None:
        self.hook("on_resume")

    @override
    def on_exception(self) -> None:
        self.hook("on_exception")

    @override
    def on_stop(self) -> None:
        self.hook("on_stop")

    @override
    def state_dict(self) -> object:
        self.hook("state_dict")
        return {"name": self.name, "updates": len(self.dts)}

    @override
    def load_state_dict(self, state: object) -> None:
        self.loaded = state
        self.hook("load_state_dict")


type MakeRuntime = Callable[[Sequence[Recorder]], Runtime]


@pytest.fixture
def make_runtime() -> Iterator[MakeRuntime]:
    """Create runtimes that are stopped at the end of the test."""
    created: list[Runtime] = []

    def make(nodes: Sequence[Recorder]) -> Runtime:
        runtime = Runtime(nodes, [node.name for node in nodes])
        created.append(runtime)
        return runtime

    yield make
    for runtime in created:
        try:
            runtime.shutdown()
        except NetworkError:
            pass


def recorders(log: Log, *names: str) -> list[Recorder]:
    return [Recorder(name, log) for name in names]


def assert_ended_by_exception(log: Log, name: str) -> None:
    """``on_exception`` then ``on_stop``, once each, with nothing in between but
    the state capture on the exception."""
    events = log.of(name)
    assert events.count("on_exception") == 1
    assert events.count("on_stop") == 1
    after = events[events.index("on_exception") + 1 :]
    assert [e for e in after if e != "state_dict"] == ["on_stop"]


class TestStart:
    def test_hooks_run_in_order_on_the_node_thread(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        nodes = recorders(log, "a", "b")
        runtime = make_runtime(nodes)
        runtime.start(manual=True, states=[{"s": 0}, {"s": 1}])
        virtual_timer.advance(H)
        runtime.step()
        runtime.capture(lambda states: None)
        runtime.shutdown()

        for i, node in enumerate(nodes):
            events = log.of(node.name)
            assert events[:4] == [
                "_on_thread_start",
                "on_start",
                "load_state_dict",
                "on_update",
            ]
            assert events[4:] == ["on_pause", "state_dict", "on_resume", "on_stop"]
            assert node.loaded == {"s": i}
            node_thread = {t for e, t in node.threads.items() if e != "state_dict"}
            assert len(node_thread) == 1
            assert node.threads["state_dict"] not in node_thread

    def test_on_start_takes_no_global_time(self, make_runtime: MakeRuntime):
        class SlowStart(Recorder):
            @override
            def on_start(self) -> None:
                time.sleep(0.3)

        log = Log()
        node = SlowStart("a", log)
        runtime = make_runtime([node])
        runtime.start()
        log.wait_for(lambda: len(node.dts) >= 1)
        # Real clock at scale 1: the 0.3 s of `on_start` must not be in the first Δt.
        # Only the upper bound is checked, so a loaded machine does not fail it.
        assert node.dts[0] < 0.25

    @pytest.mark.parametrize(
        "hook", ["_on_thread_start", "on_start", "load_state_dict"]
    )
    def test_an_error_while_starting_stops_before_any_update(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime, hook: str
    ):
        log = Log()
        nodes = [Recorder("a", log), Recorder("b", log, raise_in=hook)]
        runtime = make_runtime(nodes)
        with pytest.raises(NetworkError, match="'b'") as exc_info:
            runtime.start(states=[None, None])
        assert exc_info.value.__cause__ is nodes[1].error
        # `start` raised after the network had stopped.
        assert runtime.is_finished
        for node in nodes:
            assert "on_update" not in log.of(node.name)
            assert_ended_by_exception(log, node.name)
        assert not virtual_timer.is_paused()

    def test_starts_only_once(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start()
        with pytest.raises(RuntimeError, match="already started"):
            runtime.start()


class TestFreeRun:
    def test_dt_sums_to_the_time_since_the_start(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        nodes = recorders(log, "a", "b")
        runtime = make_runtime(nodes)
        t0 = virtual_timer.time_ns()
        runtime.start()
        advances = [0.001, 0.0025, 3e-4, 0.1, 1e-9]
        for k, seconds in enumerate(advances, start=1):
            virtual_timer.advance(seconds)
            # Each node updates exactly once per advance: with Δt = 0 it waits.
            log.wait_for(lambda k=k: all(len(n.dts) >= k for n in nodes))
        runtime.shutdown()

        for node, last_ns in zip(nodes, runtime.last_update_ns(), strict=True):
            assert len(node.dts) == len(advances)
            assert last_ns == virtual_timer.time_ns()
            assert sum(node.dts) == pytest.approx((last_ns - t0) / 1e9, rel=1e-12)

    def test_waits_without_spinning_until_the_time_moves(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        runtime = make_runtime(recorders(log, "a"))
        runtime.start()
        time.sleep(0.1)
        assert log.count("a", "on_update") == 0
        # The waiting node wakes on control requests, not only on the time.
        states = runtime.capture(lambda states: states)
        assert states == [{"name": "a", "updates": 0}]
        start = time.monotonic()
        runtime.shutdown()
        assert time.monotonic() - start < 1.0
        assert log.count("a", "on_update") == 0

    def test_step_needs_the_manual_mode(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start()
        with pytest.raises(RuntimeError, match="manual"):
            runtime.step()


class TestCapture:
    def test_states_are_taken_at_one_frozen_time(self, make_runtime: MakeRuntime):
        log = Log()
        nodes = recorders(log, "a", "b", "c")
        runtime = make_runtime(nodes)
        runtime.start()
        log.wait_for(lambda: all(n.dts for n in nodes))

        def build(states: list[object]) -> tuple[list[object], int, bool, list[int]]:
            return states, timer.time_ns(), timer.is_paused(), runtime.last_update_ns()

        states, frozen_ns, paused, last_ns = runtime.capture(build)
        assert paused
        assert all(ns <= frozen_ns for ns in last_ns)
        assert [s["name"] for s in states if isinstance(s, dict)] == ["a", "b", "c"]
        assert not timer.is_paused()
        runtime.shutdown()
        for node in nodes:
            events = [e for e in log.of(node.name) if e != "on_update"]
            assert events[2:5] == ["on_pause", "state_dict", "on_resume"]

    def test_gives_up_when_a_node_does_not_pause(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        entered, release = threading.Event(), threading.Event()

        class Stuck(Recorder):
            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                entered.set()
                assert release.wait(WAIT)

        log = Log()
        stuck, other = Stuck("stuck", log), Recorder("other", log)
        runtime = make_runtime([stuck, other])
        runtime.start()
        virtual_timer.advance(H)
        assert entered.wait(WAIT)
        log.wait_for(lambda: len(other.dts) == 1)

        with pytest.raises(PauseTimeoutError, match="stuck") as exc_info:
            runtime.capture(lambda states: None, timeout=0.1)
        assert "other" not in str(exc_info.value)
        # The network keeps running on a running timer.
        assert runtime.is_running
        assert not virtual_timer.is_paused()
        log.wait_for(lambda: log.of("other")[-1] == "on_resume")
        release.set()
        virtual_timer.advance(H)
        log.wait_for(lambda: len(other.dts) == 2 and len(stuck.dts) == 2)
        runtime.shutdown()

    def test_an_error_in_build_propagates_and_the_network_keeps_running(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start()

        def build(states: list[object]) -> None:
            raise KeyError("build")

        with pytest.raises(KeyError, match="build"):
            runtime.capture(build)
        runtime.check()
        assert runtime.is_running
        assert not virtual_timer.is_paused()

    def test_build_cannot_wait_for_the_runtime(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start()
        # Waiting from the control thread for the control thread would deadlock.
        with pytest.raises(RuntimeError, match="own threads"):
            runtime.capture(lambda states: runtime.shutdown())
        assert runtime.is_running


def chain_update(x: list[float], i: int, dt: float, u: float, w: list[float]) -> None:
    """Explicit Euler of dx_i/dt = -x_i + w_i * x_{i-1} with x_{-1} = u."""
    prev = u if i == 0 else x[i - 1]
    x[i] = x[i] + dt * (-x[i] + w[i] * prev)


class Chain(Recorder):
    """Node i of the chain; ``x`` stands for the slots (node i writes only x[i])."""

    def __init__(
        self, i: int, x: list[float], u: float, w: list[float], log: Log
    ) -> None:
        super().__init__(f"x{i}", log)
        self.i, self.x, self.u, self.w = i, x, u, w

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        chain_update(self.x, self.i, dt, self.u, self.w)


class TestManual:
    U, W, X0, STEPS = 1.5, [0.9, -1.3, 0.7], [0.2, -0.4, 0.1], 7

    def run_chain(self, timer: GlobalTimer, make: MakeRuntime, order: list[int]):
        x = list(self.X0)
        log = Log()
        nodes = [Chain(i, x, self.U, self.W, log) for i in range(3)]
        runtime = make(nodes)
        runtime.start(manual=True)
        for _ in range(self.STEPS):
            timer.advance(H)
            runtime.step(order)
        runtime.shutdown()
        for node in nodes:
            assert node.dts == [H] * self.STEPS
        return x, log

    def test_forward_order_is_gauss_seidel(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        x, log = self.run_chain(virtual_timer, make_runtime, [0, 1, 2])
        expected = list(self.X0)
        for _ in range(self.STEPS):
            for i in range(3):  # Each node reads its predecessor's new value.
                chain_update(expected, i, H, self.U, self.W)
        assert x == expected
        updates = [n for n, e in log.events if e == "on_update"]
        assert updates == ["x0", "x1", "x2"] * self.STEPS

    def test_backward_order_is_jacobi(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        x, _ = self.run_chain(virtual_timer, make_runtime, [2, 1, 0])
        expected = list(self.X0)
        for _ in range(self.STEPS):
            old = list(expected)  # Every node reads the values of the last step.
            for i in range(3):
                new = list(old)
                chain_update(new, i, H, self.U, self.W)
                expected[i] = new[i]
        assert x == expected

    def test_a_step_without_time_passing_does_not_update(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        runtime = make_runtime(recorders(log, "a"))
        runtime.start(manual=True)
        runtime.step()
        assert log.count("a", "on_update") == 0

    def test_rejects_an_unknown_node(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start(manual=True)
        with pytest.raises(ValueError, match="out of range"):
            runtime.step([1])


class TestException:
    def test_an_update_error_stops_every_node(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        nodes = [
            Recorder("a", log),
            Recorder("x", log, "on_update", 3),
            Recorder("c", log),
        ]
        runtime = make_runtime(nodes)
        runtime.start(manual=True)
        for _ in range(2):
            virtual_timer.advance(H)
            runtime.step()
        virtual_timer.advance(H)
        with pytest.raises(NetworkError) as exc_info:
            runtime.step()
        assert "'x'" in str(exc_info.value)
        assert "on_update" in str(exc_info.value)
        assert exc_info.value.__cause__ is nodes[1].error
        for node in nodes:
            assert_ended_by_exception(log, node.name)
        # Every later request reports the same exception.
        for request in [
            runtime.step,
            lambda: runtime.capture(lambda states: None),
            runtime.shutdown,
            runtime.join,
            runtime.check,
        ]:
            with pytest.raises(NetworkError) as exc_info:
                request()
            assert exc_info.value.__cause__ is nodes[1].error

    @pytest.mark.parametrize("hook", ["on_pause", "state_dict", "on_resume"])
    def test_an_error_around_a_capture_stops_every_node(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime, hook: str
    ):
        log = Log()
        nodes = [Recorder("a", log), Recorder("x", log, raise_in=hook)]
        runtime = make_runtime(nodes)
        runtime.start()
        # `on_resume` raises after the capture has returned, so wait for the end.
        with pytest.raises(NetworkError) as exc_info:
            runtime.capture(lambda states: None)
            runtime.join(WAIT)
        assert exc_info.value.__cause__ is nodes[1].error
        for node in nodes:
            assert_ended_by_exception(log, node.name)
            if hook != "on_resume":
                # Stopping while paused does not resume.
                assert "on_resume" not in log.of(node.name)

    def test_the_state_is_captured_at_the_exception(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        nodes = [Recorder("x", log, "on_update", 2), Recorder("b", log)]
        runtime = make_runtime(nodes)
        seen: list[tuple[dict[int, object], int, str, BaseException, int, bool]] = []
        last_ns: list[int] = []

        def capture(
            states: dict[int, object], node: int, hook: str, exc: BaseException
        ) -> str:
            seen.append((states, node, hook, exc, timer.time_ns(), timer.is_paused()))
            last_ns.extend(runtime.last_update_ns())
            return "tree"

        runtime.start(manual=True, on_exception_capture=capture)
        virtual_timer.advance(H)
        runtime.step()
        virtual_timer.advance(H)
        with pytest.raises(NetworkError):
            runtime.step()

        [(states, node, hook, exc, frozen_ns, paused)] = seen
        # The failing node's state is in it too; "b" did not get its second step.
        assert states == {
            0: {"name": "x", "updates": 2},
            1: {"name": "b", "updates": 1},
        }
        assert (node, hook) == (0, "on_update")
        assert exc is nodes[0].error
        assert paused
        assert all(ns <= frozen_ns for ns in last_ns)
        assert runtime.exception_state() == "tree"
        assert not virtual_timer.is_paused()

    def test_a_state_that_fails_at_the_exception_is_left_out(
        self,
        virtual_timer: GlobalTimer,
        make_runtime: MakeRuntime,
        caplog: pytest.LogCaptureFixture,
    ):
        log = Log()
        nodes = [Recorder("x", log, "on_update"), Recorder("b", log, "state_dict")]
        runtime = make_runtime(nodes)
        runtime.start(
            manual=True,
            on_exception_capture=lambda states, node, hook, exc: sorted(states),
        )
        virtual_timer.advance(H)
        with pytest.raises(NetworkError):
            runtime.step()
        assert runtime.exception_state() == [0]
        assert "node 'b' raised in state_dict" in caplog.text
        for node in nodes:
            assert_ended_by_exception(log, node.name)

    def test_only_the_first_of_two_exceptions_is_the_cause(
        self,
        virtual_timer: GlobalTimer,
        make_runtime: MakeRuntime,
        caplog: pytest.LogCaptureFixture,
    ):
        both_updating = threading.Barrier(2, timeout=WAIT)

        class Simultaneous(Recorder):
            @override
            def on_update(self, dt: float) -> None:
                both_updating.wait()
                super().on_update(dt)

        log = Log()
        nodes = [Simultaneous(n, log, "on_update") for n in ["a", "b"]]
        runtime = make_runtime(nodes)
        runtime.start()
        virtual_timer.advance(H)
        with pytest.raises(NetworkError) as exc_info:
            runtime.join(WAIT)
        errors = [node.error for node in nodes]
        assert any(exc_info.value.__cause__ is e for e in errors)
        assert "node 'a' raised in on_update" in caplog.text
        assert "node 'b' raised in on_update" in caplog.text

    def test_an_error_in_on_stop_is_logged_and_the_others_still_stop(
        self,
        virtual_timer: GlobalTimer,
        make_runtime: MakeRuntime,
        caplog: pytest.LogCaptureFixture,
    ):
        log = Log()
        nodes = [Recorder("a", log, "on_stop"), Recorder("b", log)]
        runtime = make_runtime(nodes)
        runtime.start()
        runtime.shutdown()
        assert log.count("b", "on_stop") == 1
        assert "node 'a' raised in on_stop" in caplog.text
        runtime.check()
        assert runtime.join()


class TestWaiting:
    def test_join_waits_without_holding_the_gil(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        runtime = make_runtime(recorders(log, "a"))
        runtime.start()
        assert runtime.join(timeout=0.05) is False

        def advance_and_stop() -> None:
            # On a GIL build this thread runs only if `join` released the GIL.
            virtual_timer.advance(H)
            log.wait_for(lambda: log.count("a", "on_update") == 1)
            runtime.shutdown()

        thread = threading.Thread(target=advance_and_stop)
        thread.start()
        assert runtime.join()
        thread.join(WAIT)
        assert runtime.is_finished
        assert not runtime.is_running
