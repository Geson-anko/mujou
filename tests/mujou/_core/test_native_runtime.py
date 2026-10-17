import contextvars
import gc
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import override

import pytest

from mujou._core._native import (
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
    def on_thread_start(self) -> None:
        self.hook("on_thread_start")

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
        runtime = Runtime(nodes, [node.name for node in nodes], timer=timer)
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
    """``on_exception`` then ``on_stop``, once each, with nothing in between: the
    state is not taken on an exception."""
    events = log.of(name)
    assert events.count("on_exception") == 1
    assert events.count("on_stop") == 1
    after = events[events.index("on_exception") + 1 :]
    assert after == ["on_stop"]


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
                "on_thread_start",
                "on_start",
                "load_state_dict",
                "on_update",
            ]
            assert events[4:] == ["on_pause", "state_dict", "on_resume", "on_stop"]
            assert node.loaded == {"s": i}
            node_thread = {t for e, t in node.threads.items() if e != "state_dict"}
            assert len(node_thread) == 1
            assert node.threads["state_dict"] not in node_thread

    @pytest.mark.parametrize("hook", ["on_thread_start", "on_start", "load_state_dict"])
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

    def test_hooks_share_one_python_thread_state(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        local = threading.local()
        var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "var", default=None
        )
        seen: dict[str, tuple[object, object, threading.Thread]] = {}

        class ThreadLocal(Recorder):
            @override
            def hook(self, event: str) -> None:
                if event == "on_start":
                    local.value = self.name
                    var.set(self.name)
                seen[event] = (
                    getattr(local, "value", None),
                    var.get(),
                    threading.current_thread(),
                )
                super().hook(event)

        runtime = make_runtime([ThreadLocal("a", Log())])
        runtime.start(manual=True)
        virtual_timer.advance(H)
        runtime.step()
        runtime.capture(lambda states: None)
        runtime.shutdown()

        on_start_thread = seen["on_start"][2]
        for event in ["on_start", "on_update", "on_pause", "on_resume", "on_stop"]:
            value, context_value, thread = seen[event]
            assert value == "a", event
            assert context_value == "a", event
            assert thread is on_start_thread, event

    def test_the_control_thread_keeps_one_python_thread_state(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        local = threading.local()
        seen: list[tuple[object, threading.Thread]] = []

        class Remember(Recorder):
            @override
            def state_dict(self) -> object:
                seen.append((getattr(local, "value", None), threading.current_thread()))
                local.value = "set"
                return super().state_dict()

        runtime = make_runtime([Remember("a", Log())])
        runtime.start()

        def build(states: list[object]) -> None:
            seen.append((getattr(local, "value", None), threading.current_thread()))

        for _ in range(2):
            runtime.capture(build)
        # state_dict, build, state_dict, build: all on one thread state.
        assert [value for value, _ in seen] == [None, "set", "set", "set"]
        assert len({id(thread) for _, thread in seen}) == 1

    def test_hooks_replaced_while_starting_take_effect(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        calls: list[str] = []

        class Wrapping(Recorder):
            @override
            def on_start(self) -> None:
                plain = self.on_update

                def wrapped(dt: float) -> None:
                    calls.append("wrapped")
                    plain(dt)

                self.on_update = wrapped

            @override
            def on_update(self, dt: float) -> None:
                calls.append("plain")

        runtime = make_runtime([Wrapping("a", Log())])
        runtime.start(manual=True)
        virtual_timer.advance(H)
        runtime.step()
        assert calls == ["wrapped", "plain"]

    def test_restores_the_time_and_where_each_dt_starts(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        t_ns, behind_ns, ahead_ns = 5_000_000_000, 3_000_000, 2 * H_NS
        times: list[tuple[str, int, bool]] = []

        class Restored(Recorder):
            @override
            def hook(self, event: str) -> None:
                if event in ("on_start", "load_state_dict"):
                    times.append((event, timer.time_ns(), timer.is_paused()))
                super().hook(event)

        log = Log()
        nodes = [Restored("behind", log), Restored("ahead", log)]
        runtime = make_runtime(nodes)
        runtime.start(
            manual=True,
            states=[{}, {}],
            timer_state={"time_ns": t_ns, "scale": 1.0},
            last_update_ns=[t_ns - behind_ns, t_ns + ahead_ns],
        )
        # The time is restored first and does not run while the nodes start.
        assert times and all(time_ns == t_ns and paused for _, time_ns, paused in times)
        assert virtual_timer.time_ns() == t_ns
        virtual_timer.advance(H)
        runtime.step()
        assert nodes[0].dts == [(behind_ns + H_NS) / 1e9]
        # A node whose Δt would start in the future waits until it is positive.
        assert nodes[1].dts == []
        virtual_timer.advance(2 * H)
        runtime.step()
        assert nodes[1].dts == [(3 * H_NS - ahead_ns) / 1e9]

    @pytest.mark.parametrize(
        "start",
        [
            lambda runtime: runtime.start(last_update_ns=[1, 2]),
            lambda runtime: runtime.start(states=[None, None]),
            lambda runtime: runtime.start(timer_state={"time_ns": 0, "scale": 0.0}),
        ],
        ids=["last_update_ns", "states", "scale"],
    )
    def test_invalid_arguments_start_nothing(
        self,
        virtual_timer: GlobalTimer,
        make_runtime: MakeRuntime,
        start: Callable[[Runtime], None],
    ):
        log = Log()
        runtime = make_runtime(recorders(log, "a"))
        with pytest.raises(ValueError):
            start(runtime)
        assert log.events == []
        with pytest.raises(RuntimeError, match="not started"):
            runtime.join(timeout=0)
        assert not virtual_timer.is_paused()
        # Nothing was used up: the runtime can still start.
        runtime.start()
        assert runtime.join(timeout=0) is False

    def test_the_start_of_each_dt_is_known_when_starting_fails(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        virtual_timer.advance(1.0)
        start_ns = timer.time_ns()
        runtime = make_runtime([Recorder("a", Log(), "on_start"), Recorder("b", Log())])
        with pytest.raises(NetworkError):
            runtime.start()
        assert runtime.last_update_ns() == [start_ns, start_ns]

    def test_starts_only_once(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start()
        with pytest.raises(RuntimeError, match="already started"):
            runtime.start()

    def test_a_runtime_without_references_stops(self, virtual_timer: GlobalTimer):
        log = Log()
        runtime = Runtime(recorders(log, "a"), ["a"], timer=timer)
        runtime.start()
        del runtime
        gc.collect()
        # Dropping does not wait (the GIL may be held); the stop follows soon.
        log.wait_for(lambda: log.count("a", "on_stop") == 1)
        wait_until_no_runtime_runs()


def wait_until_no_runtime_runs() -> None:
    """Wait until restoring the time is allowed again, so that a runtime ending
    in the background does not leak into the next test."""
    deadline = time.monotonic() + WAIT
    while True:
        probe = Runtime([], [], timer=timer)
        try:
            probe.start(timer_state=timer.state_dict())
            probe.shutdown()
            return
        except ValueError:
            assert time.monotonic() < deadline, "a runtime kept running"
            time.sleep(0.001)


class FrameworkHooks(Recorder):
    """Records the framework's hooks too, and wraps the state in _capture_state."""

    @override
    def _on_thread_start(self) -> None:
        self.hook("_on_thread_start")

    @override
    def _restore_state(self, state: object) -> None:
        self.hook("_restore_state")
        super()._restore_state(state)

    @override
    def _after_update(self) -> None:
        self.hook("_after_update")

    @override
    def _capture_state(self) -> object:
        self.hook("_capture_state")
        return {"captured": super()._capture_state()}


class TestFrameworkHooks:
    def test_the_framework_starts_the_thread_before_the_node(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = FrameworkHooks("a", log)
        runtime = make_runtime([node])
        runtime.start(manual=True, states=[{"s": 0}])
        runtime.shutdown()
        assert log.of("a")[:5] == [
            "_on_thread_start",
            "on_thread_start",
            "on_start",
            "_restore_state",
            "load_state_dict",  # called by the default
        ]
        assert node.loaded == {"s": 0}
        assert len({node.threads[e] for e in log.of("a")[:5]}) == 1

    def test_an_error_in_the_framework_thread_start_stops_before_the_node_hooks(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = FrameworkHooks("a", log, raise_in="_on_thread_start")
        runtime = make_runtime([node])
        # Reported as the public hook it runs before.
        with pytest.raises(NetworkError, match="raised in on_thread_start$") as raised:
            runtime.start(manual=True)
        assert raised.value.__cause__ is node.error
        assert "on_thread_start" not in log.of("a")
        assert "on_start" not in log.of("a")

    def test_an_error_in_load_state_dict_still_names_load_state_dict(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = FrameworkHooks("a", log, raise_in="load_state_dict")
        runtime = make_runtime([node])
        with pytest.raises(NetworkError, match="in load_state_dict"):
            runtime.start(manual=True, states=[None])

    def test_after_update_follows_each_update_on_the_node_thread(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = FrameworkHooks("a", log)
        runtime = make_runtime([node])
        runtime.start(manual=True)
        for _ in range(2):
            virtual_timer.advance(H)
            runtime.step()
        runtime.shutdown()
        updates = [e for e in log.of("a") if e in ("on_update", "_after_update")]
        assert updates == ["on_update", "_after_update"] * 2
        assert node.threads["_after_update"] == node.threads["on_update"]

    def test_capture_state_replaces_state_dict(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = FrameworkHooks("a", log)
        runtime = make_runtime([node])
        runtime.start(manual=True)
        states = runtime.capture(lambda states: states)
        runtime.shutdown()
        assert states == [{"captured": {"name": "a", "updates": 0}}]
        assert log.of("a").count("_capture_state") == 1
        assert log.of("a").count("state_dict") == 1  # called by the default

    def test_an_error_in_state_dict_still_names_state_dict(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = Recorder("a", log, raise_in="state_dict")
        runtime = make_runtime([node])
        runtime.start(manual=True)
        with pytest.raises(NetworkError, match="state_dict"):
            runtime.capture(lambda states: states)

    def test_an_error_after_an_update_stops_the_network(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        log = Log()
        node = FrameworkHooks("a", log, raise_in="_after_update")
        runtime = make_runtime([node])
        runtime.start(manual=True)
        virtual_timer.advance(H)
        # Reported as the public hook it follows (a CUDA error of on_update
        # surfaces at the synchronization there).
        with pytest.raises(NetworkError, match="raised in on_update$") as raised:
            runtime.step()
        assert raised.value.__cause__ is node.error


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
        cpu_start = time.process_time()
        time.sleep(0.2)
        # A node spinning on Δt = 0 would use a core for the whole 0.2 s.
        assert time.process_time() - cpu_start < 0.05
        assert log.count("a", "on_update") == 0
        # The waiting node wakes on control requests, not only on the time.
        states = runtime.capture(lambda states: states)
        assert states == [{"name": "a", "updates": 0}]
        start = time.monotonic()
        runtime.shutdown()
        assert time.monotonic() - start < 1.0
        assert log.count("a", "on_update") == 0


class TestCapture:
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
        runtime.start(pause_timeout=0.1)
        virtual_timer.advance(H)
        assert entered.wait(WAIT)
        log.wait_for(lambda: len(other.dts) == 1)

        with pytest.raises(PauseTimeoutError, match="stuck") as exc_info:
            runtime.capture(lambda states: None)
        assert "other" not in str(exc_info.value)
        # The network keeps running on a running timer.
        assert runtime.join(timeout=0) is False
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
        assert runtime.join(timeout=0) is False
        assert not virtual_timer.is_paused()

    def test_build_cannot_wait_for_the_runtime(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start()
        # Waiting from the control thread for the control thread would deadlock.
        with pytest.raises(RuntimeError, match="own threads"):
            runtime.capture(lambda states: runtime.shutdown())
        assert runtime.join(timeout=0) is False


class TestManual:
    def test_rejects_an_unknown_node(
        self, virtual_timer: GlobalTimer, make_runtime: MakeRuntime
    ):
        runtime = make_runtime(recorders(Log(), "a"))
        runtime.start(manual=True)
        with pytest.raises(ValueError, match="out of range"):
            runtime.step([1])


class TestException:
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
        # The hold of the capture is released at the end.
        assert not virtual_timer.is_paused()
        for node in nodes:
            assert_ended_by_exception(log, node.name)
            if hook != "on_resume":
                # Stopping while paused does not resume.
                assert "on_resume" not in log.of(node.name)

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
