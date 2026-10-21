import contextlib
import contextvars
import gc
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, override

import pytest

from mujou._core._native import ReadOnlyTimer, Runtime, VirtualTimer, current, start
from mujou._core.errors import NetworkError

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


class Recorder:
    """A driver (the runtime's side of a node, ``mujou._core.driver.Driver``)
    that records the calls of its methods; raises in ``raise_in`` on its
    ``at``-th call."""

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

    def start(self) -> None:
        self.hook("start")

    def update(self, dt: float) -> None:
        self.dts.append(dt)
        self.hook("update")

    def pause(self) -> None:
        self.hook("pause")

    def resume(self) -> None:
        self.hook("resume")

    def exception(self) -> None:
        self.hook("exception")

    def stop(self) -> None:
        self.hook("stop")

    def capture(self) -> object:
        self.hook("capture")
        return {"name": self.name, "updates": len(self.dts)}

    def restore(self, state: object) -> None:
        self.loaded = state
        self.hook("restore")


class StartRuntime(Protocol):
    def __call__(
        self,
        nodes: Sequence[Recorder],
        *,
        manual: bool = False,
        dedicated: bool = False,
        states: Sequence[object] | None = None,
        last_update_ns: Sequence[int | None] | None = None,
        workers: int | None = None,
    ) -> Runtime: ...


@pytest.fixture
def virtual_timer() -> VirtualTimer:
    """A new timer on the virtual clock at time 0, for one test."""
    return VirtualTimer()


@pytest.fixture
def start_runtime(virtual_timer: VirtualTimer) -> Iterator[StartRuntime]:
    """Start runtimes on the test's ``virtual_timer``, each node with a period of
    0 and named by its ``name``, and stop them at the end of the test."""
    started: list[Runtime] = []

    def start_runtime(
        nodes: Sequence[Recorder],
        *,
        manual: bool = False,
        dedicated: bool = False,
        states: Sequence[object] | None = None,
        last_update_ns: Sequence[int | None] | None = None,
        workers: int | None = None,
    ) -> Runtime:
        nones = [None] * len(nodes)
        records = [
            (node, node.name, 0.0, dedicated, ns, state)
            for node, ns, state in zip(
                nodes, last_update_ns or nones, states or nones, strict=True
            )
        ]
        clock = ReadOnlyTimer(virtual_timer)
        runtime = start(records, clock, manual=manual, workers=workers)
        started.append(runtime)
        return runtime

    yield start_runtime
    for runtime in started:
        runtime.stop()
        with contextlib.suppress(NetworkError):
            runtime.join()


def shut_down(runtime: Runtime) -> None:
    """Stop ``runtime`` and wait for its end, as the host does."""
    runtime.stop()
    runtime.join()


def recorders(log: Log, *names: str) -> list[Recorder]:
    return [Recorder(name, log) for name in names]


def assert_ended_by_exception(log: Log, name: str) -> None:
    """``exception`` then ``stop``, once each, with nothing in between: the
    state is not taken on an exception."""
    events = log.of(name)
    assert events.count("exception") == 1
    assert events.count("stop") == 1
    after = events[events.index("exception") + 1 :]
    assert after == ["stop"]


class TestStart:
    @pytest.mark.parametrize("dedicated", [False, True])
    def test_hooks_run_in_order(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime, dedicated: bool
    ):
        log = Log()
        nodes = recorders(log, "a", "b")
        runtime = start_runtime(
            nodes,
            manual=True,
            states=[{"s": 0}, {"s": 1}],
            dedicated=dedicated,
        )
        virtual_timer.advance(H)
        runtime.step()
        runtime.capture(lambda states: None, WAIT)
        shut_down(runtime)

        for i, node in enumerate(nodes):
            assert log.of(node.name) == [
                "start",
                "restore",
                "update",
                "pause",
                "capture",
                "resume",
                "stop",
            ]
            assert node.loaded == {"s": i}
            if dedicated:
                node_thread = {t for e, t in node.threads.items() if e != "capture"}
                assert len(node_thread) == 1
                assert node.threads["capture"] not in node_thread

    def test_a_state_of_none_is_not_restored(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        log = Log()
        nodes = recorders(log, "a", "b")
        runtime = start_runtime(nodes, manual=True, states=[None, {"s": 1}])
        shut_down(runtime)

        assert log.of("a") == ["start", "stop"]
        assert log.of("b") == ["start", "restore", "stop"]

    @pytest.mark.parametrize(
        ("hook", "dedicated"),
        [("start", True), ("start", False), ("restore", False)],
    )
    def test_an_error_while_starting_stops_before_any_update(
        self,
        virtual_timer: VirtualTimer,
        start_runtime: StartRuntime,
        hook: str,
        dedicated: bool,
    ):
        log = Log()
        nodes = [Recorder("a", log), Recorder("b", log, raise_in=hook)]
        # `start` raises after the network has stopped.
        with pytest.raises(NetworkError, match="'b'") as exc_info:
            start_runtime(nodes, states=[{}, {}], dedicated=dedicated)
        assert exc_info.value.__cause__ is nodes[1].error
        for node in nodes:
            assert "update" not in log.of(node.name)
            assert_ended_by_exception(log, node.name)
        assert not virtual_timer.is_paused()

    def test_the_hooks_of_a_dedicated_node_share_one_python_thread_state(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        # Only a dedicated thread keeps threading.local and current_thread
        # between the hooks; a worker keeps only the contextvars (below).
        local = threading.local()
        var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
            "var", default=None
        )
        seen: dict[str, tuple[object, object, threading.Thread]] = {}

        class ThreadLocal(Recorder):
            @override
            def hook(self, event: str) -> None:
                if event == "start":
                    local.value = self.name
                    var.set(self.name)
                seen[event] = (
                    getattr(local, "value", None),
                    var.get(),
                    threading.current_thread(),
                )
                super().hook(event)

        runtime = start_runtime([ThreadLocal("a", Log())], manual=True, dedicated=True)
        virtual_timer.advance(H)
        runtime.step()
        runtime.capture(lambda states: None, WAIT)
        shut_down(runtime)

        on_start_thread = seen["start"][2]
        for event in ["start", "update", "pause", "resume", "stop"]:
            value, context_value, thread = seen[event]
            assert value == "a", event
            assert context_value == "a", event
            assert thread is on_start_thread, event

    def test_the_control_thread_keeps_one_python_thread_state(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        local = threading.local()
        seen: list[tuple[object, threading.Thread]] = []

        class Remember(Recorder):
            @override
            def capture(self) -> object:
                seen.append((getattr(local, "value", None), threading.current_thread()))
                local.value = "set"
                return super().capture()

        runtime = start_runtime([Remember("a", Log())])

        def build(states: list[object]) -> None:
            seen.append((getattr(local, "value", None), threading.current_thread()))

        for _ in range(2):
            runtime.capture(build, WAIT)
        # capture, build, capture, build: all on one thread state.
        assert [value for value, _ in seen] == [None, "set", "set", "set"]
        assert len({id(thread) for _, thread in seen}) == 1

    def test_restores_where_each_dt_starts_with_the_time_held(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        t_ns, behind_ns, ahead_ns = 5_000_000_000, 3_000_000, 2 * H_NS
        virtual_timer.advance(t_ns / 1e9)
        times: list[tuple[str, int, bool]] = []

        class Restored(Recorder):
            @override
            def hook(self, event: str) -> None:
                if event in ("start", "restore"):
                    now = virtual_timer.now().ns
                    times.append((event, now, virtual_timer.is_paused()))
                super().hook(event)

        log = Log()
        nodes = [Restored("behind", log), Restored("ahead", log)]
        runtime = start_runtime(
            nodes,
            manual=True,
            states=[{}, {}],
            last_update_ns=[t_ns - behind_ns, t_ns + ahead_ns],
        )
        # The time does not run while the nodes start and restore.
        assert times and all(time_ns == t_ns and paused for _, time_ns, paused in times)
        assert virtual_timer.now().ns == t_ns
        virtual_timer.advance(H)
        runtime.step()
        assert nodes[0].dts == [(behind_ns + H_NS) / 1e9]
        # A node whose Δt would start in the future waits until it is positive.
        assert nodes[1].dts == []
        virtual_timer.advance(2 * H)
        runtime.step()
        assert nodes[1].dts == [(3 * H_NS - ahead_ns) / 1e9]

    def test_a_runtime_without_references_stops(self, virtual_timer: VirtualTimer):
        log = Log()
        clock = ReadOnlyTimer(virtual_timer)
        start([(Recorder("a", log), "a", 0.0, False, None, None)], clock)
        gc.collect()
        # Dropping does not wait (the GIL may be held); the stop follows soon.
        log.wait_for(lambda: log.count("a", "stop") == 1)


class TestWorkers:
    def test_the_hooks_run_on_at_most_the_given_number_of_threads(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        idents: list[int] = []

        class Ident(Recorder):
            @override
            def update(self, dt: float) -> None:
                idents.append(threading.get_ident())
                super().update(dt)

        log = Log()
        nodes = [Ident(f"n{k}", log) for k in range(32)]
        runtime = start_runtime(nodes, manual=True, workers=2)
        for _ in range(2):
            virtual_timer.advance(H)
            runtime.step()
        shut_down(runtime)
        assert all(len(node.dts) == 2 for node in nodes)
        assert len(idents) == 64
        assert len(set(idents)) <= 2
        assert threading.get_ident() not in idents

    def test_the_pool_has_no_more_workers_than_nodes(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        runtime = start_runtime(recorders(Log(), "a", "b"), manual=True, workers=8)
        assert runtime.workers == 2

    def test_nodes_all_on_their_own_threads_run_without_a_pool(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        nodes = recorders(Log(), "a", "b")
        runtime = start_runtime(nodes, manual=True, dedicated=True)
        assert runtime.workers == 0
        virtual_timer.advance(H)
        runtime.step()
        shut_down(runtime)
        assert [len(node.dts) for node in nodes] == [1, 1]


# Where the nodes of a test run: (dedicated, workers).
MODES = [(True, None), (False, None), (False, 1), (False, 2)]


class TestOneCallPerHook:
    @pytest.mark.parametrize(("dedicated", "workers"), MODES)
    def test_each_hook_is_one_call_of_the_driver(
        self,
        virtual_timer: VirtualTimer,
        start_runtime: StartRuntime,
        dedicated: bool,
        workers: int | None,
    ):
        log = Log()
        nodes = recorders(log, "a", "b")
        runtime = start_runtime(
            nodes,
            manual=True,
            dedicated=dedicated,
            workers=workers,
            states=[{"s": 0}, {"s": 1}],
        )
        for order in ([0, 1], [1, 0]):
            virtual_timer.advance(H)
            runtime.step(order)
        runtime.capture(lambda states: None, WAIT)
        shut_down(runtime)
        for node in nodes:
            assert log.of(node.name) == [
                "start",
                "restore",
                "update",
                "update",
                "pause",
                "capture",
                "resume",
                "stop",
            ]

    @pytest.mark.parametrize(("dedicated", "workers"), MODES)
    def test_an_exception_is_one_call_too(
        self,
        virtual_timer: VirtualTimer,
        start_runtime: StartRuntime,
        dedicated: bool,
        workers: int | None,
    ):
        log = Log()
        nodes = [Recorder("a", log, raise_in="update"), Recorder("b", log)]
        runtime = start_runtime(
            nodes, manual=True, dedicated=dedicated, workers=workers
        )
        virtual_timer.advance(H)
        with pytest.raises(NetworkError, match="raised in on_update$"):
            runtime.step([0])
        assert log.of("a") == ["start", "update", "exception", "stop"]
        assert log.of("b") == ["start", "exception", "stop"]

    def test_an_error_in_restore_names_load_state_dict(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        node = Recorder("a", Log(), raise_in="restore")
        with pytest.raises(NetworkError, match="raised in load_state_dict$"):
            start_runtime([node], manual=True, states=[{}])

    def test_an_error_in_capture_names_state_dict(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        node = Recorder("a", Log(), raise_in="capture")
        runtime = start_runtime([node], manual=True)
        with pytest.raises(NetworkError, match="raised in state_dict$"):
            runtime.capture(lambda states: states, WAIT)


class TestCurrent:
    def test_is_the_driver_in_its_methods_and_none_outside(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        seen: list[tuple[str, object]] = []

        class Sees(Recorder):
            @override
            def hook(self, event: str) -> None:
                seen.append((event, current()))
                super().hook(event)

        node = Sees("a", Log())
        runtime = start_runtime([node], manual=True)
        virtual_timer.advance(H)
        runtime.step()
        runtime.capture(lambda states: states, WAIT)
        shut_down(runtime)
        # After the end, on the calling thread.
        runtime.capture(lambda states: states, WAIT)
        assert [event for event, _ in seen] == [
            "start",
            "update",
            "pause",
            "capture",
            "resume",
            "stop",
            "capture",
        ]
        for _, running in seen:
            assert running is node
        assert current() is None


class TestCaptureAfterTheEnd:
    def test_takes_the_states_on_the_calling_thread(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        log = Log()
        node = Recorder("a", log)
        runtime = start_runtime([node], manual=True)
        virtual_timer.advance(H)
        runtime.step()
        shut_down(runtime)
        states, thread = runtime.capture(
            lambda states: (states, threading.get_ident()), WAIT
        )
        assert states == [{"name": "a", "updates": 1}]
        assert thread == node.threads["capture"] == threading.get_ident()
        # Nothing else of the node ran.
        assert log.of("a")[-2:] == ["stop", "capture"]

    def test_an_error_in_state_dict_is_raised_and_ends_nothing_more(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        node = Recorder("a", Log(), raise_in="capture")
        runtime = start_runtime([node], manual=True)
        shut_down(runtime)
        # The node's own exception: the network has not stopped because of it.
        with pytest.raises(ValueError, match="a raised in capture") as raised:
            runtime.capture(lambda states: states, WAIT)
        assert raised.value is node.error
        runtime.check()


class TestFreeRun:
    def test_dt_sums_to_the_time_since_the_start(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        log = Log()
        nodes = recorders(log, "a", "b")
        runtime = start_runtime(nodes)
        t0 = virtual_timer.now().ns
        advances = [0.001, 0.0025, 3e-4, 0.1, 1e-9]
        for k, seconds in enumerate(advances, start=1):
            virtual_timer.advance(seconds)
            # Each node updates exactly once per advance: with Δt = 0 it waits.
            log.wait_for(lambda k=k: all(len(n.dts) >= k for n in nodes))
        shut_down(runtime)

        for node, last_ns in zip(nodes, runtime.last_update_ns(), strict=True):
            assert len(node.dts) == len(advances)
            assert last_ns == virtual_timer.now().ns
            assert sum(node.dts) == pytest.approx((last_ns - t0) / 1e9, rel=1e-12)

    def test_waits_without_spinning_until_the_time_moves(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        log = Log()
        runtime = start_runtime(recorders(log, "a"))
        cpu_start = time.process_time()
        time.sleep(0.2)
        # A node spinning on Δt = 0 would use a core for the whole 0.2 s.
        assert time.process_time() - cpu_start < 0.05
        assert log.count("a", "update") == 0
        # The waiting node wakes on control requests, not only on the time.
        states = runtime.capture(lambda states: states, WAIT)
        assert states == [{"name": "a", "updates": 0}]
        start = time.monotonic()
        shut_down(runtime)
        assert time.monotonic() - start < 1.0
        assert log.count("a", "update") == 0


class TestCapture:
    def test_gives_up_when_a_node_does_not_pause(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        entered, release = threading.Event(), threading.Event()

        class Stuck(Recorder):
            @override
            def update(self, dt: float) -> None:
                super().update(dt)
                entered.set()
                assert release.wait(WAIT)

        log = Log()
        stuck, other = Stuck("stuck", log), Recorder("other", log)
        runtime = start_runtime([stuck, other], workers=2)
        virtual_timer.advance(H)
        assert entered.wait(WAIT)
        log.wait_for(lambda: len(other.dts) == 1)

        with pytest.raises(TimeoutError, match="stuck") as exc_info:
            runtime.capture(lambda states: None, 0.1)
        assert "other" not in str(exc_info.value)
        # The network keeps running on a running timer.
        assert runtime.join(timeout=0) is False
        assert not virtual_timer.is_paused()
        log.wait_for(lambda: log.of("other")[-1] == "resume")
        release.set()
        virtual_timer.advance(H)
        log.wait_for(lambda: len(other.dts) == 2 and len(stuck.dts) == 2)
        shut_down(runtime)

    def test_an_error_in_build_propagates_and_the_network_keeps_running(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        runtime = start_runtime(recorders(Log(), "a"))

        def build(states: list[object]) -> None:
            raise KeyError("build")

        with pytest.raises(KeyError, match="build"):
            runtime.capture(build, WAIT)
        runtime.check()
        assert runtime.join(timeout=0) is False
        assert not virtual_timer.is_paused()

    def test_build_cannot_wait_for_the_runtime(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        runtime = start_runtime(recorders(Log(), "a"))
        # Waiting from the control thread for the control thread would deadlock.
        with pytest.raises(RuntimeError, match="own threads"):
            runtime.capture(lambda states: runtime.join(), WAIT)
        assert runtime.join(timeout=0) is False


class TestException:
    @pytest.mark.parametrize("hook", ["pause", "capture", "resume"])
    def test_an_error_around_a_capture_stops_every_node(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime, hook: str
    ):
        log = Log()
        nodes = [Recorder("a", log), Recorder("x", log, raise_in=hook)]
        runtime = start_runtime(nodes)
        # `resume` raises after the capture has returned, so wait for the end.
        with pytest.raises(NetworkError) as exc_info:
            runtime.capture(lambda states: None, WAIT)
            runtime.join(WAIT)
        assert exc_info.value.__cause__ is nodes[1].error
        # The hold of the capture is released at the end.
        assert not virtual_timer.is_paused()
        for node in nodes:
            assert_ended_by_exception(log, node.name)
            if hook != "resume":
                # Stopping while paused does not resume.
                assert "resume" not in log.of(node.name)

    def test_only_the_first_of_two_exceptions_is_the_cause(
        self,
        virtual_timer: VirtualTimer,
        start_runtime: StartRuntime,
        caplog: pytest.LogCaptureFixture,
    ):
        both_updating = threading.Barrier(2, timeout=WAIT)

        class Simultaneous(Recorder):
            @override
            def update(self, dt: float) -> None:
                both_updating.wait()
                super().update(dt)

        log = Log()
        nodes = [Simultaneous(n, log, "update") for n in ["a", "b"]]
        runtime = start_runtime(nodes, workers=2)
        virtual_timer.advance(H)
        with pytest.raises(NetworkError) as exc_info:
            runtime.join(WAIT)
        errors = [node.error for node in nodes]
        assert any(exc_info.value.__cause__ is e for e in errors)
        assert "node 'a' raised in on_update" in caplog.text
        assert "node 'b' raised in on_update" in caplog.text

    def test_an_error_in_on_stop_is_logged_and_the_others_still_stop(
        self,
        virtual_timer: VirtualTimer,
        start_runtime: StartRuntime,
        caplog: pytest.LogCaptureFixture,
    ):
        log = Log()
        nodes = [Recorder("a", log, "stop"), Recorder("b", log)]
        runtime = start_runtime(nodes)
        shut_down(runtime)
        assert log.count("b", "stop") == 1
        assert "node 'a' raised in on_stop" in caplog.text
        runtime.check()
        assert runtime.join()


class TestWaiting:
    def test_join_waits_without_holding_the_gil(
        self, virtual_timer: VirtualTimer, start_runtime: StartRuntime
    ):
        log = Log()
        runtime = start_runtime(recorders(log, "a"))
        assert runtime.join(timeout=0.05) is False

        def advance_and_stop() -> None:
            # On a GIL build this thread runs only if `join` released the GIL.
            virtual_timer.advance(H)
            log.wait_for(lambda: log.count("a", "update") == 1)
            shut_down(runtime)

        thread = threading.Thread(target=advance_and_stop)
        thread.start()
        assert runtime.join()
        thread.join(WAIT)
        assert runtime.finished
