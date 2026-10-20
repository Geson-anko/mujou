import copy
import math
import pickle
import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast, override

import pytest
import torch
from torch import nn

from mujou import Group, NetworkError, Node, Port, TensorSpec
from mujou._core import _native
from mujou.testing import VirtualTimer
from mujou.time import ReadOnlyTimer, Timer, Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual, Step


class Probe(Node):
    """Overrides ``on_start`` without calling the base, and records the thread's
    settings there."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        init_base: bool = True,
        dedicated_thread: bool = False,
    ) -> None:
        if init_base:
            super().__init__(seed=seed, dedicated_thread=dedicated_thread)
            self.out = Port(send=TensorSpec((3,)))
        self.grad_enabled: bool | None = None
        self.draws: torch.Tensor | None = None
        self.ident: int | None = None

    @override
    def on_start(self) -> None:
        self.grad_enabled = torch.is_grad_enabled()
        self.draws = torch.rand(3, generator=self.generator)
        self.ident = threading.get_ident()


# The tests run the networks in manual mode only so that idle nodes do not spin:
# on_start runs either way.


class TestThreadSetup:
    def test_on_start_runs_on_a_worker_recording_gradients(
        self, start_manual: StartManual
    ):
        # specs: the framework does not change gradient recording; a new thread
        # records gradients, as in plain PyTorch.
        node = Probe(seed=0)
        start_manual(Group(node=node))
        assert node.grad_enabled is True
        assert node.ident not in (None, threading.get_ident())

    def test_the_generator_follows_the_seed(self, start_manual: StartManual):
        a, b, c = Probe(seed=7), Probe(seed=7), Probe(seed=8)
        for node in (a, b, c):
            start_manual(Group(node=node))
        assert a.draws is not None and b.draws is not None and c.draws is not None
        expected = torch.rand(3, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a.draws, expected)
        assert torch.equal(b.draws, expected)
        assert not torch.equal(c.draws, expected)

    def test_works_without_calling_the_base_init(self, start_manual: StartManual):
        node = Probe(init_base=False)
        start_manual(Group(node=node))
        assert node.device == torch.device("cpu")
        assert node.draws is not None

    def test_leaves_the_hooks_of_the_node_alone(self, start_manual: StartManual):
        node = Probe(seed=0)
        run = start_manual(Group(node=node))
        run.state_dict()
        assert not {"on_update", "state_dict"} & set(vars(node))

    def test_the_generator_is_ready_in_init(self):
        node = InitDraws(seed=7)
        expected = torch.rand(3, generator=torch.Generator().manual_seed(7))
        assert torch.equal(node.draws_in_init, expected)

    def test_reading_the_generator_before_the_base_init_raises(self):
        class Early(Node):
            def __init__(self) -> None:
                _ = self.generator
                super().__init__()

        with pytest.raises(RuntimeError, match=r"super\(\).__init__"):
            Early()

    def test_on_start_continues_the_sequence_from_init(self, start_manual: StartManual):
        node = InitDraws(seed=7)
        start_manual(Group(node=node))
        generator = torch.Generator().manual_seed(7)
        _ = torch.rand(3, generator=generator)  # what __init__ drew
        assert node.draws is not None
        assert torch.equal(node.draws, torch.rand(3, generator=generator))

    def test_a_hook_backpropagates_without_turning_gradients_on(
        self, step: Step, start_manual: StartManual
    ):
        node = Backprop()
        run = start_manual(Group(node=node))
        step(run)
        run.shutdown()
        assert node.grad is not None
        assert torch.equal(node.grad, torch.full((3,), 2.0))  # d(sum(2w))/dw

    def test_leaves_what_a_hook_set_to_the_next_hook_on_the_thread(
        self, virtual: VirtualTimer, start: Start
    ):
        # specs: the framework does not set gradient recording back, like any
        # other per-thread setting: a hook that turns it off without ``with``
        # turns it off for the next hook on the same worker too.
        leaky = GradSetter(leak=True)
        run = start(Group(leaky=leaky), timer=virtual, workers=1)
        for _ in range(3):
            virtual.advance(0.01)
            leaky.updated.get(timeout=5.0)
        run.shutdown()
        assert leaky.grad_enabled == {
            "on_start": [True],
            "on_update": [False, False, False],
            "on_stop": [False],
        }
        # The setting is per thread: the host still records gradients.
        assert threading.get_ident() not in leaky.idents
        assert torch.is_grad_enabled()

    def test_a_node_whose_setup_failed_cleans_up_as_itself(
        self, virtual: VirtualTimer, start: Start
    ):
        # specs: when the framework's setup of a node fails, the runtime still calls
        # on_exception and on_stop; they run as that node (not as the node the
        # worker ran before). BrokenSetup fails through a
        # private override: the real failure (setting up CUDA) needs a GPU, and no
        # public way makes the setup fail.
        other = GradSetter(leak=False)
        broken = BrokenSetup(other.inp)
        with pytest.raises(NetworkError, match="raised in on_start"):
            start(Group(other=other, broken=broken), timer=virtual, workers=1)
        assert set(broken.results) == {"on_exception", "on_stop"}
        for hook, result in broken.results.items():
            assert isinstance(result, RuntimeError), hook
            assert "only the node that owns" in str(result)


class Backprop(Node):
    """Backpropagates through ``sum(2 * w)`` in each update, as in plain PyTorch."""

    def __init__(self) -> None:
        super().__init__(period=0.0)
        self.grad: torch.Tensor | None = None

    @override
    def on_start(self) -> None:
        self.w = torch.ones(3, requires_grad=True)

    @override
    def on_update(self, dt: float) -> None:
        self.w.grad = None
        (2 * self.w).sum().backward()
        self.grad = self.w.grad


class BrokenSetup(Node):
    """Fails in the framework's setup (creating the generator) every time a
    thread enters it, and tries to read another node's port while it cleans up."""

    failing = False  # not while super().__init__ creates the generator

    def __init__(self, other: Port[Any, Any, Any]) -> None:
        super().__init__(period=0.01)
        self.out = Port(send=TensorSpec((1,)))
        # From now on the setup creates the generator again, and fails.
        self._generator = None
        self.failing = True
        vars(self)["other"] = other  # not registered as one of its ports
        self.results: dict[str, object] = {}

    @override
    def _make_generator(self) -> torch.Generator:
        if self.failing:
            raise RuntimeError("setup failed")
        return super()._make_generator()

    def clean_up(self, hook: str) -> None:
        other = cast(Port[Any, Any, Any], vars(self)["other"])
        try:
            result: object = other.recv()
        except RuntimeError as e:
            result = e
        self.results[hook] = result

    @override
    def on_exception(self) -> None:
        self.clean_up("on_exception")

    @override
    def on_stop(self) -> None:
        self.clean_up("on_stop")


class GradSetter(Node):
    """Records whether gradients are recorded at the entry of each hook, and,
    with ``leak``, turns them off without ``with`` at the end of each hook."""

    def __init__(self, *, leak: bool) -> None:
        super().__init__(period=0.01)
        self.out = Port(send=TensorSpec((1,)))
        self.inp = Port(recv=TensorSpec((1,)))
        self.leak = leak
        self.grad_enabled: dict[str, list[bool]] = {}
        self.idents: set[int] = set()
        self.updated: queue.Queue[None] = queue.Queue()

    def record(self, hook: str) -> None:
        self.grad_enabled.setdefault(hook, []).append(torch.is_grad_enabled())
        self.idents.add(threading.get_ident())
        if self.leak:
            torch.set_grad_enabled(False)

    @override
    def on_start(self) -> None:
        self.record("on_start")

    @override
    def on_update(self, dt: float) -> None:
        self.record("on_update")
        self.updated.put(None)

    @override
    def on_stop(self) -> None:
        self.record("on_stop")


class InitDraws(Probe):
    """Draws from ``generator`` in ``__init__``, right after the base's."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.draws_in_init = torch.rand(3, generator=self.generator)


def reachable(root: object) -> list[object]:
    """Every object reached from ``root`` through instance attributes, the
    object a builtin or bound method is bound to, and closure cells."""
    found: list[object] = []
    todo = [root]
    while todo:
        obj = todo.pop()
        if any(obj is f for f in found) or isinstance(obj, type):
            continue
        found.append(obj)
        todo.extend(getattr(obj, "__dict__", {}).values())
        for name in ("__self__", "__func__", "__wrapped__"):
            if hasattr(obj, name):
                todo.append(getattr(obj, name))
        for cell in getattr(obj, "__closure__", None) or ():
            todo.append(cell.cell_contents)
    return found


class Timed(Node):
    """Keeps the time of its run's timer in each hook."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=TensorSpec((3,)))
        self.timers: list[ReadOnlyTimer] = []
        self.times: list[Timestamp] = []

    def keep(self) -> None:
        self.timers.append(self.timer)
        self.times.append(self.timer.now())

    @override
    def on_start(self) -> None:
        self.keep()

    @override
    def on_update(self, dt: float) -> None:
        self.keep()

    @override
    def on_stop(self) -> None:
        self.keep()

    @override
    def state_dict(self) -> object:
        return self.timer.now()


class TestTimer:
    def test_is_the_timer_of_the_run_from_on_start_to_on_stop(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Timed()
        virtual.advance(1.0)
        run = start_manual(Group(node=node), timer=virtual)
        virtual.advance(0.5)
        run.step()
        run.shutdown()
        assert all(isinstance(timer, ReadOnlyTimer) for timer in node.timers)
        assert node.times == [
            Timestamp(ns=1_000_000_000),
            Timestamp(ns=1_500_000_000),
            Timestamp(ns=1_500_000_000),
        ]

    @pytest.mark.parametrize("ended", [False, True])
    def test_reaches_nothing_that_changes_the_timer(
        self, virtual: VirtualTimer, start_manual: StartManual, ended: bool
    ):
        node = Timed()
        seen: list[ReadOnlyTimer] = []
        node.state_dict = lambda: seen.append(node.timer)  # type: ignore[method-assign]
        run = start_manual(Group(node=node), timer=virtual)
        if ended:
            run.shutdown()  # the state_dict after the end sees a stopped timer
        run.state_dict()
        (timer,) = seen
        # Only the reading methods, public or private, on the instance.
        assert set(vars(timer)) == {"now", "get_time_scale"}
        assert timer.now() == virtual.now()
        assert timer.get_time_scale() == 1.0
        # Nothing reached from them (bound objects, closures) is a timer. The
        # class's own functions still lead to module globals, as any class's do.
        for obj in reachable(timer):
            assert not isinstance(obj, Timer | _native.Timer), obj
            assert not hasattr(obj, "set_time_scale"), obj

    def test_stays_at_the_time_of_the_state_after_a_normal_end(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Timed()
        run = start_manual(Group(node=node), timer=virtual)
        virtual.advance(0.5)
        run.step()
        run.shutdown()
        virtual.advance(100.0)  # after the end: not part of the state
        tree = run.state_dict()
        assert tree["nodes"]["node"] == Timestamp(ns=tree["timer"]["time_ns"])
        assert tree["timer"]["time_ns"] == 500_000_000
        assert node.timer.now() == Timestamp(ns=500_000_000)

    def test_stays_stopped_while_another_thread_reads_the_run(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        class Slow(Timed):
            @override
            def state_dict(self) -> object:
                time.sleep(0.05)  # while the host's other thread reads
                return self.timer.now()

        node = Slow()
        run = start_manual(Group(node=node), timer=virtual)
        run.shutdown()
        done = threading.Event()

        def read() -> None:
            while not done.is_set():
                run.read(node.out)

        reader = threading.Thread(target=read)
        reader.start()
        try:
            for _ in range(2):
                assert run.state_dict()["nodes"]["node"] == Timestamp(ns=0)
        finally:
            done.set()
            reader.join()

    def test_raises_before_a_run_and_after_an_end_by_an_exception(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        class Fails(Timed):
            @override
            def on_update(self, dt: float) -> None:
                raise ValueError("boom")

        node = Fails()
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer
        run = start_manual(Group(node=node), timer=virtual)
        virtual.advance(0.001)
        with pytest.raises(NetworkError):
            run.step()
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer

    def test_is_released_when_a_run_is_refused(self, start_manual: StartManual):
        tree = start_manual(Group(node=Timed())).state_dict()
        node = Timed()
        with pytest.raises(ValueError):
            # Refused at the start: the timer of the run that is still running
            # cannot be set back.
            start_manual(Group(node=node), state=tree)
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer


class TestDevice:
    def test_defaults_to_the_cpu(self):
        assert Probe().device == torch.device("cpu")

    def test_takes_a_string(self):
        class OnCpu(Node):
            def __init__(self) -> None:
                super().__init__(device="cpu")

        assert OnCpu().device == torch.device("cpu")

    @pytest.mark.skipif(torch.cuda.is_available(), reason="needs no CUDA")
    def test_rejects_cuda_without_cuda(self):
        with pytest.raises(ValueError, match="CUDA is not available"):
            Node(device="cuda")

    @pytest.mark.cuda
    def test_fixes_the_cuda_index(self):
        assert Node(device="cuda").device.index == torch.cuda.current_device()

    # Rejected whether or not torch can use them here: MPS crashes when two
    # threads use it, and the others are not supported.
    @pytest.mark.parametrize("device", ["meta", "mps", "xpu", "mps:0"])
    def test_rejects_devices_other_than_the_cpu_and_cuda(self, device: str):
        with pytest.raises(ValueError, match=r"CPU or CUDA \(only these"):
            Node(device=device)


class Ticker(Node):
    """Puts each ``dt`` it updates with into ``dts``."""

    def __init__(self, period: float) -> None:
        super().__init__(period=period)
        self.out = Port(send=TensorSpec((1,)))
        self.dts: queue.Queue[float] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        self.dts.put(dt)


class TestPeriod:
    def test_defaults_to_ten_milliseconds(self):
        assert Probe().period == 0.01
        assert Probe(init_base=False).period == 0.01

    @pytest.mark.parametrize("period", [0.0, 0.5])
    def test_is_kept(self, period: float):
        assert Node(period=period).period == period

    @pytest.mark.parametrize("period", [-0.01, math.inf, math.nan])
    def test_rejects_what_is_negative_or_not_finite(self, period: float):
        with pytest.raises(ValueError, match="period"):
            Node(period=period)

    def test_rejects_a_bool(self):
        with pytest.raises(TypeError, match="period"):
            Node(period=True)

    def test_is_a_reserved_name(self, virtual: VirtualTimer, start: Start):
        class OnTheClass(Ticker):
            period = 0.5  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]

        with pytest.raises(TypeError, match="reserved"):
            start(Group(node=OnTheClass(period=0.0)), timer=virtual)

    def test_the_node_waits_for_its_period_on_the_timer(
        self, virtual: VirtualTimer, start: Start
    ):
        # Free running: a manual run (mujou.testing.run) ignores the period.
        node = Ticker(period=0.01)
        start(Group(node=node), timer=virtual)
        virtual.advance(0.005)
        with pytest.raises(queue.Empty):
            node.dts.get(timeout=0.05)
        virtual.advance(0.005)
        assert node.dts.get(timeout=5.0) == 0.01

    def test_a_period_of_zero_updates_as_soon_as_the_time_moves(
        self, virtual: VirtualTimer, start: Start
    ):
        node = Ticker(period=0.0)
        start(Group(node=node), timer=virtual)
        virtual.advance(1e-9)
        assert node.dts.get(timeout=5.0) == 1e-9

    def test_a_period_below_a_nanosecond_is_one_nanosecond(
        self, virtual: VirtualTimer, start: Start
    ):
        node = Ticker(period=1e-10)
        start(Group(node=node), timer=virtual)
        virtual.advance(1e-9)
        assert node.dts.get(timeout=5.0) == 1e-9


class TestDedicatedThread:
    def test_defaults_to_the_workers(self):
        assert Probe().dedicated_thread is False
        assert Probe(init_base=False).dedicated_thread is False

    @pytest.mark.parametrize("dedicated", [1, None, "yes"])
    def test_rejects_what_is_not_a_bool(self, dedicated: object):
        with pytest.raises(TypeError, match="dedicated_thread"):
            Node(dedicated_thread=dedicated)  # pyright: ignore[reportArgumentType]

    def test_is_a_reserved_name(self, virtual: VirtualTimer, start: Start):
        class OnTheClass(Ticker):
            dedicated_thread = True  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]

        with pytest.raises(TypeError, match="reserved"):
            start(Group(node=OnTheClass(period=0.0)), timer=virtual)

    def test_a_dedicated_node_runs_every_hook_on_one_thread_of_its_own(
        self, step: Step, start_manual: StartManual
    ):
        dedicated = ThreadRecorder(dedicated_thread=True)
        others = [ThreadRecorder() for _ in range(3)]
        run = start_manual(Group(dedicated, *others))
        for _ in range(3):
            step(run)
        run.shutdown()
        assert len(dedicated.idents) == 1
        assert not dedicated.idents & set().union(*(o.idents for o in others))


class ThreadRecorder(Node):
    """Records the OS thread of each hook."""

    def __init__(self, *, dedicated_thread: bool = False) -> None:
        super().__init__(dedicated_thread=dedicated_thread)
        self.idents: set[int] = set()

    @override
    def on_start(self) -> None:
        self.idents.add(threading.get_ident())

    @override
    def on_update(self, dt: float) -> None:
        self.idents.add(threading.get_ident())

    @override
    def on_stop(self) -> None:
        self.idents.add(threading.get_ident())


class Stateful(Node):
    """Holds a public and a private tensor, a parameter that records gradients, a
    module and values that are not state."""

    def __init__(self, fill: float) -> None:
        super().__init__()
        self.out = Port(send=TensorSpec((3,)))
        self.w = torch.full((2, 3), fill)
        self._h = torch.full((3,), fill + 1)
        self.p = torch.full((3,), fill + 2, requires_grad=True)
        self.linear = nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.fill_(fill + 3)
            self.linear.bias.fill_(fill + 4)
        self.rate = 0.5
        self.history: list[torch.Tensor] = []


def at(state: object, *path: str) -> torch.Tensor:
    """The tensor at ``path`` in a tree of mappings."""
    for key in path:
        state = cast(Mapping[str, object], state)[key]
    assert isinstance(state, torch.Tensor)
    return state


class TestStateCollection:
    def test_collects_tensors_and_modules_by_attribute_name(self):
        node = Stateful(0.0)
        state = node.state_dict()
        assert list(state) == ["w", "_h", "p", "linear"]
        assert torch.equal(at(state, "w"), node.w)
        assert torch.equal(at(state, "_h"), node._h)  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(at(state, "p"), node.p.detach())
        assert set(cast(Mapping[str, object], state["linear"])) == {"weight", "bias"}
        assert torch.equal(at(state, "linear", "weight"), node.linear.weight.detach())

    def test_the_tree_holds_detached_views_of_the_live_tensors(self):
        node = Stateful(0.0)
        state = node.state_dict()
        p, weight = at(state, "p"), at(state, "linear", "weight")
        assert not p.requires_grad and not weight.requires_grad
        assert p.data_ptr() == node.p.data_ptr()
        assert weight.data_ptr() == node.linear.weight.data_ptr()

    def test_the_tree_pickles_even_with_a_tensor_in_a_graph(self):
        node = Stateful(0.0)
        y = node.p * 2
        assert y.grad_fn is not None
        setattr(node, "y", y)  # noqa: B010  # not declared by Stateful
        state = pickle.loads(pickle.dumps(node.state_dict()))
        assert torch.equal(at(state, "y"), torch.full((3,), 4.0))

    def test_follows_reassignment_and_deletion(self):
        node = Stateful(0.0)
        node.w = torch.ones(1)  # another tensor under the same name
        setattr(node, "_h", None)  # noqa: B010  # not the declared type
        del node.p
        setattr(node, "linear", nn.Identity())  # noqa: B010  # a module without state
        setattr(node, "rate", torch.tensor(0.25))  # noqa: B010  # becomes a tensor
        state = node.state_dict()
        assert list(state) == ["w", "linear", "rate"]
        assert torch.equal(at(state, "w"), torch.ones(1))
        assert state["linear"] == {}
        node._h = torch.zeros(3)  # pyright: ignore[reportPrivateUsage]  # registered again, at the end
        assert list(node.state_dict()) == ["w", "linear", "rate", "_h"]

    def test_a_node_without_tensors_has_an_empty_state(self):
        node = Probe()  # draws is None until on_start
        assert node.state_dict() == {}
        node.load_state_dict({})

    def test_loads_in_place_into_the_same_objects(self):
        source, target = Stateful(1.0), Stateful(-1.0)
        before = [target.w, target._h, target.p, target.linear.weight]  # pyright: ignore[reportPrivateUsage]
        target.load_state_dict(copy.deepcopy(source.state_dict()))
        after = [target.w, target._h, target.p, target.linear.weight]  # pyright: ignore[reportPrivateUsage]
        assert all(a is b for a, b in zip(after, before, strict=True))
        assert torch.equal(target.w, source.w)
        assert torch.equal(target._h, source._h)  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(target.p, source.p)
        assert torch.equal(target.linear.weight, source.linear.weight)
        assert torch.equal(target.linear.bias, source.linear.bias)
        # The parameter is still a leaf that records gradients.
        assert target.p.requires_grad and target.p.is_leaf

    @pytest.mark.parametrize(
        ("change", "error", "match"),
        [
            (lambda s: s.pop("p"), ValueError, r"missing \['p'\]$"),
            (
                lambda s: s.update(extra=[1.0]),
                ValueError,
                r"not tensors and not registered \['extra'\]$",
            ),
            (lambda s: s.update(p=torch.zeros(2)), ValueError, "'p'.*shape"),
            (lambda s: s.update(p=[1.0, 2.0, 3.0]), TypeError, "'p'.*not a tensor"),
            (
                lambda s: cast(dict[str, object], s["linear"]).pop("bias"),
                RuntimeError,
                r"'linear'.*missing \['bias'\]",
            ),
            (
                lambda s: cast(dict[str, object], s["linear"]).update(
                    weight=torch.zeros(1)
                ),
                RuntimeError,
                r"'linear'.*other shapes \['weight'\]",
            ),
        ],
    )
    def test_rejects_a_tree_that_does_not_match_and_changes_nothing(
        self,
        change: Callable[[dict[str, object]], object],
        error: type[Exception],
        match: str,
    ):
        source, target = Stateful(1.0), Stateful(-1.0)
        state = copy.deepcopy(source.state_dict())
        change(state)
        with pytest.raises(error, match=match):
            target.load_state_dict(state)
        # Nothing changed, also before the item that does not match.
        assert torch.equal(target.w, torch.full((2, 3), -1.0))
        assert torch.equal(target.p, torch.full((3,), 1.0))
        assert torch.equal(target.linear.weight, torch.full((2, 3), 2.0))
        assert "extra" not in vars(target)

    def test_rejects_a_tree_that_is_not_a_mapping(self):
        with pytest.raises(TypeError, match="mapping"):
            Stateful(0.0).load_state_dict([1, 2])

    def test_sets_a_tensor_that_is_not_registered_yet(self):
        source, target = Stateful(1.0), Stateful(-1.0)
        setattr(source, "last", torch.arange(3.0))  # noqa: B010  # not declared
        state = copy.deepcopy(source.state_dict())
        target.load_state_dict(state)
        last = getattr(target, "last")  # noqa: B009  # not declared
        assert torch.equal(last, torch.arange(3.0))
        assert last is not state["last"]  # a copy
        assert list(target.state_dict()) == ["w", "_h", "p", "linear", "last"]

    def test_rejects_a_tensor_whose_elements_share_memory(self):
        source, target = Stateful(1.0), Stateful(-1.0)
        target.w = torch.zeros(3).expand(2, 3)
        with pytest.raises(ValueError, match="'w'.*share memory"):
            target.load_state_dict(copy.deepcopy(source.state_dict()))

    def test_does_not_collect_a_property(self):
        class WithProperty(Node):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.ones(2)

            @property
            def w(self) -> torch.Tensor:
                return torch.tensor(self._w)

            @w.setter
            def w(self, value: torch.Tensor) -> None:
                self._w = value.tolist()  # stored as a list: not state

        node = WithProperty()
        assert node.state_dict() == {}

    def test_a_shallow_copy_has_its_own_registry(self):
        node = Stateful(0.0)
        clone = copy.copy(node)
        setattr(clone, "z", torch.zeros(1))  # noqa: B010  # not declared
        del clone.w
        assert list(node.state_dict()) == ["w", "_h", "p", "linear"]
        assert list(clone.state_dict()) == ["_h", "p", "linear", "z"]


class Counting(Stateful):
    """``Stateful`` with plain values registered as state: ``count`` between the
    tensors and the module, and ``log`` (a list) at the end."""

    def __init__(self, fill: float) -> None:
        super().__init__(fill)
        self.count = int(fill)
        self.register_state("count")
        self.log = [fill]
        self.register_state("log")


class TestRegisterState:
    def test_adds_any_value_to_the_state_in_the_order_of_registration(self):
        node = Counting(2.0)
        state = node.state_dict()
        assert list(state) == ["w", "_h", "p", "linear", "count", "log"]
        assert state["count"] == 2 and state["log"] == [2.0]

    def test_follows_reassignment_of_any_value(self):
        node = Counting(2.0)
        node.count = 7
        node.log = []
        setattr(node, "count", torch.ones(1))  # noqa: B010  # not the declared type
        setattr(node, "count", None)  # noqa: B010  # still registered
        state = node.state_dict()
        assert state["count"] is None and state["log"] == []
        assert list(state) == ["w", "_h", "p", "linear", "count", "log"]

    def test_deletion_forgets_it(self):
        node = Counting(2.0)
        del node.count
        assert list(node.state_dict()) == ["w", "_h", "p", "linear", "log"]
        node.count = 3  # a plain value again: not state until registered
        assert "count" not in node.state_dict()

    def test_registering_twice_keeps_the_order(self):
        node = Counting(2.0)
        node.register_state("count")
        node.register_state("w")  # a tensor, already collected
        assert list(node.state_dict()) == ["w", "_h", "p", "linear", "count", "log"]

    def test_registering_a_collected_name_keeps_it_on_a_plain_value(self):
        node = Counting(2.0)
        node.register_state("w")
        setattr(node, "w", None)  # noqa: B010  # not the declared type
        assert node.state_dict()["w"] is None
        assert list(node.state_dict()) == ["w", "_h", "p", "linear", "count", "log"]

    def test_a_registered_tensor_is_restored_in_place(self):
        source, target = Counting(1.0), Counting(-1.0)
        source.register_state("w")
        target.register_state("w")
        w = target.w
        target.load_state_dict(copy.deepcopy(source.state_dict()))
        assert target.w is w
        assert torch.equal(w, torch.ones(2, 3))

    def test_loads_by_assigning_a_copy(self):
        source, target = Counting(1.0), Counting(-1.0)
        source.count, source.log = 5, [1.0, 2.0]
        state = copy.deepcopy(source.state_dict())
        target.load_state_dict(state)
        assert target.count == 5 and target.log == [1.0, 2.0]
        assert target.log is not state["log"]  # a copy
        assert torch.equal(target.w, source.w)

    def test_a_missing_value_is_rejected_and_nothing_changes(self):
        source, target = Counting(1.0), Counting(-1.0)
        state = copy.deepcopy(source.state_dict())
        del state["log"]
        with pytest.raises(ValueError, match=r"missing \['log'\]$"):
            target.load_state_dict(state)
        assert target.count == -1 and torch.equal(target.w, torch.full((2, 3), -1.0))

    def test_rejects_a_name_without_an_attribute(self):
        node = Stateful(0.0)
        with pytest.raises(AttributeError, match="'steps'.*assign"):
            node.register_state("steps")

    @pytest.mark.parametrize("name", ["device", "period", "timer"])
    def test_rejects_a_property(self, name: str):
        with pytest.raises(ValueError, match=f"'{name}'.*property"):
            Stateful(0.0).register_state(name)

    def test_a_shallow_copy_has_its_own_registry(self):
        node = Stateful(0.0)
        clone = copy.copy(node)
        clone.register_state("rate")
        assert "rate" not in node.state_dict()
        assert clone.state_dict()["rate"] == 0.5


INPUT = torch.tensor([1.0, -1.0, 0.5])
TARGET = torch.tensor([0.5, 2.0])


class Learner(Node):
    """Fits ``linear`` to ``TARGET`` by backpropagation inside the node, and relaxes
    ``_x`` toward 1 with momentum, both in proportion to ``dt``. ``_last`` is first
    assigned in ``on_update``. ``elapsed``, a plain float registered as state, sums
    the ``dt``."""

    _last: torch.Tensor | None = None

    def __init__(self, fill: float, device: str = "cpu") -> None:
        super().__init__(seed=0, device=device)
        self.out = Port(send=TensorSpec((3,)))
        self.linear = nn.Linear(3, 2, device=self.device)
        with torch.no_grad():
            self.linear.weight.fill_(fill)
            self.linear.bias.fill_(fill)
        self.elapsed = 0.0
        self.register_state("elapsed")

    @override
    def on_start(self) -> None:
        self._x = torch.zeros(3, device=self.device)
        self.steps = torch.zeros((), dtype=torch.int64, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        params = list(self.linear.parameters())
        output = self.linear(INPUT.to(self.device))
        loss = (output - TARGET.to(self.device)).pow(2).sum()
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, g in zip(params, grads, strict=True):
                p.sub_(100.0 * dt * g)
        x = self._x.clone()
        self._x.add_(dt * (1 - self._x))
        if self._last is not None:
            self._x.add_(0.5 * (x - self._last))
        self._last = x
        self.steps.add_(1)
        self.elapsed += dt


class TestRestore:
    @pytest.mark.parametrize(
        "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.cuda)]
    )
    def test_a_restored_node_continues_the_same(
        self, step: Step, start_manual: StartManual, device: str
    ):
        k, m = 4, 6
        node = Learner(0.0, device)
        run = start_manual(Group(node=node))
        for _ in range(k):
            step(run)
        tree = pickle.loads(pickle.dumps(run.state_dict()))
        for _ in range(m):
            step(run)
        expected = copy.deepcopy(node.state_dict())
        run.shutdown()
        assert int(at(expected, "steps")) == k + m

        restored_node = Learner(5.0, device)
        weight = restored_node.linear.weight
        timer = VirtualTimer()  # the first run's timer cannot be set back
        timer.advance(100.0)  # far from the tree's time
        restored = start_manual(Group(node=restored_node), timer=timer, state=tree)
        x = restored_node._x  # pyright: ignore[reportPrivateUsage]  # made in on_start
        for _ in range(m):
            step(restored)
        actual = restored_node.state_dict()
        assert torch.equal(at(actual, "_x"), at(expected, "_x"))
        assert torch.equal(at(actual, "_last"), at(expected, "_last"))
        assert int(at(actual, "steps")) == k + m
        assert actual["elapsed"] == expected["elapsed"] > 0
        for name in ("weight", "bias"):
            assert torch.equal(at(actual, "linear", name), at(expected, "linear", name))
        # Restored in place: the objects made in __init__ and on_start remain.
        assert restored_node.linear.weight is weight
        assert restored_node._x is x  # pyright: ignore[reportPrivateUsage]


class OnDevice(Node):
    """Records the thread's CUDA settings in ``on_update``."""

    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.out = Port(send=TensorSpec((3,)))
        self.current_device: int | None = None
        self.own_stream: bool | None = None

    @override
    def on_update(self, dt: float) -> None:
        self.current_device = torch.cuda.current_device()
        default = torch.cuda.default_stream(self.device)
        self.own_stream = torch.cuda.current_stream() != default


class StreamRecorder(Node):
    """Records the current CUDA device and stream in each ``on_update``, and
    switches the stream away (to its device's default stream) at its end without
    restoring it."""

    def __init__(self, device: str) -> None:
        super().__init__(device=device, period=0.01)
        self.out = Port(send=TensorSpec((3,)))
        self.devices: list[int] = []
        self.streams: list[int] = []
        self.updated: queue.Queue[None] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        self.devices.append(torch.cuda.current_device())
        self.streams.append(torch.cuda.current_stream().cuda_stream)
        # Left behind for whatever runs next on this thread.
        torch.cuda.set_stream(torch.cuda.default_stream(self.device))
        self.updated.put(None)


class Draws(Node):
    """Draws from ``generator`` in ``on_start``."""

    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__(device=device, seed=seed)
        self.draws: torch.Tensor | None = None

    @override
    def on_start(self) -> None:
        super().on_start()
        self.draws = torch.rand(3, generator=self.generator, device=self.device)


class Busy(Node):
    """Each ``on_update`` adds 1 to a large tensor 20 times, in place, and first
    records whether the node's stream had finished the previous update."""

    def __init__(self) -> None:
        super().__init__(device="cuda:0")
        self.out = Port(send=TensorSpec((3,)))
        self.idle_at_update: list[bool] = []

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(1 << 26, device=self.device)  # 256 MB of float32

    @override
    def on_update(self, dt: float) -> None:
        self.idle_at_update.append(torch.cuda.current_stream().query())
        for _ in range(20):
            self.w.add_(1.0)


@pytest.mark.cuda
class TestCuda:
    @pytest.mark.parametrize("index", [0, pytest.param(1, marks=pytest.mark.cuda(2))])
    def test_hooks_run_on_the_device_on_a_stream_of_their_own(
        self, step: Step, start_manual: StartManual, index: int
    ):
        node = OnDevice(f"cuda:{index}")
        run = start_manual(Group(node=node))
        step(run)
        assert node.current_device == index
        assert node.own_stream is True

    @pytest.mark.parametrize(
        "indices", [(0, 0), pytest.param((0, 1), marks=pytest.mark.cuda(2))]
    )
    def test_nodes_sharing_a_worker_keep_their_own_device_and_stream(
        self, virtual: VirtualTimer, start: Start, indices: tuple[int, int]
    ):
        a, b = (StreamRecorder(f"cuda:{i}") for i in indices)
        run = start(Group(a=a, b=b), timer=virtual, workers=1)
        for _ in range(3):
            virtual.advance(0.01)
            for node in (a, b):
                node.updated.get(timeout=5.0)
        run.shutdown()
        for node, index in zip((a, b), indices, strict=True):
            assert node.devices == [index] * 3
            assert len(set(node.streams)) == 1
            assert node.streams[0] != torch.cuda.default_stream(index).cuda_stream
        assert a.streams[0] != b.streams[0]

    def test_the_generator_is_on_the_device_and_follows_the_seed(
        self, start_manual: StartManual
    ):
        device = torch.device("cuda", 0)
        a, b = Draws(device, seed=7), Draws(device, seed=7)
        start_manual(Group(a=a, b=b))
        generator = torch.Generator(device).manual_seed(7)
        expected = torch.rand(3, generator=generator, device=device)
        assert a.draws is not None and b.draws is not None
        assert a.draws.device == device
        assert torch.equal(a.draws, expected)
        assert torch.equal(b.draws, expected)

    @pytest.mark.slow
    def test_waits_for_its_stream_after_each_update(
        self, step: Step, start_manual: StartManual
    ):
        node = Busy()
        run = start_manual(Group(node=node))
        for _ in range(5):
            step(run)
        # The first update follows on_start, after which nothing waits.
        assert node.idle_at_update[1:] == [True] * 4
