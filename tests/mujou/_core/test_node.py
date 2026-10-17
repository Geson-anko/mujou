import copy
import math
import pickle
import queue
import threading
import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast, override

import pytest
import torch
from torch import nn

import mujou
from mujou import Network, NetworkError, Node, Port, TensorSpec
from mujou._core import _native
from mujou._core.node import thread_ident
from mujou.testing import ManualRun, VirtualTimer
from mujou.time import ReadOnlyTimer, Timer, Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual


class Probe(Node):
    """Overrides ``on_start`` without calling the base, and records the thread's
    settings there."""

    def __init__(self, seed: int | None = None, *, init_base: bool = True) -> None:
        if init_base:
            super().__init__(seed=seed)
        self.out = Port(send=TensorSpec((3,)))
        self.grad_enabled: bool | None = None
        self.draws: torch.Tensor | None = None
        self.ident: int | None = None

    @override
    def on_start(self) -> None:
        self.grad_enabled = torch.is_grad_enabled()
        self.draws = torch.rand(3, generator=self.generator)
        self.ident = threading.get_ident()


class Solo(Network):
    def __init__(self, node: Node) -> None:
        self.node = node


# The tests run the networks in manual mode only so that idle nodes do not spin:
# on_start runs either way.


class TestThreadSetup:
    def test_on_start_runs_without_gradients_on_its_own_thread(
        self, start_manual: StartManual
    ):
        node = Probe(seed=0)
        start_manual(Solo(node))
        assert node.grad_enabled is False
        assert node.ident not in (None, threading.get_ident())
        # The setting is per thread: the host still records gradients.
        assert torch.is_grad_enabled()

    def test_the_generator_follows_the_seed(self, start_manual: StartManual):
        a, b, c = Probe(seed=7), Probe(seed=7), Probe(seed=8)
        for node in (a, b, c):
            start_manual(Solo(node))
        assert a.draws is not None and b.draws is not None and c.draws is not None
        expected = torch.rand(3, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a.draws, expected)
        assert torch.equal(b.draws, expected)
        assert not torch.equal(c.draws, expected)

    def test_works_without_calling_the_base_init(self, start_manual: StartManual):
        node = Probe(init_base=False)
        start_manual(Solo(node))
        assert node.device == torch.device("cpu")
        assert node.grad_enabled is False
        assert node.draws is not None

    def test_the_thread_is_forgotten_after_the_end(self, start_manual: StartManual):
        node = Probe()
        run = start_manual(Solo(node))
        assert thread_ident(node) == node.ident
        run.shutdown()
        assert thread_ident(node) is None

    def test_leaves_the_hooks_of_the_node_alone(self, start_manual: StartManual):
        node = Probe(seed=0)
        run = start_manual(Solo(node))
        run.state_dict()
        assert not {"on_update", "state_dict"} & set(vars(node))

    def test_the_generator_is_made_on_the_thread(self):
        with pytest.raises(RuntimeError, match="node's thread"):
            _ = Probe(seed=0).generator


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
        run = start_manual(Solo(node), timer=virtual)
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
        run = start_manual(Solo(node), timer=virtual)
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
        run = start_manual(Solo(node), timer=virtual)
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
        run = start_manual(Solo(node), timer=virtual)
        run.shutdown()
        done = threading.Event()

        def read() -> None:
            while not done.is_set():
                run.read(node.out)

        reader = threading.Thread(target=read)
        reader.start()
        try:
            for _ in range(5):
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
        run = start_manual(Solo(node), timer=virtual)
        virtual.advance(0.001)
        with pytest.raises(NetworkError):
            run.step()
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer

    def test_is_released_when_a_run_is_refused(self, start_manual: StartManual):
        tree = start_manual(Solo(Timed())).state_dict()
        node = Timed()
        with pytest.raises(ValueError):
            # Refused at the start: the timer of the run that is still running
            # cannot be set back.
            start_manual(Solo(node), state=tree)
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer


def is_available(kind: str) -> bool:
    """Whether ``kind`` is the type of torch's current accelerator, available."""
    found = torch.accelerator.current_accelerator(check_available=True)
    return found is not None and found.type == kind


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_fixes_the_cuda_index(self):
        assert Node(device="cuda").device.index == torch.cuda.current_device()

    @pytest.mark.parametrize("kind", ["mps", "xpu"])
    def test_rejects_an_accelerator_that_is_not_available(self, kind: str):
        if is_available(kind):
            pytest.skip(f"needs no {kind}")
        with pytest.raises(ValueError, match=f"{kind.upper()} is not available"):
            Node(device=kind)

    def test_fixes_the_index_of_an_accelerator(self, accelerator: torch.device):
        index = torch.accelerator.current_device_index()
        assert Node(device=accelerator.type).device == torch.device(
            accelerator.type, index
        )

    def test_rejects_devices_other_than_the_cpu_and_accelerators(self):
        with pytest.raises(ValueError, match="CPU or an available accelerator"):
            Node(device="meta")


class Ticker(Node):
    """Puts each ``dt`` it updates with into ``dts``."""

    def __init__(self, period: float | None) -> None:
        super().__init__(period=period)
        self.out = Port(send=TensorSpec((1,)))
        self.dts: queue.Queue[float] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        self.dts.put(dt)


class TestPeriod:
    def test_defaults_to_none(self):
        assert Probe().period is None
        assert Probe(init_base=False).period is None

    def test_is_kept(self):
        assert Node(period=0.01).period == 0.01

    @pytest.mark.parametrize("period", [0.0, -0.01, math.inf, math.nan])
    def test_rejects_what_is_not_finite_and_positive(self, period: float):
        with pytest.raises(ValueError, match="period"):
            Node(period=period)

    def test_rejects_a_bool(self):
        with pytest.raises(TypeError, match="period"):
            Node(period=True)

    def test_is_a_reserved_name(self, virtual: VirtualTimer, start: Start):
        class OnTheClass(Ticker):
            period = 0.5  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]

        with pytest.raises(TypeError, match="reserved"):
            start(Solo(OnTheClass(period=None)), timer=virtual)

    def test_the_node_waits_for_its_period_on_the_timer(
        self, virtual: VirtualTimer, start: Start
    ):
        # Free running: a manual run (mujou.testing.run) ignores the period.
        node = Ticker(period=0.01)
        start(Solo(node), timer=virtual)
        virtual.advance(0.005)
        with pytest.raises(queue.Empty):
            node.dts.get(timeout=0.05)
        virtual.advance(0.005)
        assert node.dts.get(timeout=5.0) == 0.01

    def test_a_period_below_a_nanosecond_is_one_nanosecond(
        self, virtual: VirtualTimer, start: Start
    ):
        node = Ticker(period=1e-10)
        start(Solo(node), timer=virtual)
        virtual.advance(1e-9)
        assert node.dts.get(timeout=5.0) == 1e-9


class OwnSetup(Probe):
    """Overrides ``on_thread_start`` without calling the base, and tries a host
    method there."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed=seed)
        self.host_error: str | None = None

    @override
    def on_thread_start(self) -> None:
        self.grad_at_thread_start = torch.is_grad_enabled()
        self.draws_at_thread_start = torch.rand(3, generator=self.generator)
        self.ident_at_thread_start = thread_ident(self)
        try:
            mujou.run(Solo(Probe()))
        except RuntimeError as e:
            self.host_error = str(e)


class TestOnThreadStart:
    def test_runs_after_the_settings_without_calling_the_base(
        self, start_manual: StartManual
    ):
        node = OwnSetup(seed=7)
        start_manual(Solo(node))
        assert node.grad_at_thread_start is False
        assert node.grad_enabled is False
        generator = torch.Generator().manual_seed(7)
        expected = torch.rand(3, generator=generator)
        assert torch.equal(node.draws_at_thread_start, expected)
        # on_start draws next from the same generator.
        assert node.draws is not None
        assert torch.equal(node.draws, torch.rand(3, generator=generator))
        assert node.ident_at_thread_start == node.ident == thread_ident(node)
        # The guard against the host methods holds in it too.
        assert node.host_error is not None and "host methods" in node.host_error


class Stateful(Node):
    """Holds a public and a private tensor, a parameter that records gradients, a
    module and values that are not state."""

    def __init__(self, fill: float) -> None:
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
        with torch.enable_grad():  # mujou: local-autograd
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


INPUT = torch.tensor([1.0, -1.0, 0.5])
TARGET = torch.tensor([0.5, 2.0])


class Learner(Node):
    """Fits ``linear`` to ``TARGET`` by backpropagation inside the node, and relaxes
    ``_x`` toward 1 with momentum, both in proportion to ``dt``. ``_last`` is first
    assigned in ``on_update``."""

    _last: torch.Tensor | None = None

    def __init__(self, fill: float, device: str = "cpu") -> None:
        super().__init__(seed=0, device=device)
        self.out = Port(send=TensorSpec((3,)))
        self.linear = nn.Linear(3, 2, device=self.device)
        with torch.no_grad():
            self.linear.weight.fill_(fill)
            self.linear.bias.fill_(fill)

    @override
    def on_start(self) -> None:
        self._x = torch.zeros(3, device=self.device)
        self.steps = torch.zeros((), dtype=torch.int64, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        params = list(self.linear.parameters())
        with torch.enable_grad():  # mujou: local-autograd
            output = self.linear(INPUT.to(self.device))
            loss = (output - TARGET.to(self.device)).pow(2).sum()
            grads = torch.autograd.grad(loss, params)
        for p, g in zip(params, grads, strict=True):
            p.sub_(100.0 * dt * g)
        x = self._x.clone()
        self._x.add_(dt * (1 - self._x))
        if self._last is not None:
            self._x.add_(0.5 * (x - self._last))
        self._last = x
        self.steps.add_(1)


def steps(run: ManualRun, timer: VirtualTimer, n: int) -> None:
    for _ in range(n):
        timer.advance(0.001)
        run.step()


class TestRestore:
    @pytest.mark.parametrize("kind", ["cpu", "accelerator"])
    def test_a_restored_node_continues_the_same(
        self,
        virtual: VirtualTimer,
        start_manual: StartManual,
        request: pytest.FixtureRequest,
        kind: str,
    ):
        device = "cpu" if kind == "cpu" else str(request.getfixturevalue(kind))
        k, m = 4, 6
        node = Learner(0.0, device)
        run = start_manual(Solo(node))
        steps(run, virtual, k)
        tree = pickle.loads(pickle.dumps(run.state_dict()))
        steps(run, virtual, m)
        expected = copy.deepcopy(node.state_dict())
        run.shutdown()
        assert int(at(expected, "steps")) == k + m

        restored_node = Learner(5.0, device)
        weight = restored_node.linear.weight
        timer = VirtualTimer()  # the first run's timer cannot be set back
        timer.advance(100.0)  # far from the tree's time
        restored = start_manual(Solo(restored_node), timer=timer, state=tree)
        x = restored_node._x  # pyright: ignore[reportPrivateUsage]  # made in on_start
        steps(restored, timer, m)
        actual = restored_node.state_dict()
        assert torch.equal(at(actual, "_x"), at(expected, "_x"))
        assert torch.equal(at(actual, "_last"), at(expected, "_last"))
        assert int(at(actual, "steps")) == k + m
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
        self.grad_enabled: bool | None = None

    @override
    def on_update(self, dt: float) -> None:
        self.current_device = torch.cuda.current_device()
        default = torch.cuda.default_stream(self.device)
        self.own_stream = torch.cuda.current_stream() != default
        self.grad_enabled = torch.is_grad_enabled()


class OnAccelerator(Node):
    """Records the thread's accelerator settings in ``on_start``, and draws from
    ``generator`` there."""

    def __init__(self, device: torch.device, seed: int) -> None:
        super().__init__(device=device, seed=seed)
        self.out = Port(send=TensorSpec((3,)))
        self.current_device: int | None = None
        self.stream: torch.Stream | None = None
        self.grad_enabled: bool | None = None
        self.draws: torch.Tensor | None = None

    @override
    def on_start(self) -> None:
        self.current_device = torch.accelerator.current_device_index()
        self.stream = torch.accelerator.current_stream(self.device)
        self.grad_enabled = torch.is_grad_enabled()
        self.draws = torch.rand(3, generator=self.generator, device=self.device)


class TestAccelerator:
    def test_hooks_run_on_the_device(
        self, start_manual: StartManual, accelerator: torch.device
    ):
        node = OnAccelerator(accelerator, seed=0)
        start_manual(Solo(node))
        assert node.current_device == accelerator.index
        assert node.grad_enabled is False
        # Only CUDA gives a node a stream of its own.
        host_stream = torch.accelerator.current_stream(accelerator)
        assert (node.stream != host_stream) == (accelerator.type == "cuda")

    def test_the_generator_is_on_the_device_and_follows_the_seed(
        self, start_manual: StartManual, accelerator: torch.device
    ):
        a, b = OnAccelerator(accelerator, seed=7), OnAccelerator(accelerator, seed=7)
        for node in (a, b):
            # One at a time: MPS takes one node at most.
            start_manual(Solo(node)).shutdown()
        generator = torch.Generator(accelerator).manual_seed(7)
        expected = torch.rand(3, generator=generator, device=accelerator)
        assert a.draws is not None and b.draws is not None
        assert a.draws.device == accelerator
        assert torch.equal(a.draws, expected)
        assert torch.equal(b.draws, expected)


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


def needs_gpus(n: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < n,
        reason=f"needs {n} CUDA device(s)",
    )


@needs_gpus(1)
class TestCuda:
    @pytest.mark.parametrize("index", [0, pytest.param(1, marks=needs_gpus(2))])
    def test_hooks_run_on_the_device_on_a_stream_of_their_own(
        self, virtual: VirtualTimer, start_manual: StartManual, index: int
    ):
        node = OnDevice(f"cuda:{index}")
        run = start_manual(Solo(node))
        virtual.advance(0.001)
        run.step()
        assert node.current_device == index
        assert node.own_stream is True
        assert node.grad_enabled is False

    @pytest.mark.slow
    def test_waits_for_its_stream_after_each_update(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Busy()
        run = start_manual(Solo(node))
        for _ in range(5):
            virtual.advance(0.001)
            run.step()
        # The first update follows on_start, after which nothing waits.
        assert node.idle_at_update[1:] == [True] * 4
