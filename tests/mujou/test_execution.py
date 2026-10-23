import gc
import math
import os
import queue
import re
import subprocess
import sys
import sysconfig
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import pytest
import torch

import mujou
import mujou.testing
from mujou import NetworkError, Run
from mujou.network import Group, Network
from mujou.node import Node
from mujou.port import Port
from mujou.spec import TensorSpec
from mujou.testing import ManualRun
from mujou.time import ReadOnlyTimer, Timer, Timestamp, VirtualTimer, global_timer
from tests.mujou.nodes import (
    F64,
    Calls,
    Chain,
    Leaky,
    Pair,
    assert_same_tree,
    entries,
    port_slots,
    updates,
    vec,
)

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual

H_NS = 1_000_000
H = H_NS / 1e9  # seconds per step


def u(k: float) -> torch.Tensor:
    """The k-th input: different at every step."""
    return torch.tensor([math.sin(k), math.cos(k)], dtype=F64)


class Echo(Node):
    """Sends the time of the message it receives, in nanoseconds."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = Port(recv=vec())
        self.next = Port(send=TensorSpec((), torch.int64))

    @override
    def on_update(self, dt: float) -> None:
        _, t = self.prev.recv_with_time()
        assert t is not None  # the host writes before each step
        self.next.send(torch.tensor(t.ns))


def euler(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The reference of one ``Leaky`` update with ``tau=1`` and ``dt=H``."""
    return x + min(H / 1.0, 1.0) * (target - x)


def group(nodes: Sequence[Node]) -> Group:
    """``nodes`` side by side, at ``"n0"``, ``"n1"``, ...."""
    return Group(**{f"n{k}": node for k, node in enumerate(nodes)})


class TestRun:
    def test_returns_once_every_node_has_started(self, start: Start):
        net = Chain()
        run = start(net)
        assert isinstance(run, Run)
        assert net.a.started and net.b.started  # run returns after on_start

    @pytest.mark.parametrize(
        ("make", "again", "message"),
        [
            pytest.param(Chain, lambda c: c, "a", id="network"),
            pytest.param(Leaky, lambda c: c, "Leaky", id="node"),
            pytest.param(
                Leaky, lambda c: Group(node=c), "node", id="node-in-a-network"
            ),
            pytest.param(
                Chain,
                lambda c: Group(x=c, y=Chain()),
                "x.a",
                id="network-among-several",
            ),
        ],
    )
    def test_runs_once(
        self,
        start: Start,
        make: Callable[[], Node | Network],
        again: Callable[[Node | Network], Node | Network],
        message: str,
    ):
        component = make()
        start(component).shutdown()
        with pytest.raises(RuntimeError) as raised:
            start(again(component))
        assert str(raised.value) == (
            f"{message} has already run; create a new instance (its attributes "
            "stay readable)"
        )

    def test_raises_what_on_start_raised(self, start: Start):
        net = Chain(raise_in_b=0)
        with pytest.raises(NetworkError) as raised:
            start(net)
        assert raised.value.__cause__ is net.b.error
        assert net.a.updates == 0 and net.b.updates == 0

    def test_runs_a_dataclass_network(self, start_manual: StartManual):
        @dataclass
        class Pair(Network):
            a: Leaky = field(default_factory=Leaky)
            b: Leaky = field(default_factory=Leaky)

            def __post_init__(self) -> None:
                self.a.next.connect(self.b.prev)

        net = Pair()
        run = start_manual(net, inputs={net.a.prev: u(1)})
        run.step(H, net.a, net.b)
        assert net.b.updates == 1 and torch.equal(run.read(net.b.next), net.b.x)
        assert list(run.state_dict()["nodes"]) == ["a", "b"]

    def test_a_network_without_nodes_runs_nothing(self, start_manual: StartManual):
        with start_manual(Network()) as run:
            assert run.state_dict()["nodes"] == {}


class TestRunANode:
    def test_runs_as_the_root(self, start_manual: StartManual):
        node = Leaky()
        run = start_manual(node, inputs={node.prev: u(1)})
        run.step(H, node)
        y = run.read(node.next)
        assert torch.equal(y, euler(torch.zeros(2, dtype=F64), u(1)))
        tree = run.state_dict()
        assert list(tree["nodes"]) == [""]
        assert tree["nodes"][""]["ports"].keys() == {"prev", "next"}

    def test_names_the_root_node_by_its_class_in_a_structure_error(self, start: Start):
        class Outer(Node):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Leaky()

        with pytest.raises(TypeError, match=r"^Outer\.inner is a Leaky"):
            start(Outer())

    def test_names_the_root_node_by_its_class_when_it_raises(self, start: Start):
        node = Leaky(raise_at=0)
        with pytest.raises(NetworkError) as raised:
            start(node)
        assert "node 'Leaky' raised in on_start" in str(raised.value)

    def test_names_its_ports_by_its_class_in_errors(self, start_manual: StartManual):
        # Their paths ("x") alone would not say which node they are of.
        node = Pair()
        run = start_manual(node)
        dtype = r"^slot 'Pair\.x/in': expected dtype float64, got float32$"
        with pytest.raises(ValueError, match=dtype):
            run.write({node.x: torch.zeros(2, dtype=torch.float32)})
        with pytest.raises(TypeError, match=r"^Pair\.x: expected a torch\.Tensor"):
            run.write({node.x: [0.0, 0.0]})  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match=r"^the host does not write Pair\.out"):
            run.write({node.out: u(0)})
        with pytest.raises(ValueError, match=r"^the host does not read Pair\.x"):
            run.read(node.x)

    def test_a_node_in_a_network_names_its_ports_by_their_paths(
        self, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError, match=r"^the host does not write b\.prev"):
            run.write({net.b.prev: u(0)})

    def test_mixes_with_networks(self, start_manual: StartManual):
        net, node = Chain(), Leaky()
        net.next.connect(node.prev)
        run = start_manual(Group(net=net, node=node), inputs={net.prev: u(1)})
        assert list(run.state_dict()["nodes"]) == ["net.a", "net.b", "node"]
        a = b = n = torch.zeros(2, dtype=F64)
        for _ in range(3):
            run.step(H, net.a, net.b, node)
            # In the order of the steps, each reads the value just sent before it.
            a = euler(a, u(1))
            b = euler(b, a)
            n = euler(n, b)
        y = run.read(node.next)
        assert torch.equal(y, n)


class TestRefusedRun:
    @pytest.mark.parametrize(
        ("given", "match"),
        [
            ("inputs", r"^slot 'a\.prev/in': expected shape \(2,\)"),
            ("inputs-connected", r"^the host does not write b\.prev"),
            ("initial", r"^slot 'bad\.next/out': expected shape \(2,\)"),
            ("workers", r"^workers must be at least 1"),
            ("state", r"^the state tree does not match this run"),
            ("timer-and-state", r"^a restored run runs on a timer of its own"),
        ],
    )
    def test_changes_nothing_and_the_same_instances_run(
        self, start_manual: StartManual, start: Start, given: str, match: str
    ):
        net = Chain()
        wrong = torch.zeros(3, dtype=F64)
        source = start_manual(Chain())
        tree = source.state_dict()
        source.shutdown()
        broken = {**tree, "nodes": {"a": tree["nodes"]["a"]}}
        refused: Callable[[], object] = {
            "inputs": partial(start_manual, net, inputs={net.prev: wrong}),
            "inputs-connected": partial(start_manual, net, inputs={net.b.prev: u(1)}),
            "initial": partial(start_manual, Group(net=net, bad=Initial(wrong))),
            "workers": partial(start, net, timer=VirtualTimer(), workers=0),
            "state": partial(start_manual, net, state=broken),
            "timer-and-state": partial(
                start_manual, net, timer=VirtualTimer(), state=tree
            ),
        }[given]
        with pytest.raises(ValueError, match=match):
            refused()
        assert not (net.a.started or net.b.started)
        run = start_manual(net, state=tree if "state" in given else None)
        run.step(H)
        assert net.a.updates > 0

    def test_a_node_that_raised_while_starting_has_run(self, start: Start):
        net = Chain(raise_in_b=0)
        with pytest.raises(NetworkError):
            start(net)
        net.b.raise_at = None
        with pytest.raises(RuntimeError, match="^a has already run"):
            start(net)


class Initial(Node):
    """Sends ``initial`` until its first update."""

    def __init__(self, initial: torch.Tensor) -> None:
        super().__init__()
        self.next = Port(send=vec(), initial=initial)


class Noisy(Node):
    """``dx/dt = u + xi - x`` by explicit Euler, ``xi ~ U[0, 1)^2`` drawn from the
    node's own generator at each update.

    ``u`` comes from the previous side and ``x`` goes to the next side. Records
    the OS thread of each update and signals it.
    """

    def __init__(self, seed: int, *, dedicated_thread: bool = False) -> None:
        super().__init__(seed=seed, dedicated_thread=dedicated_thread)
        self.prev = Port(recv=vec())
        self.next = Port(send=vec())
        self.x = torch.zeros(2, dtype=F64)
        self.idents: list[int] = []
        self.updated: queue.Queue[None] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        self.idents.append(threading.get_ident())
        xi = torch.rand(2, dtype=F64, generator=self.generator)
        self.x += dt * (self.prev.recv() + xi - self.x)
        self.next.send(self.x)
        self.updated.put(None)


def noisy_chain(n: int, *, dedicated_thread: bool = False) -> list[Noisy]:
    """``n`` ``Noisy`` nodes seeded ``0..n-1``, each sending to the next."""
    nodes = [Noisy(seed=k, dedicated_thread=dedicated_thread) for k in range(n)]
    for sender, receiver in zip(nodes, nodes[1:]):
        sender.next.connect(receiver.prev)
    return nodes


def noisy_reference(n: int, steps: int, dt: float, chained: bool) -> list[torch.Tensor]:
    """The states of ``noisy_chain(n)`` (or of ``n`` unconnected ``Noisy``) after
    ``steps`` updates of every node in the order of the chain, each by ``dt``."""
    generators = [torch.Generator().manual_seed(k) for k in range(n)]
    xs = [torch.zeros(2, dtype=F64) for _ in range(n)]
    for _ in range(steps):
        for k in range(n):
            inp = xs[k - 1] if chained and k > 0 else torch.zeros(2, dtype=F64)
            xi = torch.rand(2, dtype=F64, generator=generators[k])
            xs[k] = xs[k] + dt * (inp + xi - xs[k])
    return xs


def more_nodes_than_cpus() -> int:
    """A node count above the default number of workers, the logical CPUs."""
    return (os.cpu_count() or 1) + 2


def workers_of(run: Run) -> int:
    """The number of workers of the pool of ``run``, as its runtime counts them."""
    return run._runtime.workers  # pyright: ignore[reportPrivateUsage]


class Holding(Node):
    """Holds its worker in each update (``time.sleep`` lets the GIL go) and
    records the OS thread."""

    def __init__(self) -> None:
        super().__init__(period=0.0)
        self.idents: set[int] = set()

    @override
    def on_update(self, dt: float) -> None:
        self.idents.add(threading.get_ident())
        time.sleep(0.005)


class TestWorkers:
    @pytest.mark.parametrize("workers", [0, -1])
    def test_rejects_less_than_one_before_starting(
        self, virtual: VirtualTimer, start: Start, workers: int
    ):
        net = Chain()
        with pytest.raises(ValueError, match="^workers must be at least 1"):
            start(net, timer=virtual, workers=workers)
        assert not (net.a.started or net.b.started)
        start(net, timer=virtual, workers=1)
        assert net.a.started and net.b.started

    def test_more_nodes_than_workers_all_update_exactly(
        self, virtual: VirtualTimer, start: Start
    ):
        # Unconnected, so the order of the updates within a period does not
        # matter: each node's state is a function of its own seed and dt only.
        n, steps, period = 8, 3, 0.01
        nodes = [Noisy(seed=k) for k in range(n)]
        start(group(nodes), timer=virtual, workers=2)
        for _ in range(steps):
            virtual.advance(period)
            for node in nodes:
                node.updated.get(timeout=5.0)
        # dt is 10_000_000 ns / 1e9, the same float as the period.
        expected = noisy_reference(n, steps, period, chained=False)
        for node, x in zip(nodes, expected, strict=True):
            assert len(node.idents) == steps
            assert torch.equal(node.x, x)
        assert len({i for node in nodes for i in node.idents}) <= 2

    @pytest.mark.parametrize("dedicated_thread", [False, True])
    def test_a_manual_run_gives_the_same_bits_on_workers_and_dedicated_threads(
        self, start_manual: StartManual, dedicated_thread: bool
    ):
        n, steps = more_nodes_than_cpus(), 4
        nodes = noisy_chain(n, dedicated_thread=dedicated_thread)
        run = start_manual(group(nodes))
        for _ in range(steps):
            run.step(H, *nodes)  # each node reads what the previous one just sent
        expected = noisy_reference(n, steps, H, chained=True)
        for node, x in zip(nodes, expected, strict=True):
            assert torch.equal(node.x, x)

    def test_the_default_shares_threads_among_more_nodes_than_cpus(
        self, start_manual: StartManual
    ):
        n = more_nodes_than_cpus()
        nodes = noisy_chain(n)
        run = start_manual(group(nodes))
        for _ in range(2):
            run.step(H, *nodes)
        idents = {i for node in nodes for i in node.idents}
        assert len(idents) <= workers_of(run) < n
        assert threading.get_ident() not in idents

    def test_the_default_is_at_most_the_cpus_or_half_of_them_with_the_gil(
        self, start_manual: StartManual
    ):
        # The runtime also counts a CPU quota, which Python cannot see, so only
        # the bound is known here.
        run = start_manual(group(noisy_chain(more_nodes_than_cpus())))
        cpus = os.process_cpu_count() or 1
        gil = sys._is_gil_enabled()  # pyright: ignore[reportPrivateUsage]
        assert 1 <= workers_of(run) <= (max(1, cpus // 2) if gil else cpus)

    def test_the_default_number_of_workers_all_take_updates(self, start: Start):
        # Every update holds its worker and some node is always due, so each
        # worker takes updates: one thread per worker, as many as the default.
        nodes = [Holding() for _ in range(more_nodes_than_cpus())]
        run = start(group(nodes), timer=Timer())
        time.sleep(0.5)
        run.shutdown()
        idents = set[int]().union(*(node.idents for node in nodes))
        assert len(idents) == workers_of(run)


class TestWith:
    def test_shuts_down_at_the_end(self):
        net = Chain()
        with mujou.run(net) as run:
            assert net.a.started
        with pytest.raises(RuntimeError, match="stopped"):
            run.write({net.prev: u(0)})

    def test_raises_what_a_node_raised(self, virtual: VirtualTimer):
        net = Chain(raise_in_b=1)
        run = mujou.testing.run(net, timer=virtual)
        with pytest.raises(NetworkError) as raised, run:
            with pytest.raises(NetworkError):
                run.step(H)
        # From the shutdown at the end of the with statement.
        assert raised.value.__cause__ is net.b.error

    def test_lets_the_exception_of_the_body_through(self, virtual: VirtualTimer):
        net = Chain(raise_in_b=1)
        run = mujou.testing.run(net, timer=virtual)
        with pytest.raises(KeyError), run:
            with pytest.raises(NetworkError):
                run.step(H)
            raise KeyError("the body's own")
        with pytest.raises(NetworkError):
            run.join()


class TestTimer:
    def test_the_slots_and_the_nodes_run_on_the_timer_of_the_run(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        run = start_manual(Group(node=node), timer=virtual)
        virtual.advance(2.5)
        run.write({node.x: u(1)})
        run.step(H)
        ((_, written_at), _) = node.seen[0]
        assert written_at == Timestamp(ns=2_500_000_000)
        tree = run.state_dict()
        assert tree["timer"]["time_ns"] == virtual.now().ns
        assert tree["nodes"]["node"]["last_update_ns"] == virtual.now().ns

    def test_a_pause_stops_only_the_timer_of_its_run(self, start: Start):
        held, free = Timer(), Timer()

        class Measures(Node):
            @override
            def state_dict(self) -> dict[str, Any]:
                # Called while the run on `held` is paused.
                h0, f0, g0 = held.now(), free.now(), global_timer.now()
                time.sleep(0.05)
                return {
                    "moved": (held.now() - h0, free.now() - f0, global_timer.now() - g0)
                }

        start(Chain(tau=0.01), timer=free)
        run = start(Group(node=Measures()), timer=held)
        on_held, on_free, on_global = run.state_dict()["nodes"]["node"]["state"][
            "moved"
        ]
        assert on_held == 0.0
        assert on_free > 0.0 and on_global > 0.0

    def test_gives_the_timer_it_runs_on(self, start: Start):
        timer = Timer()
        assert start(Chain(), timer=timer).timer is timer
        assert start(Chain()).timer is global_timer

    def test_a_manual_run_gives_its_virtual_timer(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        assert start_manual(Chain(), timer=virtual).timer is virtual

    def test_the_global_timer_by_default(self, start: Start):
        class ReadsTimer(Node):
            seen: ReadOnlyTimer | None = None

            @override
            def on_start(self) -> None:
                self.seen = self.timer

        node = ReadsTimer()
        start(Group(node=node))
        seen = node.seen
        assert seen is not None
        # On one time axis with the global timer.
        first = seen.now()
        now_ns = global_timer.now().ns
        assert first.ns <= now_ns <= seen.now().ns

    def test_the_tree_after_a_normal_end_is_not_earlier_than_anything_of_the_run(
        self, start: Start
    ):
        timer = Timer()
        net = Chain(tau=0.01)
        run = start(net, timer=timer, inputs={net.prev: u(1)})
        run.wait_for(net.next, lambda y: bool(y.abs().sum() > 0), timeout=10.0)
        run.shutdown()
        tree = run.state_dict()
        end_ns = tree["timer"]["time_ns"]
        times = [entry["time_ns"] for entry in entries(tree)]
        assert max([*times, *updates(tree).values()]) <= end_ns
        # The time of the run stays at its end while the timer goes on.
        time.sleep(0.01)
        assert timer.now().ns > end_ns == run.state_dict()["timer"]["time_ns"]

    def test_a_restored_run_goes_on_from_the_tree_on_a_timer_of_its_own(
        self, start: Start
    ):
        net = Chain(tau=0.01)
        run = start(net, inputs={net.prev: u(1)})
        global_timer.sleep(0.02)
        run.shutdown()
        tree = run.state_dict()
        before = global_timer.now()
        real_before = time.perf_counter_ns()
        restored = start(Chain(tau=0.01), state=tree)
        now_ns = restored.timer.now().ns
        real_ns = time.perf_counter_ns() - real_before
        assert restored.timer is not global_timer
        # From the tree's time on, no faster than the real time (scale 1).
        tree_ns = tree["timer"]["time_ns"]
        assert tree_ns <= now_ns <= tree_ns + real_ns
        assert restored.timer.time_scale() == tree["timer"]["time_scale"]
        # The global timer did not move back to the tree's time.
        assert global_timer.now() >= before

    def test_the_timer_of_a_restored_run_runs_once_the_run_has_started(
        self, start: Start
    ):
        class SlowLoad(Leaky):
            """Takes 0.2 s to load its state, as a large one would."""

            updated_at: Timestamp | None = None

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                if self.updated_at is None:
                    self.updated_at = self.timer.now()

            @override
            def load_state_dict(self, state: object) -> None:
                time.sleep(0.2)
                super().load_state_dict(state)

        first = SlowLoad(tau=0.01)
        run = start(first, timer=Timer(), inputs={first.prev: u(1)})
        run.wait_for(first.next, lambda y: bool(y.abs().sum() > 0), timeout=10.0)
        run.shutdown()
        tree = run.state_dict()
        tree_ns = tree["timer"]["time_ns"]
        node = SlowLoad(tau=0.01)
        global_before = global_timer.now()
        restored = start(node, state=tree, inputs={node.prev: u(2)})
        # Only the run's own timer stood still: the global one ran meanwhile.
        assert global_timer.now() - global_before >= 0.2
        restored.wait_for(
            node.next, lambda y: node.updated_at is not None, timeout=10.0
        )
        # The 0.2 s of the restore are not in the run's time: the inputs are
        # written, and the first update comes, at the tree's time (but for the
        # time the run takes to lay out its slots, and a period). The bound
        # stays well under the 0.2 s, with room for a slow CI machine.
        assert node.updated_at is not None
        written_ns = port_slots(restored.state_dict(), "", "prev")["in"]["time_ns"]
        for ns in (written_ns, node.updated_at.ns):
            assert 0 <= ns - tree_ns < 150_000_000

    def test_a_restored_manual_run_starts_its_virtual_timer_at_the_tree(
        self, start_manual: StartManual
    ):
        run = start_manual(Chain())
        run.step(0.25)
        tree = run.state_dict()
        restored = start_manual(Chain(), state=tree)
        assert restored.timer.now() == Timestamp(ns=250_000_000)


class Ticks(Node):
    """Counts its updates, in a tensor of its state, sends the count, and records
    each ``dt``."""

    def __init__(self) -> None:
        super().__init__(period=0.001)
        self.next = Port(send=TensorSpec((), torch.int64))
        self.dts: list[float] = []

    @override
    def on_start(self) -> None:
        super().on_start()
        self.count = torch.zeros((), dtype=torch.int64)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.dts.append(dt)
        self.count += 1
        self.next.send(self.count)


def wait_for_updates(run: mujou.Run, node: Ticks, more_than: int) -> None:
    run.wait_for(node.next, lambda count: count.item() > more_than, timeout=10.0)


class TestPausedTimer:
    """The host stops a timer to do something at one time T, such as taking the
    states of every run on it."""

    def test_the_states_of_two_runs_on_one_timer_are_of_one_time(self):
        timer = Timer()  # the real clock: one state after the other would differ
        a, b = Ticks(), Ticks()
        with mujou.run(a, timer=timer) as run_a, mujou.run(b, timer=timer) as run_b:
            wait_for_updates(run_a, a, 2)
            wait_for_updates(run_b, b, 2)
            timer.pause()
            try:
                paused_at = timer.now()
                trees = [run_a.state_dict(), run_b.state_dict()]
                assert timer.now() == paused_at
                assert timer.is_paused()  # the runs' own holds left the switch on
            finally:
                timer.resume()
            assert not timer.is_paused()
        for tree in trees:
            assert tree["timer"]["time_ns"] == paused_at.ns
            assert tree["nodes"][""]["last_update_ns"] <= paused_at.ns
            assert tree["nodes"][""]["state"]["count"] > 2

    def test_nothing_moves_while_paused(self):
        timer = Timer()
        node = Ticks()
        with mujou.run(node, timer=timer) as run:
            wait_for_updates(run, node, 2)
            timer.pause()
            try:
                before = run.state_dict()
                time.sleep(0.05)
                after = run.state_dict()
            finally:
                timer.resume()
        assert_same_tree(after, before)

    def test_the_first_dt_after_resume_leaves_out_the_pause(self):
        timer = Timer()
        node = Ticks()
        real_start, start = time.monotonic(), timer.now()
        with mujou.run(node, timer=timer) as run:
            wait_for_updates(run, node, 2)
            timer.pause()
            time.sleep(0.05)
            timer.resume()
            wait_for_updates(run, node, int(run.read(node.next)) + 1)
        end, real_end = timer.now(), time.monotonic()
        # Every dt is on the timer, which stood still for the 0.05 real seconds
        # of the pause: a dt that counted the pause would sum past the timer.
        assert sum(node.dts) <= end - start
        assert end - start <= real_end - real_start - 0.05 + 1e-6

    def test_a_virtual_timer_cannot_be_stepped_while_paused(self):
        timer = VirtualTimer()
        node = Ticks()
        with mujou.testing.run(node, timer=timer) as run:
            run.step(H)
            timer.pause()
            with pytest.raises(RuntimeError, match="paused"):
                run.step(H)
            timer.resume()
            run.step(H)
        assert node.dts == [H, H]

    def test_runs_go_on_from_states_taken_while_paused(self):
        timer = Timer()
        a, b = Ticks(), Ticks()
        with mujou.run(a, timer=timer) as run_a, mujou.run(b, timer=timer) as run_b:
            wait_for_updates(run_a, a, 2)
            wait_for_updates(run_b, b, 2)
            timer.pause()
            try:
                trees = [run_a.state_dict(), run_b.state_dict()]
            finally:
                timer.resume()
        paused_at_ns = trees[0]["timer"]["time_ns"]
        for tree in trees:
            # Each on a timer of its own, both from the same time.
            restored = Ticks()
            with mujou.run(restored, state=tree) as run:
                assert run.timer.now().ns >= paused_at_ns
                saved = int(tree["nodes"][""]["state"]["count"])
                wait_for_updates(run, restored, saved)
            assert run.state_dict()["nodes"][""]["last_update_ns"] > paused_at_ns


class TestOnlyTheHostControlsTheTime:
    """A node reads the time and changes nothing: a node changing the scale, the
    virtual clock or waiting on the time would reach the other nodes past the
    ports."""

    @pytest.mark.parametrize("scaled", ["timer", "global"])
    def test_a_node_cannot_scale_the_time(
        self, virtual: VirtualTimer, start_manual: StartManual, scaled: str
    ):
        target = virtual if scaled == "timer" else global_timer
        node = Calls(lambda: target.set_time_scale(1000.0))
        run = start_manual(Group(node=node, other=Calls()), timer=virtual)
        try:
            run.step(H)
            assert isinstance(node.errors[0], RuntimeError)
            assert "only the host" in str(node.errors[0])
            assert virtual.time_scale() == 1.0
            assert global_timer.time_scale() == 1.0
        finally:
            # Should the check regress, leave no scale to the later tests.
            global_timer.set_time_scale(1.0)

    def test_a_node_cannot_advance_the_virtual_clock(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        first = Calls(lambda: virtual.advance(5.0))
        second = Calls()
        run = start_manual(Group(first=first, second=second), timer=virtual)
        run.step(H, first, second)
        assert isinstance(first.errors[0], RuntimeError)
        # Every dt is the time the test stepped.
        assert second.dts == [H]
        assert virtual.now() == Timestamp(ns=1_000_000)

    @pytest.mark.parametrize("sleep", ["timer", "global"])
    def test_a_node_cannot_wait_on_the_time(
        self, virtual: VirtualTimer, start_manual: StartManual, sleep: str
    ):
        wait = virtual.sleep if sleep == "timer" else global_timer.sleep
        node = Calls(lambda: wait(0.001))
        run = start_manual(Group(node=node, other=Calls()), timer=virtual)
        # Nothing in the step waits for the clock. Should it regress, the step
        # would wait for an advance: fail instead of hanging.
        stepper = threading.Thread(target=run.step, args=(H,))
        stepper.start()
        stepper.join(5.0)
        if stepper.is_alive():
            virtual.advance(1.0)  # release the sleeping node
            stepper.join()
            pytest.fail("a node's sleep held the step")
        assert isinstance(node.errors[0], RuntimeError)
        run.state_dict()  # the pause does not wait for a sleeping node

    @pytest.mark.parametrize("target", ["timer", "global"])
    @pytest.mark.parametrize("switch", ["pause", "resume"])
    def test_a_node_cannot_pause_or_resume_the_time(
        self, virtual: VirtualTimer, start_manual: StartManual, target: str, switch: str
    ):
        owner = virtual if target == "timer" else global_timer
        node = Calls(getattr(owner, switch))
        run = start_manual(Group(node=node, other=Calls()), timer=virtual)
        try:
            run.step(H)
            assert isinstance(node.errors[0], RuntimeError)
            assert "only the host" in str(node.errors[0])
            assert not virtual.is_paused()
            assert not global_timer.is_paused()
        finally:
            # Should the check regress, leave no pause to the later tests.
            virtual.resume()
            global_timer.resume()

    def test_a_node_reads_its_timer_only_in_its_hooks(self, start_manual: StartManual):
        seen: list[Timestamp] = []
        node = Calls(lambda: seen.append(node.timer.now()))
        run = start_manual(node)
        run.step(H)
        assert node.errors == [] and seen == [Timestamp(ns=H_NS)]
        with pytest.raises(
            RuntimeError, match=r"^Calls\.timer is used only in the hooks"
        ):
            _ = node.timer


class TestInputs:
    def test_are_the_initial_values(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net, inputs={net.prev: u(1)})
        run.step(H)
        assert torch.equal(net.a.x, euler(torch.zeros(2, dtype=F64), u(1)))

    def test_after_a_state_take_precedence_and_carry_its_time(
        self, start_manual: StartManual
    ):
        first = Pair()
        run = start_manual(Group(node=first), inputs={first.x: u(1)})
        run.step(H)
        tree = run.state_dict()
        written_at = port_slots(tree, "node", "x")["in"]["time_ns"]
        frozen_at = tree["timer"]["time_ns"]
        run.shutdown()

        kept, replaced = Pair(), Pair()
        for node in (kept, replaced):
            inputs = {node.x: u(3)} if node is replaced else None
            restored = start_manual(Group(node=node), state=tree, inputs=inputs)
            restored.step(H)
            restored.shutdown()
        assert torch.equal(kept.seen[0][0][0], u(1))
        assert kept.seen[0][0][1] == Timestamp(ns=written_at)
        assert torch.equal(replaced.seen[0][0][0], u(3))
        # Written for the restored run, so stamped with the tree's time.
        assert replaced.seen[0][0][1] == Timestamp(ns=frozen_at)


class TestWrite:
    def test_switches_every_port_at_once(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        # Manual mode: the write and the update do not overlap, so the node
        # reads both values of one write.
        node = Pair()
        run = start_manual(Group(node=node))
        virtual.advance(H)
        run.write({node.x: u(1), node.y: u(2)})
        written_at = virtual.now()
        run.step(H)
        ((x, tx), (y, ty)) = node.seen[0]
        assert torch.equal(x, u(1)) and torch.equal(y, u(2))
        assert tx == ty == written_at

    def test_returns_the_time_every_value_carries(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        run = start_manual(Group(node=node))
        virtual.advance(H)
        written_at = run.write({node.x: u(1), node.y: u(2)})
        assert written_at == virtual.now()
        run.step(H)
        ((_, tx), (_, ty)) = node.seen[0]
        assert tx == ty == written_at
        tree = run.state_dict()
        assert port_slots(tree, "node", "x")["in"]["time_ns"] == written_at.ns
        assert port_slots(tree, "node", "y")["in"]["time_ns"] == written_at.ns

    def test_returns_the_time_a_node_reads_and_sends_on(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        echo = Echo()
        run = start_manual(echo)
        virtual.advance(H)
        written_at = run.write({echo.prev: u(1)})
        run.step(H)
        received_ns = run.read(echo.next)
        assert int(received_ns) == written_at.ns

    def test_of_nothing_returns_the_time_and_changes_nothing(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        run = start_manual(node, inputs={node.x: u(1)})
        virtual.advance(H)
        assert run.write({}) == Timestamp(ns=H_NS)
        slot = port_slots(run.state_dict(), "", "x")["in"]
        assert torch.equal(slot["value"], u(1)) and slot["time_ns"] == 0

    def test_is_stamped_after_the_inputs(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        virtual.advance(H)
        run = start_manual(node, inputs={node.x: u(1)})
        run.step(H)
        assert node.seen[0][0][1] == Timestamp(ns=H_NS)  # the time of the start
        written_at = run.write({node.x: u(2)})
        assert written_at == Timestamp(ns=2 * H_NS)  # later than the inputs

    def test_after_a_restore_continues_from_the_time_of_the_tree(
        self, start_manual: StartManual
    ):
        first = Pair()
        run = start_manual(first)
        run.step(H)
        tree = run.state_dict()
        run.shutdown()

        node = Pair()
        restored = start_manual(node, state=tree, inputs={node.x: u(3)})
        written_at = restored.write({node.y: u(4)})
        # The inputs carry the tree's time T and the timer goes on from T.
        assert written_at == Timestamp(ns=tree["timer"]["time_ns"])
        assert (
            port_slots(restored.state_dict(), "", "x")["in"]["time_ns"] == written_at.ns
        )

    def test_rejects_ports_it_cannot_write(self, start_manual: StartManual):
        net = Chain()
        pair = Pair()
        chain, solo = start_manual(net), start_manual(Group(node=pair))
        cases = [
            (chain, net.b.prev, "b.prev"),  # connected
            (solo, pair.many, "node.many"),  # a MultiPort
            (solo, pair.out, "node.out"),  # sends only
            (chain, pair.x, repr(pair.x)),  # of another run
        ]
        for run, written, name in cases:
            with pytest.raises(ValueError) as raised:
                run.write({written: u(0)})
            assert str(raised.value) == (
                f"the host does not write {name}: it writes an unconnected Port of "
                "this run that receives"
            )

    def test_rejects_values_that_do_not_match_the_spec(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError, match="shape"):
            run.write({net.prev: torch.zeros(3, dtype=F64)})
        with pytest.raises(ValueError, match="dtype"):
            run.write({net.prev: torch.zeros(2)})

    def test_raises_after_a_normal_end(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        run.step(H)
        run.shutdown()
        with pytest.raises(RuntimeError, match="stopped"):
            run.write({net.prev: u(2)})
        # Nothing was written: the tree keeps the input as it was at the end.
        assert torch.equal(
            port_slots(run.state_dict(), "a", "prev")["in"]["value"], u(1)
        )


class TestRead:
    def test_gives_the_latest_output(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        run.step(H)
        assert torch.equal(run.read(net.next), net.b.x)
        inner = run.read(net.a.next)  # a connected port shows an inner node
        assert torch.equal(inner, net.a.x)

    def test_with_time_gives_the_latest_output_and_its_time(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        run.step(H)
        out, t = run.read_with_time(net.next)
        assert torch.equal(out, net.b.x)
        assert isinstance(t, Timestamp) and t == virtual.now()

    def test_gives_zeros_before_the_first_send(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        out, t = run.read_with_time(net.next)
        assert torch.equal(out, torch.zeros(2, dtype=F64))
        assert t is None

    def test_rejects_a_port_that_sends_nothing(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(
            ValueError,
            match=r"^the host does not read a\.prev: it reads a port of this run that "
            "sends$",
        ):
            run.read(net.prev)

    def test_rejects_a_port_outside_the_run(self, start_manual: StartManual):
        run = start_manual(Chain())
        other = Chain()
        with pytest.raises(
            ValueError, match=rf"^the host does not read {re.escape(repr(other.next))}"
        ):
            run.read(other.next)

    def test_with_time_rejects_what_read_rejects(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError, match="does not read a.prev"):
            run.read_with_time(net.prev)


class TestSubscribe:
    def test_gives_the_latest_value_and_ends_with_the_network(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        run.step(H)
        values = run.subscribe(net.next)
        # First the value at the time of the call, without waiting for a new one.
        out, t = next(values)
        assert torch.equal(out, net.b.x) and t == virtual.now()
        for _ in range(3):
            run.step(H)
        # The first two values were skipped: only the latest is given.
        out, t = next(values)
        assert torch.equal(out, net.b.x) and t == virtual.now()
        run.step(H)
        assert next(values)[1] == virtual.now()
        run.shutdown()
        with pytest.raises(StopIteration):
            next(values)

    def test_gives_the_value_before_the_first_send(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        out, _ = next(run.subscribe(net.next))
        assert torch.equal(out, torch.zeros(2, dtype=F64))

    def test_gives_the_last_value_after_a_normal_end(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        run.step(H)
        run.shutdown()
        assert [t for _, t in run.subscribe(net.next)] == [virtual.now()]

    def test_follows_a_free_running_network(self, start: Start):
        net = Chain()
        run = start(net, inputs={net.prev: u(1)})
        times: list[Timestamp] = []
        stopper = threading.Thread(target=run.shutdown)
        for _, t in run.subscribe(net.next):
            if t is None:  # the zeros before the first send
                continue
            times.append(t)
            if len(times) == 5:
                stopper.start()
        stopper.join()
        assert times == sorted(times)
        assert len(times) >= 5

    def test_raises_after_the_timeout_and_the_network_keeps_running(
        self, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        values = run.subscribe(net.next, timeout=0.05)
        next(values)  # the first step does not wait
        begin = time.monotonic()
        with pytest.raises(TimeoutError):
            next(values)  # nothing is sent: the nodes wait for step
        assert time.monotonic() - begin < 5.0
        with pytest.raises(StopIteration):
            next(values)
        run.write({net.prev: u(1)})
        run.step(H)
        assert torch.equal(run.read(net.next), net.b.x)

    def test_the_timeout_counts_the_whole_iteration(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        values = run.subscribe(net.next, timeout=0.05)
        next(values)
        time.sleep(0.1)  # the body of a loop that took longer than the timeout
        run.step(H)  # a newer message is there
        with pytest.raises(TimeoutError):
            next(values)

    def test_yields_what_arrives_within_the_timeout(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        values = run.subscribe(net.next, timeout=60.0)
        next(values)
        run.step(H)
        assert next(values)[1] == virtual.now()
        run.shutdown()
        with pytest.raises(StopIteration):
            next(values)


class TestWaitFor:
    def test_returns_the_first_message_that_meets_the_condition(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        seen: list[torch.Tensor] = []

        def condition(out: torch.Tensor) -> bool:
            seen.append(out)
            if len(seen) < 3:
                run.step(H)  # a newer message for the next check
                return False
            return True

        out, t = run.wait_for(net.next, condition, timeout=60.0)
        # The latest message first (zeros), then each newer one.
        assert torch.equal(seen[0], torch.zeros(2, dtype=F64))
        assert len(seen) == 3
        assert torch.equal(out, net.b.x) and t == virtual.now()

    def test_raises_when_it_times_out(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(TimeoutError, match=r"within 0\.05 s"):
            run.wait_for(net.next, lambda _: False, timeout=0.05)

    def test_raises_when_the_run_shuts_down_first(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        stopper = threading.Timer(0.05, run.shutdown)
        stopper.start()
        with pytest.raises(RuntimeError, match=r"shut down before b\.next sent"):
            run.wait_for(net.next, lambda _: False)
        stopper.join()


class TestHostMethodsFromANode:
    @pytest.mark.parametrize("ended", [False, True])
    def test_raise_in_state_dict_on_the_control_thread(
        self, start_manual: StartManual, ended: bool
    ):
        runs: list[Run] = []

        class Reading(Pair):
            @override
            def state_dict(self) -> dict[str, Any]:
                try:
                    runs[0].read(self.out)
                except RuntimeError as e:
                    return {"error": str(e)}
                return {}

        runs.append(start_manual(Group(node=Reading())))
        if ended:
            runs[0].shutdown()
        assert "host methods" in runs[0].state_dict()["nodes"]["node"]["state"]["error"]

    @pytest.mark.parametrize("dedicated_thread", [False, True])
    def test_raise(self, start_manual: StartManual, dedicated_thread: bool):
        runs: list[Run] = []

        def call_every_host_method() -> None:
            run = runs[0]
            for call in (
                lambda: run.write({net.prev: u(0)}),
                lambda: run.read(net.next),
                lambda: run.subscribe(net.next),
                run.state_dict,
                run.shutdown,
                run.join,
                lambda: mujou.run(Group(node=Pair())),
            ):
                try:
                    call()
                except RuntimeError as e:
                    node.errors.append(e)

        node = Calls(call_every_host_method, dedicated_thread=dedicated_thread)
        net = Chain()
        net.caller = node  # pyright: ignore[reportAttributeAccessIssue]
        run = start_manual(net)
        runs.append(run)
        run.step(H)
        assert len(node.errors) == 7
        assert all("host methods" in str(e) for e in node.errors)


class TestNodeException:
    def test_stops_the_network_and_reaches_the_host(self, start_manual: StartManual):
        net = Chain(raise_in_b=3)
        run = start_manual(net)
        values = run.subscribe(net.next)
        run.step(H)
        run.step(H)
        with pytest.raises(NetworkError) as raised:
            run.step(H)
        error = net.b.error
        assert raised.value.__cause__ is error
        calls: list[Callable[[], object]] = [
            lambda: run.write({net.prev: u(0)}),
            lambda: run.read(net.next),
            lambda: run.subscribe(net.next),
            lambda: next(values),
            run.state_dict,
            run.shutdown,
            run.join,
        ]
        for call in calls:
            with pytest.raises(NetworkError) as raised:
                call()
            assert raised.value.__cause__ is error

    def test_a_run_that_raised_is_freed_once_the_host_drops_it(self):
        def failed() -> weakref.ref[Chain]:
            net = Chain(raise_in_b=1)
            run = mujou.testing.run(net)
            with pytest.raises(NetworkError):
                run.step(H)
            return weakref.ref(net)

        # Without a shutdown, as in a notebook that makes the run again.
        refs = [failed() for _ in range(3)]
        gc.collect()
        assert [ref() for ref in refs] == [None, None, None]

    def test_ends_a_subscription(self, start: Start):
        net = Chain()
        run = start(net)
        values = run.subscribe(net.next)
        next(values)  # the iterator is waiting on the running network
        net.b.trigger.set()
        with pytest.raises(NetworkError) as raised:
            for _ in values:
                pass
        assert raised.value.__cause__ is net.b.error


class SlowStop(Node):
    """Waits in ``on_stop`` until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=vec())
        self.release = threading.Event()
        self.stopped = threading.Event()

    @override
    def on_stop(self) -> None:
        self.release.wait(10.0)
        self.stopped.set()


class TestShutdown:
    def test_does_nothing_after_the_end(self, start: Start):
        run = start(Chain())
        run.shutdown()
        run.shutdown()
        run.shutdown(timeout=0.0)

    def test_raises_after_the_timeout_and_the_network_goes_on_stopping(
        self, start_manual: StartManual
    ):
        node = SlowStop()
        net = Group(node=node)
        run = start_manual(net)
        before = threading.active_count()
        try:
            started = time.monotonic()
            for _ in range(5):  # asked again without a thread for each ask
                with pytest.raises(TimeoutError):
                    run.shutdown(timeout=0.02)
            assert time.monotonic() - started < 5.0
            assert not node.stopped.is_set()
            assert threading.active_count() <= before
        finally:
            node.release.set()
        run.join()
        assert node.stopped.is_set()
        run.state_dict()  # a normal end

    @pytest.mark.parametrize("timeout", [-1.0, math.nan])
    def test_a_bad_timeout_raises_before_the_stop(
        self, start_manual: StartManual, timeout: float
    ):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError):
            run.shutdown(timeout=timeout)
        run.write({net.prev: u(1)})
        run.step(H)  # still running
        assert torch.equal(run.read(net.next), net.b.x)

    def test_raises_what_a_node_raised(self, start_manual: StartManual):
        net = Chain(raise_in_b=1)
        run = start_manual(net)
        with pytest.raises(NetworkError):
            run.step(H)
        with pytest.raises(NetworkError) as raised:
            run.shutdown(timeout=1.0)
        assert raised.value.__cause__ is net.b.error


class TestJoin:
    def test_returns_once_another_thread_shuts_down(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(TimeoutError):
            run.join(timeout=0.0)  # still running
        # Shut down a little later, so that join is (almost always) already waiting.
        stopper = threading.Timer(0.05, run.shutdown)
        stopper.start()
        run.join()
        stopper.join()
        run.state_dict()  # a normal end


class TestTimeout:
    @pytest.mark.parametrize("method", ["join", "wait_for"])
    def test_raises_after_the_timeout_and_the_network_keeps_running(
        self, start_manual: StartManual, method: str
    ):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(TimeoutError):
            if method == "join":
                run.join(timeout=0.05)
            else:
                run.wait_for(net.next, lambda _: False, timeout=0.05)
        run.write({net.prev: u(1)})
        run.step(H)
        assert torch.equal(run.read(net.next), net.b.x)


class TestManualRun:
    def test_updates_only_the_nodes_whose_time_advanced(
        self, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.step(0.0)  # the time has not advanced since the start
        assert (net.a.updates, net.b.updates) == (0, 0)
        run.step(H, net.a)
        run.step(0.0)  # only b has not updated since then
        assert (net.a.updates, net.b.updates) == (1, 1)

    def test_rejects_a_node_of_another_run(self, start_manual: StartManual):
        run = start_manual(Chain())
        with pytest.raises(ValueError, match="not a node of this run"):
            run.step(0.0, Leaky())

    def test_is_what_mujou_testing_run_returns(self, virtual: VirtualTimer):
        with mujou.testing.run(Chain(), timer=virtual) as run:
            assert isinstance(run, ManualRun) and isinstance(run, Run)

    def test_runs_on_a_new_virtual_timer_by_default(self):
        net = Chain()
        with mujou.testing.run(net, inputs={net.prev: u(1)}) as run:
            timer = run.timer
            assert isinstance(timer, VirtualTimer)
            assert timer.now() == Timestamp(ns=0)
            for _ in range(3):
                run.step(H)  # the clock moves only here
            out, t = run.read_with_time(net.next)
        x = torch.zeros(2, dtype=F64)
        for _ in range(3):
            x = euler(x, u(1))
        assert torch.equal(net.a.x, x)
        assert t == Timestamp(ns=3 * H_NS) and torch.equal(out, net.b.x)

    def test_each_run_gets_its_own_timer(self):
        with (
            mujou.testing.run(Chain()) as first,
            mujou.testing.run(Chain()) as second,
        ):
            assert first.timer is not second.timer

    def test_gives_the_timer_it_was_given(self, start_manual: StartManual):
        timer = VirtualTimer()
        run = start_manual(Chain(), timer=timer)
        assert run.timer is timer


class TestEndToEnd:
    @pytest.mark.slow
    def test_approaches_the_equilibrium_in_real_time(self, start: Start):
        tau, presented = 0.01, 0.3  # 30 time constants
        net = Chain(tau=tau)
        run = start(net)
        before = torch.zeros(2, dtype=F64)
        for k in (1, 2):
            run.write({net.prev: u(k)})
            run.timer.sleep(presented)
            out = run.read(net.next)
            # After a step of the input by d, the error of two leaky stages in a
            # row is d (1 + t/tau) exp(-t/tau). Allow the nodes to have run for
            # only a third of the time (scheduling under load): t = 10 tau.
            d = (u(k) - before).abs().max().item()
            bound = d * (1 + 10) * math.exp(-10)
            assert (out - u(k)).abs().max().item() <= bound
            before = u(k)

    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize("b_first", [False, True])
    def test_manual_steps_match_the_discrete_algorithm(
        self,
        start_manual: StartManual,
        b_first: bool,
        grouped: bool,
    ):
        a, b = Leaky(), Leaky()
        if grouped:
            # The same two nodes as two networks: a network is no boundary of the
            # asynchrony, and both run on the one timer of the run.
            first, second = Group(node=a), Group(node=b)
            a.next.connect(b.prev)
            run = start_manual(Group(first=first, second=second))
            assert list(run.state_dict()["nodes"]) == ["first.node", "second.node"]
        else:
            chain = Chain()
            a, b = chain.a, chain.b
            run = start_manual(chain)
        order = (b, a) if b_first else (a, b)
        xa = xb = sent_a = torch.zeros(2, dtype=F64)
        for k in range(20):
            run.write({a.prev: u(k)})
            run.step(H, *order)
            if b_first:  # Jacobi: b reads a's value of the previous step
                xb = euler(xb, sent_a)
                xa = sent_a = euler(xa, u(k))
            else:  # Gauss-Seidel: b reads a's value of this step
                xa = sent_a = euler(xa, u(k))
                xb = euler(xb, sent_a)
        assert torch.equal(a.x, xa)
        assert torch.equal(b.x, xb)
        assert torch.equal(run.read(b.next), xb)


class TestInterpreterExit:
    """Run a script in a new interpreter and see what happens at its exit."""

    PRELUDE = """
import pathlib, sys, threading, time
import torch
import mujou
from mujou import Network, Node, Port, TensorSpec

class Slow(Node):
    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=TensorSpec((2,)))

    def on_update(self, dt: float) -> None:
        self.out.send(torch.ones(2))

    def on_stop(self) -> None:
        time.sleep(0.3)  # an actuator taking its time to stop
        pathlib.Path(sys.argv[1]).write_text("stopped")

class Net(Network):
    def __init__(self) -> None:
        super().__init__()
        self.node = Slow()
        self.out = self.node.out
"""

    def run_script(
        self, body: str, marker: Path, gil: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ if gil is None else {**os.environ, "PYTHON_GIL": gil}
        return subprocess.run(
            [sys.executable, "-c", self.PRELUDE + body, str(marker)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "gil",
        [
            pytest.param(
                "0",
                marks=pytest.mark.skipif(
                    not sysconfig.get_config_var("Py_GIL_DISABLED"),
                    reason="requires a free-threaded build",
                ),
            ),
            "1",
        ],
    )
    def test_waits_for_on_stop_of_a_dropped_network(self, tmp_path: Path, gil: str):
        marker = tmp_path / "marker"
        body = "def main() -> None:\n    mujou.run(Net())\nmain()\n"
        done = self.run_script(body, marker, gil)
        assert done.returncode == 0, done.stderr
        assert marker.read_text() == "stopped"

    @pytest.mark.slow
    def test_ctrl_c_while_starting_raises_once_the_network_has_stopped(
        self, tmp_path: Path
    ):
        marker = tmp_path / "marker"
        body = (
            "import signal\n"
            "class SlowStart(Slow):\n"
            "    def on_start(self) -> None:\n"
            "        signal.raise_signal(signal.SIGINT)\n"
            "        time.sleep(0.5)  # the host sees Ctrl-C meanwhile\n"
            "try:\n"
            "    mujou.run(SlowStart())\n"
            "except KeyboardInterrupt:\n"
            "    print(pathlib.Path(sys.argv[1]).read_text())\n"
        )
        done = self.run_script(body, marker)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "stopped"


# Large enough that the GPU is still working when the CPU moves on: a missing wait
# for a stream shows as a value in the middle of an update.
