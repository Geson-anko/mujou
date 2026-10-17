import math
import pickle
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import pytest
import torch

import mujou
import mujou.testing
from mujou import NetworkError, Run
from mujou._core import session as session_module
from mujou._core.network import Compose, Network
from mujou._core.node import Node
from mujou._core.port import NextPort, Port, PrevPort
from mujou._core.spec import TensorSpec
from mujou.testing import ManualRun, VirtualTimer
from mujou.time import ReadOnlyTimer, Timer, Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual, Step

H_NS = 1_000_000
H = H_NS / 1e9  # The step of the `step` fixture (conftest's H), in seconds.
F64 = torch.float64


def vec() -> TensorSpec:
    return TensorSpec((2,), F64)


def u(k: float) -> torch.Tensor:
    """The k-th input: different at every step."""
    return torch.tensor([math.sin(k), math.cos(k)], dtype=F64)


class Leaky(Node):
    """``dx/dt = -(x - u) / tau`` by explicit Euler, the rate clipped at 1.

    ``u`` comes from the previous side and ``x`` goes to the next side. Raises in
    the ``raise_at``-th ``on_update`` (or in ``on_start`` with ``raise_at=0``).
    """

    def __init__(self, tau: float = 1.0, raise_at: int | None = None) -> None:
        self.prev = PrevPort(recv=vec())
        self.next = NextPort(send=vec())
        self.tau = tau
        self.raise_at = raise_at
        # Set by a test to make the next on_update raise.
        self.trigger = threading.Event()
        self.x = torch.zeros(2, dtype=F64)
        self.updates = 0
        self.started = False
        self.error: Exception | None = None

    def fail(self) -> None:
        self.error = ValueError(f"boom after {self.updates} updates")
        raise self.error

    @override
    def on_start(self) -> None:
        self.started = True
        if self.raise_at == 0:
            self.fail()

    @override
    def on_update(self, dt: float) -> None:
        self.updates += 1
        if self.updates == self.raise_at or self.trigger.is_set():
            self.fail()
        self.x += min(dt / self.tau, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)

    @override
    def state_dict(self) -> object:
        return {"x": self.x, "updates": self.updates}

    @override
    def load_state_dict(self, state: object) -> None:
        loaded = cast(dict[str, Any], state)
        self.x = loaded["x"]
        self.updates = loaded["updates"]


class Chain(Network):
    """``u -> a -> b ->``: two leaky integrators in a row."""

    def __init__(
        self,
        tau: float = 1.0,
        raise_in_a: int | None = None,
        raise_in_b: int | None = None,
    ) -> None:
        self.a = Leaky(tau, raise_in_a)
        self.b = Leaky(tau, raise_in_b)
        self.a.connect(self.b)
        self.prev = self.a.prev
        self.next = self.b.next


class Pair(Node):
    """Receives two inputs from the host and keeps them with their times."""

    def __init__(self) -> None:
        self.x = Port(recv=vec())
        self.y = Port(recv=vec())
        self.many = Port(recv=vec(), multi=True)
        self.out = Port(send=vec())
        self.seen: list[
            tuple[tuple[torch.Tensor, Timestamp], tuple[torch.Tensor, Timestamp]]
        ] = []

    @override
    def on_update(self, dt: float) -> None:
        self.seen.append((self.x.recv_with_time(), self.y.recv_with_time()))


class Echo(Node):
    """Sends the time of the message it receives, in nanoseconds."""

    def __init__(self) -> None:
        self.prev = PrevPort(recv=vec())
        self.next = NextPort(send=TensorSpec((), torch.int64))

    @override
    def on_update(self, dt: float) -> None:
        _, t = self.prev.recv_with_time()
        self.next.send(torch.tensor(t.ns))


class Calls(Node):
    """Calls ``action`` in ``on_update`` and keeps the exception it raises."""

    def __init__(self, action: Callable[[], object]) -> None:
        self.action = action
        self.errors: list[Exception] = []

    @override
    def on_update(self, dt: float) -> None:
        try:
            self.action()
        except Exception as e:  # noqa: BLE001 - kept for the test to inspect
            self.errors.append(e)


def euler(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The reference of one ``Leaky`` update with ``tau=1`` and ``dt=H``."""
    return x + min(H / 1.0, 1.0) * (target - x)


class TestRun:
    def test_returns_once_every_node_has_started(self, start: Start):
        net = Chain()
        run = start(net)
        assert isinstance(run, Run)
        assert net.a.started and net.b.started  # run returns after on_start

    def test_runs_a_built_network(self, start: Start):
        net = Chain()
        graph = net.build()
        start(net)
        assert net.build() is graph

    @pytest.mark.parametrize(
        ("make", "again", "message"),
        [
            pytest.param(Chain, lambda c: (c,), "Chain", id="network"),
            # Not the path "0" of mujou.run's Compose.
            pytest.param(Leaky, lambda c: (c,), "Leaky", id="node"),
            pytest.param(
                Leaky,
                lambda c: (Compose(node=c),),
                "Leaky at 'node'",
                id="node-in-a-network",
            ),
            pytest.param(
                Leaky, lambda c: (Compose(c),), "Leaky at '0'", id="node-composed"
            ),
            # Named by its path in the tree mujou.run makes.
            pytest.param(
                Chain,
                lambda c: (Compose(x=c), Chain()),
                "Chain at '0.x'",
                id="network-among-several",
            ),
        ],
    )
    def test_runs_once(
        self,
        start: Start,
        make: Callable[[], Node | Network],
        again: Callable[[Node | Network], tuple[Node | Network, ...]],
        message: str,
    ):
        component = make()
        start(component).shutdown()
        with pytest.raises(RuntimeError) as raised:
            start(*again(component))
        assert str(raised.value) == (
            f"{message} has already run; create a new instance to run again"
        )

    def test_raises_what_on_start_raised(self, start: Start):
        net = Chain(raise_in_b=0)
        with pytest.raises(NetworkError) as raised:
            start(net)
        assert raised.value.__cause__ is net.b.error
        assert net.a.updates == 0 and net.b.updates == 0

    @pytest.mark.parametrize(
        ("make", "message"),
        [
            (Chain, "Chain has already run; create a new instance to run again"),
            (Leaky, "Leaky has already run; create a new instance to run again"),
        ],
    )
    def test_runs_once_under_concurrent_calls(
        self, started: list[Run], make: Callable[[], Node | Network], message: str
    ):
        for _ in range(20):
            net = make()
            barrier = threading.Barrier(2)
            outcomes: list[object] = []

            def run(
                net: Node | Network = net, barrier: threading.Barrier = barrier
            ) -> None:
                barrier.wait()
                try:
                    handle = mujou.testing.run(net, timer=VirtualTimer())
                except RuntimeError as e:
                    outcomes.append(e)
                    return
                started.append(handle)
                outcomes.append(handle)

            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert sum(isinstance(o, Run) for o in outcomes) == 1
            (error,) = (o for o in outcomes if not isinstance(o, Run))
            assert str(error) == message

    @pytest.mark.parametrize("shared", ["network", "node"])
    def test_runs_once_when_trees_sharing_a_part_run_together(
        self, started: list[Run], shared: str
    ):
        # Refused while the other run is still starting, a run may find the
        # shared part only built rather than run: either message is right.
        refused = (
            r"(has already run; create a new instance to run again"
            r"|belongs to a network that is already built)$"
        )
        for _ in range(5):
            given: list[tuple[Node | Network, ...]]
            if shared == "network":
                net = Chain()
                given = [(net,), (net, Chain())]
                who = r"^Chain "
            else:
                node = Leaky()
                given = [(node,), (Compose(node=node),)]
                who = r"^Leaky (at 'node' )?"
            barrier = threading.Barrier(2)
            outcomes: list[object] = []

            def run(
                components: tuple[Node | Network, ...],
                barrier: threading.Barrier = barrier,
                outcomes: list[object] = outcomes,
            ) -> None:
                barrier.wait()
                try:
                    handle = mujou.testing.run(*components, timer=VirtualTimer())
                except RuntimeError as e:
                    outcomes.append(e)
                    return
                started.append(handle)
                outcomes.append(handle)

            threads = [threading.Thread(target=run, args=(g,)) for g in given]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert sum(isinstance(o, Run) for o in outcomes) == 1
            (error,) = (o for o in outcomes if not isinstance(o, Run))
            assert isinstance(error, RuntimeError)
            assert re.match(who + refused, str(error)), str(error)

    def test_a_network_runs_either_alone_or_composed(self, start_manual: StartManual):
        alone, other = Chain(), Chain()
        start_manual(alone)
        with pytest.raises(RuntimeError, match="^Chain has already run"):
            start_manual(alone, other)
        composed, other = Chain(), Chain()
        start_manual(composed, other)
        with pytest.raises(RuntimeError, match="^Chain has already run"):
            start_manual(composed)

    def test_takes_only_nodes_and_networks(self, start: Start):
        for given in ((object(),), (Chain(), object())):
            with pytest.raises(TypeError, match="object"):
                start(*given)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("pause_timeout", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_a_pause_timeout_that_is_not_finite_and_positive(
        self, pause_timeout: float, start_manual: StartManual
    ):
        net = Chain()
        for run in (mujou.run, mujou.testing.run):
            with pytest.raises(ValueError, match="^pause_timeout must be"):
                run(net, pause_timeout=pause_timeout)
        # Refused before the build: the network is not marked as built, so it can
        # still run with others (a network built alone could not).
        start_manual(net, Chain())

    def test_a_run_is_made_only_by_mujou_run(self):
        with pytest.raises(TypeError, match=r"mujou\.run"):
            Run()


class TestRunANode:
    def test_runs_as_the_root(self, step: Step, start_manual: StartManual):
        node = Leaky()
        run = start_manual(node, inputs={node.prev: u(1)})
        step(run, node)
        y, _ = run.read(node.next)
        assert torch.equal(y, euler(torch.zeros(2, dtype=F64), u(1)))
        tree = run.state_dict()
        assert set(tree["nodes"]) == {""}
        assert set(tree["slots"]) == {"prev/in", "next/out"}
        assert set(tree["last_update_ns"]) == {""}

    def test_a_network_alone_is_the_root_and_several_are_at_their_positions(
        self, start_manual: StartManual
    ):
        alone = start_manual(Chain()).state_dict()
        assert set(alone["nodes"]) == {"a", "b"}
        several = start_manual(Leaky(), Chain()).state_dict()
        assert set(several["nodes"]) == {"0", "1.a", "1.b"}
        assert "0.prev/in" in several["slots"]
        composed = start_manual(Compose(Leaky())).state_dict()
        assert set(composed["nodes"]) == {"0"}

    def test_names_the_root_node_by_its_class_in_a_structure_error(self, start: Start):
        class Declared(Node):
            port = Port(send=vec())

        with pytest.raises(ValueError, match=r"^Declared declares the port 'port'"):
            start(Declared())

    def test_names_the_root_node_by_its_class_when_it_raises(self, start: Start):
        node = Leaky(raise_at=0)
        with pytest.raises(NetworkError) as raised:
            start(node)
        assert "node 'Leaky' raised in on_start" in str(raised.value)

    def test_names_its_ports_by_its_class_in_errors(self, start_manual: StartManual):
        # Their paths ("x") alone would not say which node they are of.
        node = Pair()
        run = start_manual(node)
        dtype = (
            r"^Pair\.x: expected dtype torch\.float64, got torch\.float32; dtypes "
            r"are not converted, use \.to\(torch\.float64\)$"
        )
        with pytest.raises(TypeError, match=dtype):
            run.write({node.x: torch.zeros(2, dtype=torch.float32)})
        with pytest.raises(ValueError, match=r"^cannot write the port 'Pair\.out'"):
            run.write({node.out: u(0)})
        with pytest.raises(RuntimeError, match=r"owns the port 'Pair\.x'"):
            node.x.recv()  # from the host, not from the node's hooks

    def test_a_node_in_a_network_names_its_ports_by_their_paths(
        self, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError, match=r"^cannot write the port 'b\.prev'"):
            run.write({net.b.prev: u(0)})

    def test_restores_a_tree_of_the_same_node_class(
        self, step: Step, start_manual: StartManual
    ):
        first = Leaky()
        run = start_manual(first, inputs={first.prev: u(1)})
        step(run)
        tree = pickle.loads(pickle.dumps(run.state_dict()))  # saved and loaded
        run.shutdown()
        again = Leaky()
        restored = start_manual(again, timer=VirtualTimer(), state=tree)
        assert torch.equal(again.x, first.x) and again.updates == 1
        back = restored.state_dict()
        assert torch.equal(back["nodes"][""]["x"], tree["nodes"][""]["x"])
        assert back["slots"]["prev/in"][1] == tree["slots"]["prev/in"][1]

    def test_a_node_of_a_built_network_is_named_without_the_path_of_mujou_run(
        self, start: Start
    ):
        net = Chain()
        net.build()
        with pytest.raises(RuntimeError) as raised:
            start(net.a)
        assert str(raised.value) == "Leaky belongs to a network that is already built"

    def test_mixes_with_networks(self, step: Step, start_manual: StartManual):
        net, node = Chain(), Leaky()
        net.connect(node)
        run = start_manual(net, node, inputs={net.prev: u(1)})
        assert set(run.state_dict()["nodes"]) == {"0.a", "0.b", "1"}
        a = b = n = torch.zeros(2, dtype=F64)
        for _ in range(3):
            step(run, net.a, net.b, node)
            # In the order of the steps, each reads the value just sent before it.
            a = euler(a, u(1))
            b = euler(b, a)
            n = euler(n, b)
        y, _ = run.read(node.next)
        assert torch.equal(y, n)


class TestRefusedRun:
    @pytest.mark.parametrize(
        "given",
        ["inputs", "inputs-of-several", "inputs-of-a-node", "state", "busy-timer"],
    )
    def test_starts_nothing_and_can_be_retried(
        self, start_manual: StartManual, given: str
    ):
        net, other, node = Chain(), Chain(), Leaky()
        wrong = torch.zeros(3, dtype=F64)
        # The refused calls with what their errors match, then the call retried
        # and the nodes of its run.
        refused: list[tuple[Callable[[], object], str | None]]
        if given in ("inputs", "inputs-of-several"):
            components = (net,) if given == "inputs" else (net, other)
            refused = [
                (partial(start_manual, *components, inputs={net.prev: wrong}), "shape")
            ]
            retry = partial(start_manual, *components, inputs={net.prev: u(1)})
            nodes = {"a", "b"} if given == "inputs" else {"0.a", "0.b", "1.a", "1.b"}
        elif given == "inputs-of-a-node":
            refused = [
                (partial(start_manual, node, inputs={node.prev: wrong}), "shape")
            ]
            # In another way: in a network.
            retry = partial(start_manual, Compose(node=node), inputs={node.prev: u(1)})
            nodes = {"node"}
        else:
            source = start_manual(Chain())
            tree = source.state_dict()
            nodes = {"a", "b"}
            if given == "state":
                source.shutdown()
                last_update_ns = {**tree, "last_update_ns": {"a": 0.5, "b": 0}}
                scale = {**tree, "timer": {**tree["timer"], "scale": -1.0}}
                refused = [
                    (partial(start_manual, net, state=last_update_ns), "not an int"),
                    # Refused by the timer before any node starts.
                    (partial(start_manual, net, state=scale), None),
                ]
                retry = partial(start_manual, net, state=tree)
            else:
                # The source still runs on the timer: it cannot be set back.
                refused = [
                    (partial(start_manual, net, state=tree), "a timer of its own")
                ]
                retry = partial(start_manual, net, timer=VirtualTimer(), state=tree)
        for call, match in refused:
            with pytest.raises(ValueError, match=match):
                call()
        assert not (net.a.started or other.a.started or node.started)
        assert set(retry().state_dict()["nodes"]) == nodes


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
            virtual.advance(H)
            with pytest.raises(NetworkError):
                run.step()
        # From the shutdown at the end of the with statement.
        assert raised.value.__cause__ is net.b.error

    def test_lets_the_exception_of_the_body_through(self, virtual: VirtualTimer):
        net = Chain(raise_in_b=1)
        run = mujou.testing.run(net, timer=virtual)
        with pytest.raises(KeyError), run:
            virtual.advance(H)
            with pytest.raises(NetworkError):
                run.step()
            raise KeyError("the body's own")
        with pytest.raises(NetworkError):
            run.join()


class TestTimer:
    def test_the_slots_and_the_nodes_run_on_the_timer_of_the_run(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        run = start_manual(Compose(node=node), timer=virtual)
        virtual.advance(2.5)
        run.write({node.x: u(1)})
        step(run)
        ((_, written_at), _) = node.seen[0]
        assert written_at == Timestamp(ns=2_500_000_000)
        tree = run.state_dict()
        assert tree["timer"]["time_ns"] == virtual.now().ns
        assert tree["last_update_ns"]["node"] == virtual.now().ns

    def test_a_pause_stops_only_the_timer_of_its_run(self, start: Start):
        held, free = Timer(), Timer()

        class Measures(Node):
            @override
            def state_dict(self) -> object:
                # Called while the run on `held` is paused.
                h0, f0, g0 = held.now(), free.now(), mujou.time.now()
                time.sleep(0.05)
                return held.now() - h0, free.now() - f0, mujou.time.now() - g0

        start(Chain(tau=0.01), timer=free)
        run = start(Compose(node=Measures()), timer=held)
        on_held, on_free, on_global = run.state_dict()["nodes"]["node"]
        assert on_held == 0.0
        assert on_free > 0.0 and on_global > 0.0

    def test_the_global_timer_by_default(self, start: Start):
        class ReadsTimer(Node):
            seen: ReadOnlyTimer | None = None

            @override
            def on_start(self) -> None:
                self.seen = self.timer

        node = ReadsTimer()
        start(Compose(node=node))
        seen = node.seen
        assert seen is not None
        # On one time axis with mujou.time.
        first = seen.now()
        now_ns = mujou.time.now().ns
        assert first.ns <= now_ns <= seen.now().ns


class TestInputs:
    def test_are_the_initial_values(self, step: Step, start_manual: StartManual):
        net = Chain()
        run = start_manual(net, inputs={net.prev: u(1)})
        step(run)
        assert torch.equal(net.a.x, euler(torch.zeros(2, dtype=F64), u(1)))

    def test_after_a_state_take_precedence_and_carry_its_time(
        self, step: Step, start_manual: StartManual
    ):
        first = Pair()
        run = start_manual(Compose(node=first), inputs={first.x: u(1)})
        step(run)
        tree = run.state_dict()
        written_at = tree["slots"]["node.x/in"][1]
        frozen_at = tree["timer"]["time_ns"]
        run.shutdown()

        kept, replaced = Pair(), Pair()
        for node in (kept, replaced):
            timer = VirtualTimer()
            timer.advance(100.0)  # far from the tree's time
            inputs = {node.x: u(3)} if node is replaced else None
            restored = start_manual(
                Compose(node=node), timer=timer, state=tree, inputs=inputs
            )
            step(restored)
            restored.shutdown()
        assert torch.equal(kept.seen[0][0][0], u(1))
        assert kept.seen[0][0][1].ns == written_at
        assert torch.equal(replaced.seen[0][0][0], u(3))
        # Written for the restored run, so stamped with the tree's time.
        assert replaced.seen[0][0][1].ns == frozen_at


class TestWrite:
    def test_switches_every_port_at_once(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        # Manual mode: the write and the update do not overlap, so the node
        # reads both values of one write.
        node = Pair()
        run = start_manual(Compose(node=node))
        virtual.advance(H)
        run.write({node.x: u(1), node.y: u(2)})
        written_at = virtual.now()
        step(run)
        ((x, tx), (y, ty)) = node.seen[0]
        assert torch.equal(x, u(1)) and torch.equal(y, u(2))
        assert tx == ty == written_at

    def test_returns_the_time_every_value_carries(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        run = start_manual(Compose(node=node))
        virtual.advance(H)
        written_at = run.write({node.x: u(1), node.y: u(2)})
        assert written_at == virtual.now()
        step(run)
        ((_, tx), (_, ty)) = node.seen[0]
        assert tx == ty == written_at
        slots = run.state_dict()["slots"]
        assert slots["node.x/in"][1] == slots["node.y/in"][1] == written_at.ns

    def test_returns_the_time_a_node_reads_and_sends_on(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        echo = Echo()
        run = start_manual(echo)
        virtual.advance(H)
        written_at = run.write({echo.prev: u(1)})
        step(run)
        (received_ns, _) = run.read(echo.next)
        assert int(received_ns) == written_at.ns

    def test_of_nothing_returns_the_time_and_changes_nothing(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        run = start_manual(node, inputs={node.x: u(1)})
        virtual.advance(H)
        assert run.write({}) == Timestamp(ns=H_NS)
        slot = run.state_dict()["slots"]["x/in"]
        assert torch.equal(slot[0], u(1)) and slot[1] == 0

    def test_is_stamped_after_the_inputs(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Pair()
        virtual.advance(H)
        run = start_manual(node, inputs={node.x: u(1)})
        step(run)
        assert node.seen[0][0][1] == Timestamp(ns=H_NS)  # the time of the start
        written_at = run.write({node.x: u(2)})
        assert written_at == Timestamp(ns=2 * H_NS) > node.seen[0][0][1]

    def test_after_a_restore_continues_from_the_time_of_the_tree(
        self, step: Step, start_manual: StartManual
    ):
        first = Pair()
        run = start_manual(first)
        step(run)
        tree = run.state_dict()
        run.shutdown()

        node, timer = Pair(), VirtualTimer()
        timer.advance(100.0)  # far from the tree's time
        restored = start_manual(node, timer=timer, state=tree, inputs={node.x: u(3)})
        written_at = restored.write({node.y: u(4)})
        # The inputs carry the tree's time T and the timer goes on from T.
        assert written_at == Timestamp(ns=tree["timer"]["time_ns"])
        assert restored.state_dict()["slots"]["x/in"][1] == written_at.ns

    def test_rejects_ports_it_cannot_write(self, start_manual: StartManual):
        net = Chain()
        pair = Pair()
        chain, solo = start_manual(net), start_manual(Compose(node=pair))
        cases = [
            (chain, net.b.prev, "is connected"),
            (solo, pair.many, "many peers"),
            (solo, pair.out, "receives no messages"),
            (chain, pair.x, "not a port of a node in this run"),
        ]
        for run, port, match in cases:
            with pytest.raises(ValueError, match=match):
                run.write({port: u(0)})

    def test_rejects_values_that_do_not_match_the_spec(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError, match="shape"):
            run.write({net.prev: torch.zeros(3, dtype=F64)})
        with pytest.raises(TypeError, match="dtype"):
            run.write({net.prev: torch.zeros(2)})

    def test_raises_after_a_normal_end(self, step: Step, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        step(run)
        run.shutdown()
        with pytest.raises(RuntimeError, match="stopped"):
            run.write({net.prev: u(2)})
        # Nothing was written: the tree keeps the input as it was at the end.
        assert torch.equal(run.state_dict()["slots"]["a.prev/in"][0], u(1))


class TestRead:
    def test_gives_the_latest_output_and_its_time(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        step(run)
        out, t = run.read(net.next)
        assert torch.equal(out, net.b.x)
        assert isinstance(t, Timestamp) and t == virtual.now()
        inner, _ = run.read(net.a.next)  # a connected port shows an inner node
        assert torch.equal(inner, net.a.x)

    def test_gives_zeros_before_the_first_send(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        out, t = run.read(net.next)
        assert torch.equal(out, torch.zeros(2, dtype=F64))
        assert t == Timestamp(ns=0)

    def test_rejects_a_port_that_sends_nothing(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(ValueError, match="sends no messages"):
            run.read(net.prev)

    def test_rejects_a_port_outside_the_run(self, start_manual: StartManual):
        run = start_manual(Chain())
        with pytest.raises(ValueError, match="is not a port of a node in this run$"):
            run.read(Chain().next)


class TestSubscribe:
    def test_gives_the_latest_value_and_ends_with_the_network(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        step(run)
        values = run.subscribe(net.next)
        # First the value at the time of the call, without waiting for a new one.
        out, t = next(values)
        assert torch.equal(out, net.b.x) and t == virtual.now()
        for _ in range(3):
            step(run)
        # The first two values were skipped: only the latest is given.
        out, t = next(values)
        assert torch.equal(out, net.b.x) and t == virtual.now()
        step(run)
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
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        step(run)
        run.shutdown()
        assert [t for _, t in run.subscribe(net.next)] == [virtual.now()]

    def test_follows_a_free_running_network(self, start: Start):
        net = Chain()
        run = start(net, inputs={net.prev: u(1)})
        times: list[Timestamp] = []
        stopper = threading.Thread(target=run.shutdown)
        for _, t in run.subscribe(net.next):
            times.append(t)
            if len(times) == 5:
                stopper.start()
        stopper.join()
        assert times == sorted(times)
        assert len(times) >= 5

    def test_raises_after_the_timeout_and_the_network_keeps_running(
        self, step: Step, start_manual: StartManual
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
        step(run)
        assert torch.equal(run.read(net.next)[0], net.b.x)

    def test_the_timeout_counts_the_whole_iteration(
        self, step: Step, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        values = run.subscribe(net.next, timeout=0.05)
        next(values)
        time.sleep(0.1)  # the body of a loop that took longer than the timeout
        step(run)  # a newer message is there
        with pytest.raises(TimeoutError):
            next(values)

    def test_yields_what_arrives_within_the_timeout(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        values = run.subscribe(net.next, timeout=60.0)
        next(values)
        step(run)
        assert next(values)[1] == virtual.now()
        run.shutdown()
        with pytest.raises(StopIteration):
            next(values)


class TestWaitFor:
    def test_returns_the_first_message_that_meets_the_condition(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        seen: list[torch.Tensor] = []

        def condition(out: torch.Tensor) -> bool:
            seen.append(out)
            if len(seen) < 3:
                step(run)  # a newer message for the next check
                return False
            return True

        out, t = run.wait_for(net.next, condition, timeout=60.0)
        # The latest message first (zeros), then each newer one.
        assert torch.equal(seen[0], torch.zeros(2, dtype=F64))
        assert len(seen) == 3
        assert torch.equal(out, net.b.x) and t == virtual.now()

    def test_names_itself_when_it_times_out(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(
            TimeoutError,
            match=r"^no message of the port 'b\.next' met the condition of wait_for "
            r"within 0\.05 s",
        ):
            run.wait_for(net.next, lambda _: False, timeout=0.05)

    def test_raises_when_the_run_shuts_down_first(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net)
        stopper = threading.Timer(0.05, run.shutdown)
        stopper.start()
        with pytest.raises(RuntimeError, match=r"shut down before the port 'b\.next'"):
            run.wait_for(net.next, lambda _: False)
        stopper.join()


class TestStateDict:
    def test_after_a_normal_end_is_the_state_at_the_end(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        for k in range(3):
            run.write({net.prev: u(k)})
            step(run)
        step(run, net.a)  # b's last update is one step behind
        run.write({net.prev: u(3)})
        running = run.state_dict()
        virtual.advance(0.3)  # the input stays presented for 0.3 s more
        virtual.set_time_scale(2.0)
        asked_at = virtual.now().ns
        run.shutdown()
        virtual.advance(100.0)  # time goes on after the end
        virtual.set_time_scale(3.0)
        ended = run.state_dict()
        # Nothing changed after the last update: the same tree as just before the
        # end, at the time of the shutdown (the time after the end is not part of
        # the run).
        assert ended["timer"] == {"time_ns": asked_at, "scale": 2.0}
        assert ended["last_update_ns"] == running["last_update_ns"]
        assert ended["slots"].keys() == running["slots"].keys()
        for name, (value, time_ns) in running["slots"].items():
            assert torch.equal(ended["slots"][name][0], value)
            assert ended["slots"][name][1] == time_ns
        for path in ("a", "b"):
            assert torch.equal(ended["nodes"][path]["x"], running["nodes"][path]["x"])
            assert ended["nodes"][path]["updates"] == running["nodes"][path]["updates"]
        pickle.dumps(ended)

    def test_after_a_normal_end_of_a_free_running_network(self, start: Start):
        net = Chain(tau=0.01)
        run = start(net, inputs={net.prev: u(1)})
        # Free running: both nodes have updated once b sends a value other than 0.
        run.wait_for(net.next, lambda out: bool(out.any()), timeout=10.0)
        asked_after = mujou.time.now().ns
        run.shutdown()
        stopped_before = mujou.time.now().ns
        mujou.time.sleep(0.05)  # time goes on after the end
        tree = run.state_dict()
        frozen_at = tree["timer"]["time_ns"]
        # As a pause at the request would: the time of the request, and nothing
        # in the tree is later than it.
        assert asked_after <= frozen_at <= stopped_before
        for updated_at in tree["last_update_ns"].values():
            assert updated_at <= frozen_at
        for _, written_at in tree["slots"].values():
            assert written_at <= frozen_at
        assert torch.equal(tree["slots"]["b.next/out"][0], tree["nodes"]["b"]["x"])

    def test_is_taken_at_one_time(self, start: Start):
        net = Chain(tau=0.01)
        run = start(net, inputs={net.prev: u(1)})
        # Free running: both nodes have updated once b sends a value other than 0.
        run.wait_for(net.next, lambda out: bool(out.any()), timeout=10.0)
        tree = run.state_dict()
        frozen_at = tree["timer"]["time_ns"]
        assert set(tree["nodes"]) == set(tree["last_update_ns"]) == {"a", "b"}
        assert set(tree["slots"]) == {
            "a.next/out",
            "a.next/ch0",
            "b.next/out",
            "a.prev/in",
        }
        for _, written_at in tree["slots"].values():
            assert written_at <= frozen_at
        for updated_at in tree["last_update_ns"].values():
            assert 0 < updated_at <= frozen_at
        # Each node sends at the end of its update, so a node paused between
        # updates has sent its state as it is.
        assert torch.equal(tree["slots"]["b.next/out"][0], tree["nodes"]["b"]["x"])

    @pytest.mark.parametrize("ended", [False, True])
    def test_is_a_copy_that_pickles(
        self, step: Step, start_manual: StartManual, ended: bool
    ):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        step(run)
        if ended:
            run.shutdown()
        tree = run.state_dict()
        x = net.a.x.clone()
        net.a.x += 1.0  # in place, as a node updates its state
        assert torch.equal(tree["nodes"]["a"]["x"], x)
        again = pickle.loads(pickle.dumps(tree))
        assert torch.equal(again["nodes"]["a"]["x"], x)
        assert again["timer"] == tree["timer"]

    @pytest.mark.parametrize("ended", [False, True])
    def test_cannot_receive_in_the_node_state_dict(
        self, start_manual: StartManual, ended: bool
    ):
        class Receiving(Pair):
            @override
            def state_dict(self) -> object:
                try:
                    self.x.recv()
                except RuntimeError as e:
                    return str(e)
                return None

        run = start_manual(Compose(node=Receiving()))
        if ended:
            run.shutdown()
        assert "only the node that owns" in run.state_dict()["nodes"]["node"]


class TestHostMethodsFromANode:
    @pytest.mark.parametrize("ended", [False, True])
    def test_raise_in_state_dict_on_the_control_thread(
        self, start_manual: StartManual, ended: bool
    ):
        runs: list[Run] = []

        class Reading(Pair):
            @override
            def state_dict(self) -> object:
                try:
                    runs[0].read(self.out)
                except RuntimeError as e:
                    return str(e)
                return None

        runs.append(start_manual(Compose(node=Reading())))
        if ended:
            runs[0].shutdown()
        assert "host methods" in runs[0].state_dict()["nodes"]["node"]

    def test_raise(self, step: Step, start_manual: StartManual):
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
                lambda: mujou.run(Compose(node=Pair())),
            ):
                try:
                    call()
                except RuntimeError as e:
                    node.errors.append(e)

        node = Calls(call_every_host_method)
        chain = Chain()
        chain.caller = node  # pyright: ignore[reportAttributeAccessIssue]
        net = chain
        run = start_manual(chain)
        runs.append(run)
        step(run)
        assert len(node.errors) == 7
        assert all("host methods" in str(e) for e in node.errors)


class TestState:
    def test_rejects_a_tree_that_does_not_match(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        source = start_manual(Chain())
        tree = source.state_dict()
        source.shutdown()
        other = Compose(node=Pair())
        with pytest.raises(ValueError, match="do not match"):
            start_manual(other, state=tree)
        broken: list[tuple[dict[str, Any], str]] = [
            ({k: v for k, v in tree.items() if k != "timer"}, "no 'timer'"),
            ({**tree, "timer": {"time_ns": 0}}, "timer do not match"),
            ({**tree, "timer": {"time_ns": "0", "scale": 1.0}}, "int time_ns"),
            ({**tree, "last_update_ns": {"a": 0.5, "b": 0}}, "is not an int"),
        ]
        for wrong, match in broken:
            with pytest.raises(ValueError, match=match):
                start_manual(Chain(), state=wrong)

    def test_the_nodes_get_copies(self, step: Step, start_manual: StartManual):
        source = Chain()
        run = start_manual(source, inputs={source.prev: u(1)})
        step(run)
        tree = run.state_dict()
        run.shutdown()
        x = tree["nodes"]["a"]["x"].clone()
        net = Chain()
        restored = start_manual(net, timer=VirtualTimer(), state=tree)
        assert restored is not None
        net.a.x += 1.0  # in place, as a node updates its state
        assert torch.equal(tree["nodes"]["a"]["x"], x)


class TestNodeException:
    def test_stops_the_network_and_reaches_the_host(
        self, step: Step, start_manual: StartManual
    ):
        net = Chain(raise_in_b=3)
        run = start_manual(net)
        values = run.subscribe(net.next)
        step(run)
        step(run)
        with pytest.raises(NetworkError) as raised:
            step(run)
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
        net = Compose(node=node)
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

    def test_raises_what_a_node_raised(self, step: Step, start_manual: StartManual):
        net = Chain(raise_in_b=1)
        run = start_manual(net)
        with pytest.raises(NetworkError):
            step(run)
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
    @pytest.mark.parametrize("method", ["shutdown", "join", "subscribe"])
    def test_rejects_a_negative_timeout(
        self, step: Step, start_manual: StartManual, method: str
    ):
        net = Chain()
        run = start_manual(net)
        call: Callable[[float], object] = {
            "shutdown": lambda t: run.shutdown(timeout=t),
            "join": lambda t: run.join(timeout=t),
            "subscribe": lambda t: run.subscribe(net.next, timeout=t),
        }[method]
        for timeout in (-1.0, math.nan):
            with pytest.raises(ValueError, match="timeout"):
                call(timeout)
        # Nothing was asked: the network still runs and updates.
        run.write({net.prev: u(1)})
        step(run)
        assert net.a.updates == 1

    @pytest.mark.parametrize("method", ["join", "wait_for"])
    def test_raises_after_the_timeout_and_the_network_keeps_running(
        self, step: Step, start_manual: StartManual, method: str
    ):
        net = Chain()
        run = start_manual(net)
        with pytest.raises(TimeoutError):
            if method == "join":
                run.join(timeout=0.05)
            else:
                run.wait_for(net.next, lambda _: False, timeout=0.05)
        run.write({net.prev: u(1)})
        step(run)
        assert torch.equal(run.read(net.next)[0], net.b.x)


class TestManualRun:
    def test_updates_only_the_nodes_whose_time_advanced(
        self, step: Step, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        run.step()  # the time has not advanced since the start
        assert (net.a.updates, net.b.updates) == (0, 0)
        step(run, net.a)
        run.step()  # only b has not updated since then
        assert (net.a.updates, net.b.updates) == (1, 1)

    def test_rejects_a_node_of_another_run(self, start_manual: StartManual):
        run = start_manual(Chain())
        with pytest.raises(ValueError, match="not a node of this run"):
            run.step(Leaky())

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
                timer.advance(H)  # the clock moves only here
                run.step()
            out, t = run.read(net.next)
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
        timer = Timer()
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
            mujou.time.sleep(presented)
            out, _ = run.read(net.next)
            # After a step of the input by d, the error of two leaky stages in a
            # row is d (1 + t/tau) exp(-t/tau). Allow the nodes to have run for
            # only a third of the time (scheduling under load): t = 10 tau.
            d = (u(k) - before).abs().max().item()
            bound = d * (1 + 10) * math.exp(-10)
            assert (out - u(k)).abs().max().item() <= bound
            before = u(k)

    @pytest.mark.parametrize("composed", [False, True])
    @pytest.mark.parametrize("b_first", [False, True])
    def test_manual_steps_match_the_discrete_algorithm(
        self,
        step: Step,
        start_manual: StartManual,
        b_first: bool,
        composed: bool,
    ):
        a, b = Leaky(), Leaky()
        if composed:
            # The same two nodes as two networks: a network is no boundary of the
            # asynchrony, and both run on the one timer of the run.
            first, second = Compose(node=a), Compose(node=b)
            a.connect(b)
            run = start_manual(first, second)
            assert set(run.state_dict()["last_update_ns"]) == {"0.node", "1.node"}
        else:
            chain = Chain()
            a, b = chain.a, chain.b
            run = start_manual(chain)
        order = (b, a) if b_first else (a, b)
        xa = xb = sent_a = torch.zeros(2, dtype=F64)
        for k in range(20):
            run.write({a.prev: u(k)})
            step(run, *order)
            if b_first:  # Jacobi: b reads a's value of the previous step
                xb = euler(xb, sent_a)
                xa = sent_a = euler(xa, u(k))
            else:  # Gauss-Seidel: b reads a's value of this step
                xa = sent_a = euler(xa, u(k))
                xb = euler(xb, sent_a)
        assert torch.equal(a.x, xa)
        assert torch.equal(b.x, xb)
        assert torch.equal(run.read(b.next)[0], xb)

    @pytest.mark.parametrize("b_first", [False, True])
    def test_a_restored_network_continues_the_same(
        self, step: Step, start_manual: StartManual, b_first: bool
    ):
        k, m = 5, 7
        net = Chain()
        run = start_manual(net)
        order = (net.b, net.a) if b_first else (net.a, net.b)
        for i in range(k):
            run.write({net.prev: u(i)})
            step(run, *order)
        step(run, net.a)  # b's last update is now one step behind
        run.write({net.prev: u(k)})  # the input slot is part of the state too
        tree = pickle.loads(pickle.dumps(run.state_dict()))
        for i in range(k, k + m):
            step(run, *order)
            run.write({net.prev: u(i + 1)})
        expected = (net.a.x.clone(), net.b.x.clone(), run.read(net.next))
        run.shutdown()

        restored = Chain()
        again_run = start_manual(restored, state=tree)
        # Right after the restore the state is the tree, including where each dt
        # starts and every channel slot (which Jacobi order reads first).
        again = again_run.state_dict()
        assert again["last_update_ns"] == tree["last_update_ns"]
        assert again["timer"] == tree["timer"]
        for name, (value, time_ns) in tree["slots"].items():
            assert torch.equal(again["slots"][name][0], value)
            assert again["slots"][name][1] == time_ns
        restored_order = (
            (restored.b, restored.a) if b_first else (restored.a, restored.b)
        )
        for i in range(k, k + m):
            step(again_run, *restored_order)
            again_run.write({restored.prev: u(i + 1)})
        assert torch.equal(restored.a.x, expected[0])
        assert torch.equal(restored.b.x, expected[1])
        out, t = again_run.read(restored.next)
        assert torch.equal(out, expected[2][0]) and t == expected[2][1]
        # The nodes got copies: their updates in place did not reach the tree.
        assert not torch.equal(tree["nodes"]["a"]["x"], restored.a.x)

    @pytest.mark.parametrize("b_first", [False, True])
    def test_a_network_restored_after_the_end_continues_the_same(
        self,
        step: Step,
        virtual: VirtualTimer,
        start_manual: StartManual,
        b_first: bool,
    ):
        k, m, gap = 5, 7, 0.3

        def present(run: ManualRun, net: Chain, steps: range) -> None:
            order = (net.b, net.a) if b_first else (net.a, net.b)
            for i in steps:
                run.write({net.prev: u(i)})
                step(run, *order)
            step(run, net.a)  # b's last update is one step behind
            run.write({net.prev: u(steps.stop)})
            virtual.advance(gap)  # the input stays presented a while

        # Saved after the end.
        ended = Chain()
        run = start_manual(ended)
        present(run, ended, range(k))
        run.shutdown()
        virtual.advance(100.0)  # time goes on after the end
        tree = pickle.loads(pickle.dumps(run.state_dict()))

        # The same run without stopping.
        through = Chain()
        run = start_manual(through)
        present(run, through, range(k))
        present(run, through, range(k + 1, k + 1 + m))
        expected = (through.a.x.clone(), through.b.x.clone())
        run.shutdown()

        restored = Chain()
        run = start_manual(restored, state=tree)
        present(run, restored, range(k + 1, k + 1 + m))
        assert torch.equal(restored.a.x, expected[0])
        assert torch.equal(restored.b.x, expected[1])


class TestInterpreterExit:
    """Run a script in a new interpreter and see what happens at its exit."""

    PRELUDE = """
import pathlib, sys, threading, time
import torch
import mujou
from mujou import Network, NextPort, Node, Port, TensorSpec

class Slow(Node):
    def __init__(self) -> None:
        self.out = NextPort(send=TensorSpec((2,)))

    def on_update(self, dt: float) -> None:
        self.out.send(torch.ones(2))

    def on_stop(self) -> None:
        time.sleep(0.3)  # an actuator taking its time to stop
        pathlib.Path(sys.argv[1]).write_text("stopped")

class Net(Network):
    def __init__(self) -> None:
        self.node = Slow()
        self.out = self.node.out
"""

    def run_script(self, body: str, marker: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", self.PRELUDE + body, str(marker)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    @pytest.mark.slow
    def test_waits_for_on_stop_of_a_dropped_network(self, tmp_path: Path):
        marker = tmp_path / "marker"
        body = "def main() -> None:\n    mujou.run(Net())\nmain()\n"
        done = self.run_script(body, marker)
        assert done.returncode == 0, done.stderr
        assert marker.read_text() == "stopped"

    @pytest.mark.slow
    def test_ends_a_subscribing_thread(self, tmp_path: Path):
        marker = tmp_path / "marker"
        body = (
            "net = Net()\n"
            "run = mujou.run(net)\n"
            "threading.Thread(target=lambda: [_ for _ in run.subscribe(net.out)])"
            ".start()\n"
        )
        done = self.run_script(body, marker)
        assert done.returncode == 0, done.stderr
        assert marker.read_text() == "stopped"


# Large enough that the GPU is still working when the CPU moves on: a missing wait
# for a stream shows as a value in the middle of an update.
BIG = 1 << 26  # 256 MB of float32
ADDS = 20  # in-place updates of the whole tensor per on_update
CUDA0 = torch.device("cuda", 0)


class DeviceLeaky(Node):
    """``Leaky`` with ``tau=1`` on ``device``."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        self.prev = PrevPort(recv=vec())
        self.next = NextPort(send=vec())

    @override
    def on_start(self) -> None:
        self.x = torch.zeros(2, dtype=F64, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        self.x += min(dt, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)


class DeviceChain(Network):
    def __init__(self, device: torch.device) -> None:
        self.a = DeviceLeaky(device)
        self.b = DeviceLeaky(device)
        self.a.connect(self.b)
        self.prev = self.a.prev
        self.next = self.b.next


class CudaSource(Node):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.next = NextPort(send=vec())

    @override
    def on_update(self, dt: float) -> None:
        self.next.send(torch.tensor([1.0, 2.0], dtype=F64, device=self.device))


class CudaSink(Node):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.prev = PrevPort(recv=vec())
        self.received: torch.Tensor | None = None

    @override
    def on_update(self, dt: float) -> None:
        self.received = self.prev.recv()


class SourceToSink(Network):
    def __init__(self, source: Node, sink: Node) -> None:
        self.source = source
        self.sink = sink
        source.connect(sink)


class Counter(Node):
    """Keeps ``BIG`` floats that all equal ``k`` between updates; each update adds
    1 to every float ``ADDS`` times, in place."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        self.first: torch.Tensor | None = None  # w as the first update finds it

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(BIG, device=self.device)
        self.k = 0

    @override
    def on_update(self, dt: float) -> None:
        if self.first is None:
            self.first = self.w.clone()
        for _ in range(ADDS):
            self.w.add_(1.0)
        self.k += ADDS

    @override
    def state_dict(self) -> object:
        return {"w": self.w, "k": self.k}

    @override
    def load_state_dict(self, state: object) -> None:
        loaded = cast(dict[str, Any], state)
        self.w.copy_(loaded["w"])
        self.k = loaded["k"]


def is_uniform(tree: dict[str, Any]) -> bool:
    """Whether every float of the captured ``Counter`` equals its ``k``."""
    state = tree["nodes"]["node"]
    w, k = state["w"], state["k"]
    return w.min().item() == k and w.max().item() == k


HOST_ADDS = 50


class QueuesOnTheHost(Node):
    """Queues ``HOST_ADDS`` updates of a large tensor on the host's stream in
    ``__init__`` and reads it in ``on_start``."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        self.w = torch.zeros(BIG, device=self.device)
        for _ in range(HOST_ADDS):
            self.w.add_(1.0)
        self.seen: float | None = None

    @override
    def on_start(self) -> None:
        self.seen = self.w.min().item()


class BusyStart(Node):
    """Leaves ``HOST_ADDS`` updates of a large tensor on its stream in
    ``on_start`` (no ``on_update`` waits for them)."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(BIG, device=self.device)
        for _ in range(HOST_ADDS):
            self.w.add_(1.0)

    @override
    def state_dict(self) -> object:
        return {"w": self.w}


class BusyStop(Node):
    """Leaves ``HOST_ADDS`` updates of a large tensor on its stream in
    ``on_stop``."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(BIG, device=self.device)

    @override
    def on_stop(self) -> None:
        for _ in range(HOST_ADDS):
            self.w.add_(1.0)

    @override
    def state_dict(self) -> object:
        return {"w": self.w}


@pytest.mark.cuda
class TestCuda:
    @pytest.mark.cuda(2)
    def test_a_message_goes_from_one_gpu_to_another(
        self, step: Step, start_manual: StartManual
    ):
        source, sink = CudaSource("cuda:0"), CudaSink("cuda:1")
        net = SourceToSink(source, sink)
        run = start_manual(net)
        step(run, source, sink)
        received = sink.received
        assert received is not None
        assert received.device == torch.device("cuda", 1)
        assert torch.equal(received.cpu(), torch.tensor([1.0, 2.0], dtype=F64))

    # The two tests below cannot run on MPS: it has no float64, and its work from
    # two threads at once (here, the host's reading while the node updates) aborts
    # the process.
    def test_manual_steps_match_the_discrete_algorithm(
        self, step: Step, start_manual: StartManual
    ):
        net = DeviceChain(CUDA0)
        run = start_manual(net)
        xa = torch.zeros(2, dtype=F64, device=CUDA0)
        xb = torch.zeros(2, dtype=F64, device=CUDA0)
        for k in range(20):
            run.write({net.prev: u(k)})
            step(run, net.a, net.b)
            xa = euler(xa, u(k).to(CUDA0))
            xb = euler(xb, xa)
        assert torch.equal(run.read(net.next)[0], xb.cpu())

    @pytest.mark.slow
    def test_state_dict_is_whole_while_a_node_updates_in_place(self, start: Start):
        run = start(Compose(node=Counter(CUDA0)))
        mujou.time.sleep(0.2)
        trees: list[dict[str, Any]] = []
        for _ in range(40):
            trees.append(run.state_dict())
            mujou.time.sleep(0.01)
        torn = [i for i, tree in enumerate(trees) if not is_uniform(tree)]
        # The copies stay as taken while the node goes on updating in place.
        mujou.time.sleep(0.2)
        changed = [i for i, tree in enumerate(trees) if not is_uniform(tree)]
        run.shutdown()
        assert torn == []
        assert changed == []
        assert trees[-1]["nodes"]["node"]["k"] > 0


class TestAccelerator:
    @pytest.mark.slow
    def test_on_start_sees_the_work_the_host_queued(
        self, start_manual: StartManual, accelerator: torch.device
    ):
        seen: list[float | None] = []
        for _ in range(5):
            # Run each node right after making it, while its work is still queued.
            node = QueuesOnTheHost(accelerator)
            start_manual(Compose(node=node)).shutdown()
            seen.append(node.seen)
        assert seen == [float(HOST_ADDS)] * 5

    @pytest.mark.slow
    def test_state_dict_waits_for_the_work_of_on_start(
        self, start_manual: StartManual, accelerator: torch.device
    ):
        captured: list[tuple[float, float]] = []
        for _ in range(5):
            run = start_manual(Compose(node=BusyStart(accelerator)))
            w = run.state_dict()["nodes"]["node"]["w"]
            run.shutdown()
            captured.append((w.min().item(), w.max().item()))
        assert captured == [(float(HOST_ADDS), float(HOST_ADDS))] * 5

    @pytest.mark.slow
    def test_state_dict_after_the_end_waits_for_the_work_of_on_stop(
        self, start_manual: StartManual, accelerator: torch.device
    ):
        captured: list[tuple[float, float]] = []
        for _ in range(5):
            run = start_manual(Compose(node=BusyStop(accelerator)))
            run.shutdown()
            w = run.state_dict()["nodes"]["node"]["w"]
            captured.append((w.min().item(), w.max().item()))
        assert captured == [(float(HOST_ADDS), float(HOST_ADDS))] * 5

    @pytest.mark.slow
    def test_a_restored_node_starts_from_the_tree(
        self,
        step: Step,
        start_manual: StartManual,
        accelerator: torch.device,
    ):
        net = Compose(node=Counter(accelerator))
        run = start_manual(net)
        for _ in range(3):
            step(run)
        tree = run.state_dict()
        run.shutdown()

        node = Counter(accelerator)
        restored = start_manual(Compose(node=node), state=tree)
        step(restored)
        assert node.first is not None
        expected = torch.full((BIG,), float(3 * ADDS), device=accelerator)
        assert torch.equal(node.first, expected)
        assert torch.equal(tree["nodes"]["node"]["w"].to(accelerator), expected)


class OnMps(Node):
    """Does nothing on MPS; records whether it started."""

    def __init__(self) -> None:
        super().__init__(device="mps")
        self.started = False

    @override
    def on_start(self) -> None:
        super().on_start()
        self.started = True


class TwoOnMps(Network):
    def __init__(self) -> None:
        self.a = OnMps()
        self.b = OnMps()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
class TestOneNodeOnMps:
    def test_refuses_two_nodes_in_one_run(self, start_manual: StartManual):
        net = TwoOnMps()
        with pytest.raises(ValueError, match=r"^MPS cannot .* but 'a', 'b' do;"):
            start_manual(net)
        assert not net.a.started and not net.b.started

    def test_refuses_a_node_while_another_run_has_one(self, start_manual: StartManual):
        first, second = OnMps(), OnMps()
        run = start_manual(first)
        with pytest.raises(ValueError, match=r"'OnMps' \(of a running run\)"):
            start_manual(second)
        assert not second.started
        run.shutdown()
        start_manual(
            Compose(node=second)
        )  # the first run has ended: it no longer counts
        assert second.started

    def test_allows_one_node_beside_nodes_on_the_cpu(self, start_manual: StartManual):
        node, net = OnMps(), Chain()
        start_manual(node, net)
        assert node.started and net.a.started


class OnCpu(Node):
    """Does nothing on the CPU; records whether it started."""

    def __init__(self) -> None:
        super().__init__()
        self.started = False

    @override
    def on_start(self) -> None:
        super().on_start()
        self.started = True


@pytest.fixture
def one_node_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count the CPU as MPS is counted, so that the one-node rule runs without
    MPS: ``Session.start`` reads the module's device types at each start."""
    monkeypatch.setattr(session_module, "_ONE_NODE_DEVICES", ("cpu",))


@pytest.mark.usefixtures("one_node_cpu")
class TestOneNodeDevice:
    def test_refuses_two_nodes_in_one_run(self, start_manual: StartManual):
        a, b = OnCpu(), OnCpu()
        with pytest.raises(ValueError, match=r"^CPU cannot .* but 'a', 'b' do;"):
            start_manual(Compose(a=a, b=b))
        assert not a.started and not b.started

    def test_refuses_one_node_in_each_of_two_runs(self, start_manual: StartManual):
        first, second = OnCpu(), OnCpu()
        run = start_manual(first)
        with pytest.raises(ValueError, match=r"'OnCpu' \(of a running run\)"):
            start_manual(second)
        assert not second.started
        run.shutdown()
        start_manual(second)  # the first run has ended: it no longer counts
        assert second.started

    def test_starts_one_of_two_runs_started_at_once(self):
        # Without the lock around the count and the registration, both runs pass
        # the count in most rounds.
        for _ in range(50):
            barrier = threading.Barrier(2)
            outcomes: list[Run | ValueError] = []

            def go(barrier: threading.Barrier = barrier) -> None:
                node = OnCpu()
                barrier.wait()
                try:
                    outcomes.append(mujou.run(node))
                except ValueError as e:
                    outcomes.append(e)

            threads = [threading.Thread(target=go) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            runs = [outcome for outcome in outcomes if isinstance(outcome, Run)]
            for run in runs:
                run.shutdown()
            assert len(runs) == 1


class Hooks(Node):
    """Appends ``(name, hook)`` to the shared ``log`` for each hook it is called
    in; raises in ``on_update`` once ``trigger`` is set."""

    def __init__(self, name: str, log: list[tuple[str, str]]) -> None:
        super().__init__()
        self.name = name
        self.log = log
        self.trigger = threading.Event()
        self.error: Exception | None = None

    def hooks(self) -> list[str]:
        return [hook for name, hook in self.log if name == self.name]

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.log.append((self.name, "on_update"))
        if self.trigger.is_set():
            self.error = ValueError(f"boom in {self.name}")
            raise self.error

    @override
    def on_pause(self) -> None:
        super().on_pause()
        self.log.append((self.name, "on_pause"))

    @override
    def on_resume(self) -> None:
        super().on_resume()
        self.log.append((self.name, "on_resume"))

    @override
    def state_dict(self) -> object:
        self.log.append((self.name, "state_dict"))
        return super().state_dict()

    @override
    def on_exception(self) -> None:
        super().on_exception()
        self.log.append((self.name, "on_exception"))

    @override
    def on_stop(self) -> None:
        super().on_stop()
        self.log.append((self.name, "on_stop"))


class ThreeHooks(Network):
    def __init__(self, log: list[tuple[str, str]]) -> None:
        self.a = Hooks("a", log)
        self.b = Hooks("b", log)
        self.c = Hooks("c", log)

    def nodes(self) -> list[Hooks]:
        return [self.a, self.b, self.c]


class TestPauseHooks:
    def test_state_dict_pauses_every_node_takes_the_state_then_resumes_each(
        self, step: Step, start_manual: StartManual
    ):
        log: list[tuple[str, str]] = []
        net = ThreeHooks(log)
        run = start_manual(net)
        step(run)
        log.clear()

        run.state_dict()
        step(run)  # on_resume comes before the next on_update

        for node in net.nodes():
            assert node.hooks() == ["on_pause", "state_dict", "on_resume", "on_update"]
        hooks = [hook for _, hook in log]
        last_pause = max(i for i, h in enumerate(hooks) if h == "on_pause")
        states = [i for i, h in enumerate(hooks) if h == "state_dict"]
        first_resume = min(i for i, h in enumerate(hooks) if h == "on_resume")
        # The state is taken at one time: every node is paused before any state,
        # and none resumes before every state is taken.
        assert last_pause < min(states)
        assert max(states) < first_resume


class TestExceptionHooks:
    def test_every_node_gets_on_exception_then_on_stop_and_the_host_the_error(
        self, step: Step, start_manual: StartManual
    ):
        log: list[tuple[str, str]] = []
        net = ThreeHooks(log)
        run = start_manual(net)
        step(run)
        net.b.trigger.set()

        with pytest.raises(NetworkError) as raised:
            step(run)
        with pytest.raises(NetworkError):
            run.join()  # every on_stop has returned

        assert raised.value.__cause__ is net.b.error
        for node in net.nodes():
            hooks = node.hooks()
            assert hooks[-2:] == ["on_exception", "on_stop"]
            assert hooks.count("on_exception") == 1
            assert hooks.count("on_stop") == 1


class Blocks(Node):
    """Blocks in its first ``on_update`` until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        if not self.release.is_set():
            self.entered.set()
            self.release.wait(10.0)


class TestPauseTimeout:
    def test_state_dict_raises_when_a_node_does_not_pause_and_the_run_goes_on(
        self, started: list[Run]
    ):
        node = Blocks()
        run = mujou.run(node, pause_timeout=0.05)
        started.append(run)
        try:
            assert node.entered.wait(10.0)
            with pytest.raises(mujou.PauseTimeoutError):
                run.state_dict()
        finally:
            node.release.set()

        assert "nodes" in run.state_dict()  # the run kept running
        run.shutdown()
        run.state_dict()  # a normal end
