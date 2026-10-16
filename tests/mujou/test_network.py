import math
import pickle
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import pytest
import torch

import mujou
from mujou import NetworkError
from mujou._core import GlobalTimer
from mujou._network import Graph, Network, graph
from mujou._node import Node
from mujou._port import NextPort, Port, PrevPort
from mujou._spec import TensorSpec

if TYPE_CHECKING:
    from tests.mujou.conftest import RunNetwork


def h(n: int = 3) -> TensorSpec:
    return TensorSpec((2, n))


class Hidden(Node):
    def __init__(self) -> None:
        self.prev = PrevPort(send=h(), recv=h())
        self.next = NextPort(send=h(), recv=h())


class Source(Node):
    def __init__(self) -> None:
        self.next = NextPort(send=h(), recv=h())


class Sink(Node):
    def __init__(self) -> None:
        self.prev = PrevPort(send=h(), recv=h())
        self.teacher = Port(recv=h(5))


class Stack(Network):
    def __init__(self) -> None:
        self.l1 = Hidden()
        self.l2 = Hidden()
        self.l1.connect(self.l2)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(Network):
    def __init__(self) -> None:
        self.body = Stack()
        self.head = Sink()
        self.body.connect(self.head)
        self.prev = self.body.prev
        self.teacher = self.head.teacher


def built_graph(net: Network) -> Graph:
    net.build()
    return graph(net)


class TestComponentConnect:
    def test_joins_next_to_prev(self):
        a, b = Hidden(), Hidden()
        ch = a.connect(b)
        assert ch.ends == (a.next, b.prev)
        assert a.prev.channels == () and b.next.channels == ()

    def test_reverse_is_a_different_connection(self):
        a, b = Hidden(), Hidden()
        ab, ba = a.connect(b), b.connect(a)
        assert ab.ends == (a.next, b.prev)
        assert ba.ends == (b.next, a.prev)

    def test_uses_the_ports_a_nested_network_exposes(self):
        stack, sink = Stack(), Sink()
        ch = stack.connect(sink)
        assert ch.ends == (stack.l2.next, sink.prev)
        source = Source()
        ch = source.connect(stack)
        assert ch.ends == (source.next, stack.l1.prev)

    def test_missing_side(self):
        with pytest.raises(ValueError, match="NextPort"):
            Sink().connect(Hidden())
        with pytest.raises(ValueError, match="PrevPort"):
            Hidden().connect(Source())

    def test_the_same_port_under_two_names_is_one_side(self):
        class Aliased(Network):
            def __init__(self) -> None:
                self.node = Hidden()
                self.next = self.node.next
                self.out = self.node.next

        net, sink = Aliased(), Sink()
        ch = net.connect(sink)
        assert ch.ends == (net.node.next, sink.prev)

    def test_two_sides(self):
        class TwoNext(Node):
            def __init__(self) -> None:
                self.a = NextPort(send=h(), recv=h())
                self.b = NextPort(send=h(), recv=h())

        with pytest.raises(ValueError, match="found 2"):
            TwoNext().connect(Hidden())

    def test_branches_from_a_next_port(self):
        a, b, c = Hidden(), Hidden(), Hidden()
        ab, ac = a.connect(b), a.connect(c)
        assert a.next.channels == (ab, ac)


class TestBuild:
    def test_paths_are_depth_first_in_definition_order(self):
        net = Classifier()
        graph = built_graph(net)
        assert [path for path, _ in graph.nodes] == ["body.l1", "body.l2", "head"]
        assert [node for _, node in graph.nodes] == [
            net.body.l1,
            net.body.l2,
            net.head,
        ]
        assert graph.port_path(net.body.l1.next) == "body.l1.next"
        assert graph.port_path(net.prev) == "body.l1.prev"
        assert graph.port_path(net.teacher) == "head.teacher"
        assert set(graph.ports) == {
            net.body.l1.prev,
            net.body.l1.next,
            net.body.l2.prev,
            net.body.l2.next,
            net.head.prev,
            net.head.teacher,
        }

    def test_port_path_of_a_foreign_port(self):
        graph = built_graph(Classifier())
        with pytest.raises(ValueError, match="not a port of this network"):
            graph.port_path(Port(send=h()))

    def test_graph_before_build(self):
        with pytest.raises(RuntimeError, match="not built"):
            graph(Classifier())

    def test_unconnected_ports_are_allowed(self):
        class Single(Network):
            def __init__(self) -> None:
                self.node = Hidden()

        Single().build()

    def test_rejects_connect_after_build(self):
        net = Classifier()
        net.build()
        with pytest.raises(RuntimeError, match="already built"):
            net.body.next.connect(Sink().prev)
        with pytest.raises(RuntimeError, match="already built"):
            Source().connect(net)
        assert len(net.body.next.channels) == 1

    def test_rejects_a_second_build(self):
        net = Classifier()
        net.build()
        with pytest.raises(RuntimeError, match="already built"):
            net.build()

    def test_rejects_a_tree_that_contains_a_built_network(self):
        class Outer(Network):
            def __init__(self, inner: Network) -> None:
                self.inner = inner

        inner = Stack()
        inner.build()
        with pytest.raises(RuntimeError, match="already built"):
            Outer(inner).build()

    def test_rejects_the_same_node_at_two_paths(self):
        class Twice(Network):
            def __init__(self) -> None:
                self.a = Hidden()
                self.b = self.a

        with pytest.raises(ValueError, match="'a' and 'b'"):
            Twice().build()

    def test_rejects_a_connection_out_of_the_tree(self):
        net = Classifier()
        Source().connect(net)
        with pytest.raises(
            ValueError, match="'body.l1.prev' is connected to a port outside"
        ):
            net.build()

    def test_rejects_a_node_holding_a_component(self):
        class Holder(Node):
            def __init__(self) -> None:
                self.inner = Hidden()

        class Net(Network):
            def __init__(self) -> None:
                self.holder = Holder()

        with pytest.raises(ValueError, match="node 'holder' holds a Hidden"):
            Net().build()

    def test_rejects_a_port_owned_by_two_nodes(self):
        class Borrower(Node):
            def __init__(self, port: Port[Any, Any, Any]) -> None:
                self.port = port

        class Net(Network):
            def __init__(self) -> None:
                self.a = Hidden()
                self.b = Borrower(self.a.next)

        with pytest.raises(ValueError, match="'b.port' is already owned as 'a.next'"):
            Net().build()

    def test_rejects_exposing_a_port_outside_the_tree(self):
        class Net(Network):
            def __init__(self) -> None:
                self.a = Hidden()
                self.out = Port(send=h())

        with pytest.raises(ValueError, match="exposed at 'out'"):
            Net().build()

    def test_rejects_exposing_a_port_of_a_sibling(self):
        class Inner(Network):
            def __init__(self, node: Hidden) -> None:
                self.own = Hidden()
                self.prev = node.prev

        class Outer(Network):
            def __init__(self) -> None:
                self.node = Hidden()
                self.inner = Inner(self.node)

        with pytest.raises(ValueError, match="exposed at 'inner.prev'"):
            Outer().build()

    def test_rejects_ports_declared_on_the_class(self):
        class ClassPort(Node):
            port = Port(send=h())

        class Net(Network):
            def __init__(self) -> None:
                self.node = ClassPort()

        with pytest.raises(ValueError, match="'port' on the class ClassPort"):
            Net().build()

    def test_rejects_two_prev_ports_on_a_network(self):
        class Net(Network):
            def __init__(self) -> None:
                self.a, self.b = Hidden(), Hidden()
                self.prev_a = self.a.prev
                self.prev_b = self.b.prev

        with pytest.raises(ValueError, match="2 PrevPort"):
            Net().build()

    def test_rejects_two_next_ports_on_a_node(self):
        class TwoNext(Node):
            def __init__(self) -> None:
                self.a = NextPort(send=h())
                self.b = NextPort(send=h())

        class Net(Network):
            def __init__(self) -> None:
                self.node = TwoNext()

        with pytest.raises(ValueError, match="'node' has 2 NextPort"):
            Net().build()

    def test_rejects_a_network_without_nodes(self):
        class Empty(Network):
            def __init__(self) -> None:
                self.inner = Network()

        with pytest.raises(ValueError, match="no node"):
            Empty().build()


# --- running networks

H_NS = 1_000_000
H = H_NS / 1e9  # The step of the virtual clock, in seconds.
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
        self.seen: list[tuple[tuple[torch.Tensor, int], tuple[torch.Tensor, int]]] = []

    @override
    def on_update(self, dt: float) -> None:
        self.seen.append((self.x.recv_with_time(), self.y.recv_with_time()))


class Solo(Network):
    def __init__(self, node: Node) -> None:
        self.node = node


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


def step(net: Network, timer: GlobalTimer, *order: Node) -> None:
    """Advance the virtual clock by ``H`` and update every node once."""
    timer.advance(H)
    net._step(order or None)  # pyright: ignore[reportPrivateUsage]


def euler(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The reference of one ``Leaky`` update with ``tau=1`` and ``dt=H``."""
    return x + min(H / 1.0, 1.0) * (target - x)


class TestRun:
    def test_builds_runs_and_returns_the_network(self, run_network: RunNetwork):
        net = Chain()
        assert run_network(net) is net
        assert net.a.started and net.b.started  # run returns after on_start
        with pytest.raises(RuntimeError, match="already built"):
            net.build()

    def test_runs_a_built_network(self, run_network: RunNetwork):
        net = Chain()
        net.build()
        assert run_network(net) is net

    def test_runs_once(self, run_network: RunNetwork):
        net = run_network(Chain())
        net.shutdown()
        with pytest.raises(RuntimeError, match="already run"):
            net.run()

    def test_needs_a_build(self):
        with pytest.raises(RuntimeError, match="not built"):
            Chain().run()

    def test_raises_what_on_start_raised(self, run_network: RunNetwork):
        net = Chain(raise_in_b=0)
        with pytest.raises(NetworkError) as raised:
            run_network(net)
        assert raised.value.__cause__ is net.b.error
        assert net.a.updates == 0 and net.b.updates == 0


class TestWrite:
    def test_before_run_gives_the_initial_value(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = Chain()
        net.build()
        net.write({net.prev: u(1)})
        run_network(net, _manual=True)
        step(net, virtual_timer)
        assert torch.equal(net.a.x, euler(torch.zeros(2, dtype=F64), u(1)))

    def test_switches_every_port_at_once(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        # Manual mode: the write and the update do not overlap, so the node
        # reads both values of one write.
        node = Pair()
        net = run_network(Solo(node), _manual=True)
        virtual_timer.advance(H)
        net.write({node.x: u(1), node.y: u(2)})
        written_at = virtual_timer.time_ns()
        step(net, virtual_timer)
        ((x, tx), (y, ty)) = node.seen[0]
        assert torch.equal(x, u(1)) and torch.equal(y, u(2))
        assert tx == ty == written_at

    def test_does_not_wait_for_a_run_in_its_start(self, run_network: RunNetwork):
        # on_start waits for the host's first input, written by another thread
        # while run() is still starting the nodes.
        class WaitsForInput(Pair):
            got: torch.Tensor | None = None

            @override
            def on_start(self) -> None:
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    value = self.x.recv()
                    if value.any():
                        self.got = value
                        return
                    time.sleep(0.005)

        node = WaitsForInput()
        net = Solo(node)
        net.build()
        returned_after: list[float] = []

        def write_later() -> None:
            time.sleep(0.2)
            started = time.monotonic()
            net.write({node.x: u(1)})
            returned_after.append(time.monotonic() - started)

        writer = threading.Thread(target=write_later)
        writer.start()
        run_network(net, _manual=True)
        writer.join()
        assert node.got is not None and torch.equal(node.got, u(1))
        assert returned_after[0] < 1.0

    def test_after_load_state_dict_takes_precedence(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        first = Pair()
        net = run_network(Solo(first), _manual=True)
        net.write({first.x: u(1)})
        step(net, virtual_timer)
        tree = net.state_dict()
        written_at = tree["slots"]["node.x/in"][1]
        frozen_at = tree["timer"]["time_ns"]
        net.shutdown()
        virtual_timer.advance(100.0)  # the process's time is far from the tree's

        kept, before, after = Pair(), Pair(), Pair()
        for node in (kept, before, after):
            restored = Solo(node)
            restored.build()
            if node is before:
                restored.write({node.x: u(2)})  # replaced by the tree
            restored.load_state_dict(tree)
            if node is after:
                restored.write({node.x: u(3)})
            run_network(restored, _manual=True)
            step(restored, virtual_timer)
            restored.shutdown()
        assert torch.equal(kept.seen[0][0][0], u(1))
        assert kept.seen[0][0][1] == written_at
        assert torch.equal(before.seen[0][0][0], u(1))
        assert torch.equal(after.seen[0][0][0], u(3))
        # Written for the restored run, so stamped with the tree's time.
        assert after.seen[0][0][1] == frozen_at

    def test_rejects_ports_it_cannot_write(self):
        net = Chain()
        pair = Pair()
        solo = Solo(pair)
        for n in (net, solo):
            n.build()
        cases = [
            (net, net.b.prev, "is connected"),
            (solo, pair.many, "many peers"),
            (solo, pair.out, "receives no messages"),
            (net, pair.x, "not a port of this network"),
        ]
        for n, port, match in cases:
            with pytest.raises(ValueError, match=match):
                n.write({port: u(0)})

    def test_rejects_values_that_do_not_match_the_spec(self):
        net = Chain()
        net.build()
        with pytest.raises(ValueError, match="shape"):
            net.write({net.prev: torch.zeros(3, dtype=F64)})
        with pytest.raises(TypeError, match="dtype"):
            net.write({net.prev: torch.zeros(2)})

    def test_needs_a_build(self):
        net = Chain()
        with pytest.raises(RuntimeError, match="not built"):
            net.write({net.prev: u(0)})

    def test_raises_after_a_normal_end(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        net.shutdown()
        with pytest.raises(RuntimeError, match="stopped"):
            net.write({net.prev: u(2)})
        # Nothing was written: the tree keeps the input as it was at the end.
        assert torch.equal(net.state_dict()["slots"]["a.prev/in"][0], u(1))


class TestRead:
    def test_gives_the_latest_output_and_its_time_in_seconds(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        out, t = net.read(net.next)
        assert torch.equal(out, net.b.x)
        assert t == virtual_timer.time_ns() / 1e9
        inner, _ = net.read(net.a.next)  # a connected port shows an inner node
        assert torch.equal(inner, net.a.x)

    def test_gives_zeros_before_the_first_send(self, run_network: RunNetwork):
        net = Chain()
        net.build()
        assert torch.equal(net.read(net.next)[0], torch.zeros(2, dtype=F64))

    def test_rejects_a_port_that_sends_nothing(self):
        net = Chain()
        net.build()
        with pytest.raises(ValueError, match="sends no messages"):
            net.read(net.prev)


class TestSubscribe:
    def test_gives_the_latest_value_and_ends_with_the_network(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        values = net.subscribe(net.next)
        # First the value at the time of the call, without waiting for a new one.
        out, t = next(values)
        assert torch.equal(out, net.b.x) and t == virtual_timer.time_ns() / 1e9
        for _ in range(3):
            step(net, virtual_timer)
        # The first two values were skipped: only the latest is given.
        out, t = next(values)
        assert torch.equal(out, net.b.x) and t == virtual_timer.time_ns() / 1e9
        step(net, virtual_timer)
        assert next(values)[1] == virtual_timer.time_ns() / 1e9
        net.shutdown()
        with pytest.raises(StopIteration):
            next(values)

    def test_gives_the_value_before_the_first_send(self, run_network: RunNetwork):
        net = run_network(Chain(), _manual=True)
        out, _ = next(net.subscribe(net.next))
        assert torch.equal(out, torch.zeros(2, dtype=F64))

    def test_gives_the_last_value_after_a_normal_end(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        net.shutdown()
        assert [t for _, t in net.subscribe(net.next)] == [
            virtual_timer.time_ns() / 1e9
        ]

    def test_follows_a_free_running_network(self, run_network: RunNetwork):
        net = run_network(Chain())
        net.write({net.prev: u(1)})
        times: list[float] = []
        stopper = threading.Thread(target=net.shutdown)
        for _, t in net.subscribe(net.next):
            times.append(t)
            if len(times) == 5:
                stopper.start()
        stopper.join()
        assert times == sorted(times)
        assert len(times) >= 5


class TestStateDict:
    def test_needs_a_run(self, run_network: RunNetwork):
        net = Chain()
        net.build()
        with pytest.raises(RuntimeError, match="not running"):
            net.state_dict()
        tree = Chain()
        tree.build()
        tree.load_state_dict(run_network(Chain(), _manual=True).state_dict())
        with pytest.raises(RuntimeError, match="not running"):
            tree.state_dict()  # restored, but not run yet

    def test_after_a_normal_end_is_the_state_at_the_end(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        for k in range(3):
            net.write({net.prev: u(k)})
            step(net, virtual_timer)
        step(net, virtual_timer, net.a)  # b's last update is one step behind
        net.write({net.prev: u(3)})
        running = net.state_dict()
        net.shutdown()
        virtual_timer.advance(100.0)  # time goes on after the end
        ended = net.state_dict()
        # Nothing changed after the last update: the same tree as just before the
        # end, at the time of the latest update or write (the time after the end
        # is not part of the run).
        assert ended["timer"] == running["timer"]
        assert ended["last_update_ns"] == running["last_update_ns"]
        assert ended["slots"].keys() == running["slots"].keys()
        for name, (value, time_ns) in running["slots"].items():
            assert torch.equal(ended["slots"][name][0], value)
            assert ended["slots"][name][1] == time_ns
        for path in ("a", "b"):
            assert torch.equal(ended["nodes"][path]["x"], running["nodes"][path]["x"])
            assert ended["nodes"][path]["updates"] == running["nodes"][path]["updates"]
        pickle.dumps(ended)

    def test_after_a_normal_end_is_a_copy(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        net.shutdown()
        tree = net.state_dict()
        x = net.a.x.clone()
        net.a.x += 1.0
        assert torch.equal(tree["nodes"]["a"]["x"], x)

    def test_after_a_normal_end_of_a_free_running_network(
        self, run_network: RunNetwork
    ):
        net = run_network(Chain(tau=0.01))
        net.write({net.prev: u(1)})
        mujou.time.sleep(0.05)
        asked_after = mujou.time.time_ns()
        net.shutdown()
        stopped_before = mujou.time.time_ns()
        mujou.time.sleep(0.05)  # time goes on after the end
        tree = net.state_dict()
        frozen_at = tree["timer"]["time_ns"]
        # As a pause at the request would: the time of the request, and nothing
        # in the tree is later than it.
        assert asked_after <= frozen_at <= stopped_before
        for updated_at in tree["last_update_ns"].values():
            assert updated_at <= frozen_at
        for _, written_at in tree["slots"].values():
            assert written_at <= frozen_at
        assert torch.equal(tree["slots"]["b.next/out"][0], tree["nodes"]["b"]["x"])

    def test_after_a_normal_end_is_taken_at_the_time_of_the_shutdown(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        virtual_timer.advance(0.3)  # the input stays presented for 0.3 s more
        virtual_timer.set_time_scale(2.0)
        asked_at = virtual_timer.time_ns()
        net.shutdown()
        virtual_timer.advance(100.0)
        virtual_timer.set_time_scale(3.0)
        tree = net.state_dict()
        assert tree["timer"] == {"time_ns": asked_at, "scale": 2.0}

    def test_after_a_normal_end_cannot_call_the_network_in_the_node_state_dict(
        self, run_network: RunNetwork
    ):
        networks: list[Network] = []

        class Reading(Pair):
            @override
            def state_dict(self) -> object:
                try:
                    networks[0].read(self.out)
                except RuntimeError as e:
                    return str(e)
                return None

        networks.append(run_network(Solo(Reading()), _manual=True))
        networks[0].shutdown()
        assert "host methods" in networks[0].state_dict()["nodes"]["node"]

    def test_after_a_normal_end_cannot_receive_in_the_node_state_dict(
        self, run_network: RunNetwork
    ):
        class Receiving(Pair):
            @override
            def state_dict(self) -> object:
                try:
                    self.x.recv()
                except RuntimeError as e:
                    return str(e)
                return None

        net = run_network(Solo(Receiving()), _manual=True)
        net.shutdown()
        assert "only the node that owns" in net.state_dict()["nodes"]["node"]

    def test_is_taken_at_one_time(self, run_network: RunNetwork):
        net = run_network(Chain(tau=0.01))
        net.write({net.prev: u(1)})
        mujou.time.sleep(0.05)
        tree = net.state_dict()
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

    def test_is_a_copy_that_pickles(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        tree = net.state_dict()
        x = net.a.x.clone()
        step(net, virtual_timer)  # the node updates its state in place
        assert not torch.equal(net.a.x, x)
        assert torch.equal(tree["nodes"]["a"]["x"], x)
        again = pickle.loads(pickle.dumps(tree))
        assert torch.equal(again["nodes"]["a"]["x"], x)
        assert again["timer"] == tree["timer"]

    def test_cannot_receive_in_the_node_state_dict(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        class Receiving(Pair):
            @override
            def state_dict(self) -> object:
                try:
                    self.x.recv()
                except RuntimeError as e:
                    return str(e)
                return None

        net = run_network(Solo(Receiving()), _manual=True)
        assert "only the node that owns" in net.state_dict()["nodes"]["node"]


class TestHostMethodsFromANode:
    def test_raise_in_state_dict_on_the_control_thread(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        networks: list[Network] = []

        class Writing(Pair):
            @override
            def state_dict(self) -> object:
                try:
                    networks[0].read(self.out)
                except RuntimeError as e:
                    return str(e)
                return None

        networks.append(run_network(Solo(Writing()), _manual=True))
        assert "host methods" in networks[0].state_dict()["nodes"]["node"]

    def test_raise(self, virtual_timer: GlobalTimer, run_network: RunNetwork):
        networks: list[Chain] = []

        def call_every_host_method() -> None:
            net = networks[0]
            for call in (
                lambda: net.write({net.prev: u(0)}),
                lambda: net.read(net.next),
                lambda: net.subscribe(net.next),
                net.state_dict,
                net.shutdown,
                net.join,
            ):
                try:
                    call()
                except RuntimeError as e:
                    node.errors.append(e)

        node = Calls(call_every_host_method)
        chain = Chain()
        chain.caller = node  # pyright: ignore[reportAttributeAccessIssue]
        networks.append(run_network(chain, _manual=True))
        step(chain, virtual_timer)
        assert len(node.errors) == 6
        assert all("host methods" in str(e) for e in node.errors)


class TestLoadStateDict:
    def test_only_before_run(self, virtual_timer: GlobalTimer, run_network: RunNetwork):
        net = run_network(Chain(), _manual=True)
        tree = net.state_dict()
        with pytest.raises(RuntimeError, match="before run"):
            net.load_state_dict(tree)

    def test_rejects_a_tree_that_does_not_match(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        source = run_network(Chain(), _manual=True)
        tree = source.state_dict()
        source.shutdown()
        other = Solo(Pair())
        other.build()
        with pytest.raises(ValueError, match="do not match"):
            other.load_state_dict(tree)
        same = Chain()
        same.build()
        broken: list[tuple[dict[str, Any], str]] = [
            ({k: v for k, v in tree.items() if k != "timer"}, "no 'timer'"),
            ({**tree, "timer": {"time_ns": 0}}, "timer do not match"),
            ({**tree, "timer": {"time_ns": "0", "scale": 1.0}}, "int time_ns"),
            ({**tree, "last_update_ns": {"a": 0.5, "b": 0}}, "is not an int"),
        ]
        for wrong, match in broken:
            with pytest.raises(ValueError, match=match):
                same.load_state_dict(wrong)

    def test_a_broken_tree_changes_nothing(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        source = run_network(Chain(), _manual=True)
        tree = source.state_dict()
        source.shutdown()
        slots = dict(tree["slots"])
        first, last = list(slots)[0], list(slots)[-1]
        slots[first] = (torch.full((2,), 7.0, dtype=F64), slots[first][1])
        slots[last] = (slots[last][0], 1.5)  # not an int
        net = Chain()
        net.build()
        with pytest.raises(ValueError, match="not an int"):
            net.load_state_dict({**tree, "slots": slots})
        assert torch.equal(net.read(net.a.next)[0], torch.zeros(2, dtype=F64))

    def test_a_refused_run_can_be_retried(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        source = run_network(Chain(), _manual=True)
        tree = source.state_dict()
        source.shutdown()
        net = Chain()
        net.build()
        net.load_state_dict({**tree, "timer": {**tree["timer"], "scale": -1.0}})
        with pytest.raises(ValueError):
            run_network(net)  # refused before any node starts
        assert not net.a.started
        net.load_state_dict(tree)
        run_network(net, _manual=True)
        assert net.a.started


class TestNodeException:
    def test_stops_the_network_and_reaches_the_host(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(raise_in_b=3), _manual=True)
        values = net.subscribe(net.next)
        step(net, virtual_timer)
        step(net, virtual_timer)
        with pytest.raises(NetworkError) as raised:
            step(net, virtual_timer)
        error = net.b.error
        assert raised.value.__cause__ is error
        calls: list[Callable[[], object]] = [
            lambda: net.write({net.prev: u(0)}),
            lambda: net.read(net.next),
            lambda: net.subscribe(net.next),
            lambda: next(values),
            net.state_dict,
            net.shutdown,
            net.join,
        ]
        for call in calls:
            with pytest.raises(NetworkError) as raised:
                call()
            assert raised.value.__cause__ is error

    def test_ends_a_subscription(self, run_network: RunNetwork):
        net = run_network(Chain())
        values = net.subscribe(net.next)
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
    def test_does_nothing_before_run_and_after_the_end(self, run_network: RunNetwork):
        net = Chain()
        net.build()
        net.shutdown()
        run_network(net)
        net.shutdown()
        net.shutdown(timeout=0.0)

    def test_raises_after_the_timeout_and_the_network_goes_on_stopping(
        self, run_network: RunNetwork
    ):
        node = SlowStop()
        net = run_network(Solo(node), _manual=True)
        try:
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                net.shutdown(timeout=0.1)
            assert time.monotonic() - started < 5.0
            assert not node.stopped.is_set()
        finally:
            node.release.set()
        net.join()
        assert node.stopped.is_set()
        net.state_dict()  # a normal end

    def test_rejects_a_negative_timeout(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        for timeout in (-1.0, math.nan):
            with pytest.raises(ValueError, match="timeout"):
                net.shutdown(timeout=timeout)
        # Nothing was asked: the network still runs and updates.
        with pytest.raises(TimeoutError):
            net.join(timeout=0.0)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        assert net.a.updates == 1

    def test_raises_what_a_node_raised(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(raise_in_b=1), _manual=True)
        with pytest.raises(NetworkError):
            step(net, virtual_timer)
        with pytest.raises(NetworkError) as raised:
            net.shutdown(timeout=1.0)
        assert raised.value.__cause__ is net.b.error

    def test_asks_on_one_thread_however_often_it_times_out(
        self, run_network: RunNetwork
    ):
        def askers() -> int:
            return sum(t.name == "mujou-shutdown" for t in threading.enumerate())

        node = SlowStop()
        net = run_network(Solo(node), _manual=True)
        before = askers()
        try:
            for _ in range(5):
                with pytest.raises(TimeoutError):
                    net.shutdown(timeout=0.01)
            assert askers() - before <= 1
        finally:
            node.release.set()
        net.join()
        ended = askers()
        net.shutdown(timeout=1.0)  # after the end: nothing to ask
        assert askers() <= ended


class TestJoin:
    def test_returns_once_another_thread_shuts_down(self, run_network: RunNetwork):
        net = run_network(Chain(), _manual=True)
        stopper = threading.Timer(0.1, net.shutdown)
        stopper.start()
        net.join()
        stopper.join()
        net.state_dict()  # a normal end

    def test_raises_after_the_timeout_and_the_network_keeps_running(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Chain(), _manual=True)
        with pytest.raises(TimeoutError):
            net.join(timeout=0.05)
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        assert torch.equal(net.read(net.next)[0], net.b.x)

    def test_rejects_a_negative_timeout(self, run_network: RunNetwork):
        net = run_network(Chain(), _manual=True)
        for timeout in (-1.0, math.nan):
            with pytest.raises(ValueError, match="timeout"):
                net.join(timeout=timeout)


class TestEndToEnd:
    @pytest.mark.slow
    def test_approaches_the_equilibrium_in_real_time(self, run_network: RunNetwork):
        tau, presented = 0.01, 0.3  # 30 time constants
        net = run_network(Chain(tau=tau))
        before = torch.zeros(2, dtype=F64)
        for k in (1, 2):
            net.write({net.prev: u(k)})
            mujou.time.sleep(presented)
            out, _ = net.read(net.next)
            # After a step of the input by d, the error of two leaky stages in a
            # row is d (1 + t/tau) exp(-t/tau). Allow the nodes to have run for
            # only a third of the time (scheduling under load): t = 10 tau.
            d = (u(k) - before).abs().max().item()
            bound = d * (1 + 10) * math.exp(-10)
            assert (out - u(k)).abs().max().item() <= bound
            before = u(k)

    @pytest.mark.parametrize("b_first", [False, True])
    def test_manual_steps_match_the_discrete_algorithm(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork, b_first: bool
    ):
        net = run_network(Chain(), _manual=True)
        order = (net.b, net.a) if b_first else (net.a, net.b)
        xa = xb = sent_a = torch.zeros(2, dtype=F64)
        for k in range(20):
            net.write({net.prev: u(k)})
            step(net, virtual_timer, *order)
            if b_first:  # Jacobi: b reads a's value of the previous step
                xb = euler(xb, sent_a)
                xa = sent_a = euler(xa, u(k))
            else:  # Gauss-Seidel: b reads a's value of this step
                xa = sent_a = euler(xa, u(k))
                xb = euler(xb, sent_a)
        assert torch.equal(net.a.x, xa)
        assert torch.equal(net.b.x, xb)
        assert torch.equal(net.read(net.next)[0], xb)

    @pytest.mark.parametrize("b_first", [False, True])
    def test_a_restored_network_continues_the_same(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork, b_first: bool
    ):
        k, m = 5, 7
        net = run_network(Chain(), _manual=True)
        order = (net.b, net.a) if b_first else (net.a, net.b)
        for i in range(k):
            net.write({net.prev: u(i)})
            step(net, virtual_timer, *order)
        step(net, virtual_timer, net.a)  # b's last update is now one step behind
        net.write({net.prev: u(k)})  # the input slot is part of the state too
        tree = pickle.loads(pickle.dumps(net.state_dict()))
        for i in range(k, k + m):
            step(net, virtual_timer, *order)
            net.write({net.prev: u(i + 1)})
        expected = (net.a.x.clone(), net.b.x.clone(), net.read(net.next))
        net.shutdown()

        restored = Chain()
        restored.build()
        restored.load_state_dict(tree)
        run_network(restored, _manual=True)
        # Right after the restore the state is the tree, including where each dt
        # starts and every channel slot (which Jacobi order reads first).
        again = restored.state_dict()
        assert again["last_update_ns"] == tree["last_update_ns"]
        assert again["timer"] == tree["timer"]
        for name, (value, time_ns) in tree["slots"].items():
            assert torch.equal(again["slots"][name][0], value)
            assert again["slots"][name][1] == time_ns
        restored_order = (
            (restored.b, restored.a) if b_first else (restored.a, restored.b)
        )
        for i in range(k, k + m):
            step(restored, virtual_timer, *restored_order)
            restored.write({restored.prev: u(i + 1)})
        assert torch.equal(restored.a.x, expected[0])
        assert torch.equal(restored.b.x, expected[1])
        out, t = restored.read(restored.next)
        assert torch.equal(out, expected[2][0]) and t == expected[2][1]
        # The nodes got copies: their updates in place did not reach the tree.
        assert not torch.equal(tree["nodes"]["a"]["x"], restored.a.x)

    @pytest.mark.parametrize("b_first", [False, True])
    def test_a_network_restored_after_the_end_continues_the_same(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork, b_first: bool
    ):
        k, m, gap = 5, 7, 0.3

        def present(net: Chain, steps: range) -> None:
            order = (net.b, net.a) if b_first else (net.a, net.b)
            for i in steps:
                net.write({net.prev: u(i)})
                step(net, virtual_timer, *order)
            step(net, virtual_timer, net.a)  # b's last update is one step behind
            net.write({net.prev: u(steps.stop)})
            virtual_timer.advance(gap)  # the input stays presented a while

        # Saved after the end.
        ended = run_network(Chain(), _manual=True)
        present(ended, range(k))
        ended.shutdown()
        virtual_timer.advance(100.0)  # time goes on after the end
        tree = pickle.loads(pickle.dumps(ended.state_dict()))

        # The same run without stopping.
        through = run_network(Chain(), _manual=True)
        present(through, range(k))
        present(through, range(k + 1, k + 1 + m))
        expected = (through.a.x.clone(), through.b.x.clone())
        through.shutdown()

        restored = Chain()
        restored.build()
        restored.load_state_dict(tree)
        run_network(restored, _manual=True)
        present(restored, range(k + 1, k + 1 + m))
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
            "net = mujou.run(Net())\n"
            "threading.Thread(target=lambda: [_ for _ in net.subscribe(net.out)])"
            ".start()\n"
        )
        done = self.run_script(body, marker)
        assert done.returncode == 0, done.stderr
        assert marker.read_text() == "stopped"


def needs_gpus(n: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < n,
        reason=f"needs {n} CUDA device(s)",
    )


# Large enough that the GPU is still working when the CPU moves on: a missing wait
# for a stream shows as a value in the middle of an update.
BIG = 1 << 26  # 256 MB of float32
ADDS = 20  # in-place updates of the whole tensor per on_update


class CudaLeaky(Node):
    """``Leaky`` with ``tau=1`` on ``cuda:0``."""

    def __init__(self) -> None:
        super().__init__(device="cuda:0")
        self.prev = PrevPort(recv=vec())
        self.next = NextPort(send=vec())

    @override
    def on_start(self) -> None:
        self.x = torch.zeros(2, dtype=F64, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        self.x += min(dt, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)


class CudaChain(Network):
    def __init__(self) -> None:
        self.a = CudaLeaky()
        self.b = CudaLeaky()
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

    def __init__(self) -> None:
        super().__init__(device="cuda:0")
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

    def __init__(self) -> None:
        super().__init__(device="cuda:0")
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

    def __init__(self) -> None:
        super().__init__(device="cuda:0")

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

    def __init__(self) -> None:
        super().__init__(device="cuda:0")

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


@needs_gpus(1)
class TestCuda:
    @needs_gpus(2)
    def test_a_message_goes_from_one_gpu_to_another(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        source, sink = CudaSource("cuda:0"), CudaSink("cuda:1")
        net = run_network(SourceToSink(source, sink), _manual=True)
        step(net, virtual_timer, source, sink)
        received = sink.received
        assert received is not None
        assert received.device == torch.device("cuda", 1)
        assert torch.equal(received.cpu(), torch.tensor([1.0, 2.0], dtype=F64))

    def test_manual_steps_match_the_discrete_algorithm(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(CudaChain(), _manual=True)
        xa = torch.zeros(2, dtype=F64, device="cuda:0")
        xb = torch.zeros(2, dtype=F64, device="cuda:0")
        for k in range(20):
            net.write({net.prev: u(k)})
            step(net, virtual_timer, net.a, net.b)
            xa = euler(xa, u(k).to("cuda:0"))
            xb = euler(xb, xa)
        assert torch.equal(net.read(net.next)[0], xb.cpu())

    @pytest.mark.slow
    def test_on_start_sees_the_work_the_host_queued(self, run_network: RunNetwork):
        seen: list[float | None] = []
        for _ in range(5):
            # Run each node right after making it, while its work is still queued.
            node = QueuesOnTheHost()
            run_network(Solo(node), _manual=True).shutdown()
            seen.append(node.seen)
        assert seen == [float(HOST_ADDS)] * 5

    @pytest.mark.slow
    def test_state_dict_waits_for_the_work_of_on_start(self, run_network: RunNetwork):
        captured: list[tuple[float, float]] = []
        for _ in range(5):
            net = run_network(Solo(BusyStart()), _manual=True)
            w = net.state_dict()["nodes"]["node"]["w"]
            net.shutdown()
            captured.append((w.min().item(), w.max().item()))
        assert captured == [(float(HOST_ADDS), float(HOST_ADDS))] * 5

    @pytest.mark.slow
    def test_state_dict_after_the_end_waits_for_the_work_of_on_stop(
        self, run_network: RunNetwork
    ):
        captured: list[tuple[float, float]] = []
        for _ in range(5):
            net = run_network(Solo(BusyStop()), _manual=True)
            net.shutdown()
            w = net.state_dict()["nodes"]["node"]["w"]
            captured.append((w.min().item(), w.max().item()))
        assert captured == [(float(HOST_ADDS), float(HOST_ADDS))] * 5

    @pytest.mark.slow
    def test_state_dict_is_whole_while_a_node_updates_in_place(
        self, run_network: RunNetwork
    ):
        net = run_network(Solo(Counter()))
        mujou.time.sleep(0.2)
        trees: list[dict[str, Any]] = []
        for _ in range(40):
            trees.append(net.state_dict())
            mujou.time.sleep(0.01)
        torn = [i for i, tree in enumerate(trees) if not is_uniform(tree)]
        # The copies stay as taken while the node goes on updating in place.
        mujou.time.sleep(0.2)
        changed = [i for i, tree in enumerate(trees) if not is_uniform(tree)]
        net.shutdown()
        assert torn == []
        assert changed == []
        assert trees[-1]["nodes"]["node"]["k"] > 0

    @pytest.mark.slow
    def test_a_restored_node_starts_from_the_tree(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(Solo(Counter()), _manual=True)
        for _ in range(3):
            step(net, virtual_timer)
        tree = net.state_dict()
        net.shutdown()

        node = Counter()
        restored = Solo(node)
        restored.build()
        restored.load_state_dict(tree)
        run_network(restored, _manual=True)
        step(restored, virtual_timer)
        assert node.first is not None
        expected = torch.full((BIG,), float(3 * ADDS), device="cuda:0")
        assert torch.equal(node.first, expected)
        assert torch.equal(tree["nodes"]["node"]["w"].to("cuda:0"), expected)
