import math
import pickle
import threading
from collections.abc import Callable
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
        if self.updates == self.raise_at:
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

    def test_after_load_state_dict_takes_precedence(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        first = Pair()
        net = run_network(Solo(first), _manual=True)
        net.write({first.x: u(1)})
        tree = net.state_dict()
        net.shutdown()

        kept, overwritten = Pair(), Pair()
        for node, value in ((kept, None), (overwritten, u(2))):
            restored = Solo(node)
            restored.build()
            restored.load_state_dict(tree)
            if value is not None:
                restored.write({node.x: value})
            run_network(restored, _manual=True)
            step(restored, virtual_timer)
            restored.shutdown()
        assert torch.equal(kept.seen[0][0][0], u(1))
        assert torch.equal(overwritten.seen[0][0][0], u(2))

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
        values = net.subscribe(net.next)
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
    def test_needs_a_running_network(self, run_network: RunNetwork):
        net = Chain()
        net.build()
        with pytest.raises(RuntimeError, match="not running"):
            net.state_dict()
        run_network(net)
        net.shutdown()
        with pytest.raises(RuntimeError):
            net.state_dict()

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

    def test_cannot_be_called_from_a_node(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        networks: list[Network] = []
        node = Calls(lambda: networks[0].state_dict())
        networks.append(run_network(Solo(node), _manual=True))
        step(networks[0], virtual_timer)
        assert isinstance(node.errors[0], RuntimeError)


class TestLoadStateDict:
    def test_only_before_run(self, run_network: RunNetwork):
        net = run_network(Chain())
        tree = net.state_dict()
        with pytest.raises(RuntimeError, match="before run"):
            net.load_state_dict(tree)

    def test_rejects_a_tree_of_another_network(self, run_network: RunNetwork):
        tree = run_network(Chain()).state_dict()
        other = Solo(Pair())
        other.build()
        with pytest.raises(ValueError, match="do not match"):
            other.load_state_dict(tree)
        del tree["timer"]
        same = Chain()
        same.build()
        with pytest.raises(ValueError, match="no 'timer'"):
            same.load_state_dict(tree)


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
        assert net.exception_state_dict() is None

    def test_keeps_the_state_when_asked(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        net = run_network(
            Chain(raise_in_b=2), capture_state_on_exception=True, _manual=True
        )
        net.write({net.prev: u(1)})
        step(net, virtual_timer)
        with pytest.raises(NetworkError):
            step(net, virtual_timer)
        tree = net.exception_state_dict()
        assert tree is not None
        assert tree["exception"]["node"] == "b"
        assert tree["exception"]["hook"] == "on_update"
        assert "boom after 2 updates" in tree["exception"]["error"]
        assert set(tree["nodes"]) == {"a", "b"}
        pickle.dumps(tree)

    def test_ends_a_subscription(self, run_network: RunNetwork):
        net = run_network(Chain(raise_in_b=100))
        with pytest.raises(NetworkError) as raised:
            for _ in net.subscribe(net.next):
                pass
        assert raised.value.__cause__ is net.b.error


class TestEndToEnd:
    def test_approaches_the_equilibrium_in_real_time(self, run_network: RunNetwork):
        net = run_network(Chain(tau=0.01))
        for k in (1, 2):
            net.write({net.prev: u(k)})
            mujou.time.sleep(0.3)  # 30 time constants
            out, _ = net.read(net.next)
            assert torch.allclose(out, u(k), rtol=0, atol=1e-6)

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

    def test_a_restored_network_continues_the_same(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        k, m = 5, 7
        net = run_network(Chain(), _manual=True)
        for i in range(k):
            net.write({net.prev: u(i)})
            step(net, virtual_timer)
        net.write({net.prev: u(k)})  # the input slot is part of the state too
        tree = pickle.loads(pickle.dumps(net.state_dict()))
        for i in range(k, k + m):
            step(net, virtual_timer)
            net.write({net.prev: u(i + 1)})
        expected = (net.a.x.clone(), net.b.x.clone(), net.read(net.next))
        net.shutdown()

        restored = Chain()
        restored.build()
        restored.load_state_dict(tree)
        run_network(restored, _manual=True)
        for i in range(k, k + m):
            step(restored, virtual_timer)
            restored.write({restored.prev: u(i + 1)})
        assert torch.equal(restored.a.x, expected[0])
        assert torch.equal(restored.b.x, expected[1])
        out, t = restored.read(restored.next)
        assert torch.equal(out, expected[2][0]) and t == expected[2][1]
