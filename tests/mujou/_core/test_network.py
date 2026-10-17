import threading
from typing import TYPE_CHECKING, Any

import pytest

from mujou._core.network import Compose, Graph, Network
from mujou._core.node import Node
from mujou._core.port import NextPort, Port, PrevPort
from mujou._core.spec import TensorSpec

if TYPE_CHECKING:
    from tests.mujou.conftest import Start


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
        graph = net.build()
        assert [path for path, _ in graph.nodes] == ["body.l1", "body.l2", "head"]
        assert [node for _, node in graph.nodes] == [
            net.body.l1,
            net.body.l2,
            net.head,
        ]
        assert graph.port_path(net.body.l1.next) == "body.l1.next"
        assert graph.port_path(net.prev) == "body.l1.prev"
        assert graph.port_path(net.teacher) == "head.teacher"
        assert graph.port_path(net.body.next) == "body.l2.next"

    def test_port_path_of_a_foreign_port(self):
        graph = Classifier().build()
        with pytest.raises(ValueError, match="not a port of this network"):
            graph.port_path(Port(send=h()))

    def test_returns_the_same_graph_every_time(self):
        net = Classifier()
        assert net.build() is net.build()

    def test_a_graph_is_made_only_by_build(self):
        with pytest.raises(TypeError, match="build"):
            Graph()

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

    def test_rejects_a_node_of_another_built_network(self):
        # A node without ports: only the mark on the node itself tells.
        class Lone(Node):
            pass

        class Net(Network):
            def __init__(self, node: Node) -> None:
                self.node = node

        lone = Lone()
        Net(lone).build()
        with pytest.raises(RuntimeError, match="'node' belongs to a network"):
            Net(lone).build()

    def test_builds_once_under_concurrent_calls(self):
        nets = [Classifier() for _ in range(20)]
        for net in nets:
            barrier = threading.Barrier(2)
            graphs: list[Graph] = []

            def build(net: Classifier = net, barrier: threading.Barrier = barrier):
                barrier.wait()
                graphs.append(net.build())

            threads = [threading.Thread(target=build) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert graphs[0] is graphs[1]

    def test_rejects_a_network_without_nodes(self):
        class Empty(Network):
            def __init__(self) -> None:
                self.inner = Network()

        with pytest.raises(ValueError, match="no node"):
            Empty().build()


class TestCompose:
    def test_paths_by_position_and_by_keyword(self):
        first, second, named = Stack(), Stack(), Classifier()
        graph = Compose(first, second, enc=named).build()
        assert [path for path, _ in graph.nodes] == [
            "0.l1",
            "0.l2",
            "1.l1",
            "1.l2",
            "enc.body.l1",
            "enc.body.l2",
            "enc.head",
        ]
        assert graph.port_path(named.teacher) == "enc.head.teacher"

    def test_nests(self):
        inner = Compose(a=Stack())
        graph = Compose(inner).build()
        assert [path for path, _ in graph.nodes] == ["0.a.l1", "0.a.l2"]

    def test_the_networks_may_be_connected(self):
        source, sink = Stack(), Stack()
        source.connect(sink)
        graph = Compose(source, sink).build()
        assert graph.port_path(sink.prev) == "1.l1.prev"

    def test_rejects_what_is_not_a_network(self):
        with pytest.raises(TypeError, match="Hidden"):
            Compose(Hidden())  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="'x'"):
            Compose(x=Hidden())  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("name", ["build", "connect", "_private", "", "a.b"])
    def test_rejects_names_that_would_hide_its_own(self, name: str):
        with pytest.raises(ValueError, match=repr(name)):
            Compose(**{name: Stack()})

    def test_rejects_the_same_network_twice(self):
        stack = Stack()
        with pytest.raises(ValueError, match="'0' and 'enc'"):
            Compose(stack, enc=stack).build()

    def test_mujou_run_composes_by_position(self, start: Start):
        # The same paths as Compose(first, second): the node states are keyed by
        # them.
        run = start(Stack(), Stack())
        assert set(run.state_dict()["nodes"]) == {"0.l1", "0.l2", "1.l1", "1.l2"}

    def test_mujou_run_of_one_network_keeps_its_paths(self, start: Start):
        assert set(start(Stack()).state_dict()["nodes"]) == {"l1", "l2"}
