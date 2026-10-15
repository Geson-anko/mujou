from typing import Any

import pytest

from mujou._network import Graph, Network
from mujou._node import Node
from mujou._port import NextPort, Port, PrevPort
from mujou._spec import TensorSpec


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
    """Build ``net`` and return its graph.

    The graph is what build hands to the runtime, and it has no public accessor.
    """
    net.build()
    graph = net._graph  # pyright: ignore[reportPrivateUsage]
    assert graph is not None
    return graph


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
