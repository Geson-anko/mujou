import threading
from typing import Any, override

import pytest

from mujou._core.graph import Graph
from mujou._core.network import Group, Network
from mujou._core.node import Node
from mujou._core.port import MultiPort, Port
from mujou._core.spec import TensorSpec
from tests.mujou.nodes import Hidden


def h(n: int = 3) -> TensorSpec:
    return TensorSpec((2, n))


class Source(Node):
    def __init__(self) -> None:
        super().__init__()
        self.next = MultiPort(send=h(), recv=h())


class Sink(Node):
    def __init__(self) -> None:
        super().__init__()
        self.prev = Port(send=h(), recv=h())
        self.teacher = Port(recv=h(5))


class Stack(Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden()
        self.l2 = Hidden()
        self.l1.next.connect(self.l2.prev)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()
        self.head = Sink()
        self.body.next.connect(self.head.prev)
        self.prev = self.body.prev
        self.teacher = self.head.teacher


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
        with pytest.raises(ValueError, match="not a port of this graph"):
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
                super().__init__()
                self.node = Hidden()

        Single().build()

    def test_rejects_a_tree_that_contains_a_built_network(self):
        class Outer(Network):
            def __init__(self, inner: Network) -> None:
                super().__init__()
                self.inner = inner

        inner = Stack()
        inner.build()
        with pytest.raises(RuntimeError, match="already built"):
            Outer(inner).build()

    def test_rejects_the_same_node_at_two_paths(self):
        class Twice(Network):
            def __init__(self) -> None:
                super().__init__()
                self.a = Hidden()
                self.b = self.a

        with pytest.raises(ValueError, match="'a' and 'b'"):
            Twice().build()

    def test_rejects_a_connection_out_of_the_tree(self):
        net = Classifier()
        Source().next.connect(net.prev)
        with pytest.raises(
            ValueError, match="'body.l1.prev' is connected to a port outside"
        ):
            net.build()

    def test_rejects_a_port_owned_by_two_nodes(self):
        class Borrower(Node):
            def __init__(self, port: Port[Any, Any] | MultiPort[Any, Any]) -> None:
                super().__init__()
                self.port = port

        class Net(Network):
            def __init__(self) -> None:
                super().__init__()
                self.a = Hidden()
                self.b = Borrower(self.a.next)

        with pytest.raises(ValueError, match="'b.port' is already owned as 'a.next'"):
            Net().build()

    def test_rejects_exposing_a_port_outside_the_tree(self):
        class Net(Network):
            def __init__(self) -> None:
                super().__init__()
                self.a = Hidden()
                self.out = Port(send=h())

        with pytest.raises(ValueError, match="exposed at 'out'"):
            Net().build()

    def test_rejects_exposing_a_port_of_a_sibling(self):
        class Inner(Network):
            def __init__(self, node: Hidden) -> None:
                super().__init__()
                self.own = Hidden()
                self.prev = node.prev

        class Outer(Network):
            def __init__(self) -> None:
                super().__init__()
                self.node = Hidden()
                self.inner = Inner(self.node)

        with pytest.raises(ValueError, match="exposed at 'inner.prev'"):
            Outer().build()

    def test_rejects_ports_declared_on_the_class(self):
        class ClassPort(Node):
            port = Port(send=h())

        class Net(Network):
            def __init__(self) -> None:
                super().__init__()
                self.node = ClassPort()

        with pytest.raises(ValueError, match="'port' on the class ClassPort"):
            Net().build()

    def test_rejects_a_node_of_another_built_network(self):
        # A node without ports: only the mark on the node itself tells.
        class Lone(Node):
            pass

        class Net(Network):
            def __init__(self, node: Node) -> None:
                super().__init__()
                self.node = node

        lone = Lone()
        Net(lone).build()
        with pytest.raises(RuntimeError, match="'node' belongs to a network"):
            Net(lone).build()

    def test_ignores_the_equality_of_the_components(self):
        # The marks follow identity: an unhashable network builds, and an equal
        # but distinct node is not taken for a built one.
        class Unhashable(Network):
            def __init__(self, node: Node) -> None:
                super().__init__()
                self.node = node

            @override
            def __eq__(self, other: object) -> bool:
                return isinstance(other, Unhashable)

            __hash__ = None  # pyright: ignore[reportAssignmentType]

        class EqualNode(Node):
            @override
            def __eq__(self, other: object) -> bool:
                return isinstance(other, EqualNode)

            @override
            def __hash__(self) -> int:
                return 0

        class Net(Network):
            def __init__(self, node: Node) -> None:
                super().__init__()
                self.node = node

        Unhashable(Hidden()).build()
        Net(EqualNode()).build()
        Net(EqualNode()).build()

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
                super().__init__()
                self.inner = Network()

        with pytest.raises(ValueError, match="no node"):
            Empty().build()


class TestGroup:
    def test_paths_by_position_and_by_keyword(self):
        first, second, named = Stack(), Stack(), Classifier()
        graph = Group(first, second, enc=named).build()
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
        inner = Group(a=Stack())
        graph = Group(inner).build()
        assert [path for path, _ in graph.nodes] == ["0.a.l1", "0.a.l2"]

    def test_the_networks_may_be_connected(self):
        source, sink = Stack(), Stack()
        source.next.connect(sink.prev)
        graph = Group(source, sink).build()
        assert graph.port_path(sink.prev) == "1.l1.prev"

    def test_takes_nodes_and_networks_alike(self):
        node, named = Hidden(), Hidden()
        graph = Group(node, Stack(), n=named).build()
        assert [path for path, _ in graph.nodes] == ["0", "1.l1", "1.l2", "n"]
        assert graph.port_path(node.prev) == "0.prev"
        assert graph.port_path(named.next) == "n.next"

    def test_rejects_what_is_neither_a_node_nor_a_network(self):
        with pytest.raises(TypeError, match="object"):
            Group(object())  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="'x'"):
            Group(x=1)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("name", ["build", "_private", "", "a.b"])
    def test_rejects_names_that_would_hide_its_own(self, name: str):
        with pytest.raises(ValueError, match=repr(name)):
            Group(**{name: Stack()})

    def test_rejects_the_same_network_twice(self):
        stack = Stack()
        with pytest.raises(ValueError, match="'0' and 'enc'"):
            Group(stack, enc=stack).build()
