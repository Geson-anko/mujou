import copy
from typing import Any

import pytest

from mujou._core.component import children_of, ports_of
from mujou._core.network import Network
from mujou._core.node import Node
from mujou._core.port import Port, PrevPort
from mujou._core.spec import TensorSpec
from tests.mujou.nodes import Hidden


def h() -> TensorSpec:
    return TensorSpec((2, 3))


class Pair(Network):
    def __init__(self) -> None:
        super().__init__()
        self.a = Hidden()
        self.b = Hidden()
        self.a.next.connect(self.b.prev)
        self.prev = self.a.prev
        self.next = self.b.next


class TestRegistration:
    def test_registers_ports_in_the_order_of_assignment(self):
        node = Hidden()
        node.count = 0  # pyright: ignore[reportAttributeAccessIssue]
        node.extra = Port(send=h())  # pyright: ignore[reportAttributeAccessIssue]
        assert dict(ports_of(node)) == {
            "prev": node.prev,
            "next": node.next,
            "extra": node.extra,  # pyright: ignore[reportAttributeAccessIssue]
        }
        assert dict(children_of(node)) == {}

    def test_registers_the_nodes_and_networks_of_a_network(self):
        class Outer(Network):
            def __init__(self) -> None:
                super().__init__()
                self.pair = Pair()
                self.node = Hidden()
                self.prev = self.pair.prev

        outer = Outer()
        assert dict(children_of(outer)) == {"pair": outer.pair, "node": outer.node}
        assert dict(ports_of(outer)) == {"prev": outer.pair.prev}

    def test_a_name_assigned_again_keeps_its_place(self):
        node = Hidden()
        port = PrevPort(send=h(), recv=h())
        node.prev = port
        assert list(ports_of(node).items()) == [("prev", port), ("next", node.next)]

    def test_another_value_or_del_removes_the_name(self):
        node = Hidden()
        node.prev = None  # pyright: ignore[reportAttributeAccessIssue]
        del node.next
        assert dict(ports_of(node)) == {}
        net = Pair()
        net.a = 1  # pyright: ignore[reportAttributeAccessIssue]
        del net.b
        assert dict(children_of(net)) == {}

    def test_moves_a_name_between_ports_and_components(self):
        net = Pair()
        port = Port(send=h())
        net.a = port  # pyright: ignore[reportAttributeAccessIssue]
        assert "a" not in children_of(net) and ports_of(net)["a"] is port
        node = Hidden()
        net.prev = node  # pyright: ignore[reportAttributeAccessIssue]
        assert "prev" not in ports_of(net) and children_of(net)["prev"] is node

    def test_does_not_walk_lists_or_dicts(self):
        class Listed(Network):
            def __init__(self) -> None:
                super().__init__()
                self.nodes = [Hidden(), Hidden()]
                self.ports = {"x": Port(send=h())}

        net = Listed()
        assert dict(children_of(net)) == {} and dict(ports_of(net)) == {}
        with pytest.raises(ValueError, match="no node"):
            net.build()

    def test_a_property_registers_what_its_setter_stores(self):
        class Wrapped(Node):
            def __init__(self) -> None:
                super().__init__()
                self.out = Port(send=h())

            @property
            def out(self) -> Port[Any, Any, Any]:
                return self._out

            @out.setter
            def out(self, port: Port[Any, Any, Any]) -> None:
                self._out = port

        node = Wrapped()
        assert list(ports_of(node)) == ["_out"]

    def test_a_shallow_copy_does_not_share_changes(self):
        node = Hidden()
        twin = copy.copy(node)
        twin.extra = Port(send=h())  # pyright: ignore[reportAttributeAccessIssue]
        assert "extra" not in ports_of(node)


class TestAssignmentChecks:
    def test_a_port_before_node_init(self):
        class Early(Node):
            def __init__(self) -> None:
                self.count = 0  # an ordinary value is fine
                self.out = Port(send=h())
                super().__init__()

        with pytest.raises(
            AttributeError,
            match=r"cannot assign a port before Node\.__init__\(\) call; call "
            r"super\(\)\.__init__\(\) first",
        ):
            Early()

    def test_a_node_before_network_init(self):
        class Early(Network):
            def __init__(self) -> None:
                self.node = Hidden()
                super().__init__()

        with pytest.raises(
            AttributeError, match=r"cannot assign a Hidden before Network\.__init__"
        ):
            Early()

    @pytest.mark.parametrize("make", [Hidden, Pair])
    def test_a_node_holding_a_component(self, make: type[Node | Network]):
        node = Hidden()
        with pytest.raises(
            ValueError,
            match=f"the node Hidden holds a {make.__name__} at 'inner'; only "
            "networks hold nodes and networks",
        ):
            node.inner = make()  # pyright: ignore[reportAttributeAccessIssue]
        assert not hasattr(node, "inner")


class TestAfterBuild:
    def test_rejects_assigning_or_removing_ports_and_components(self):
        net = Pair()
        net.build()
        node = net.a
        with pytest.raises(RuntimeError, match=r"cannot assign Pair\.c: .*built"):
            net.c = Hidden()  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(RuntimeError, match=r"cannot assign Pair\.out"):
            net.out = node.next  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(RuntimeError, match=r"cannot replace Pair\.a"):
            net.a = None  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(RuntimeError, match=r"cannot delete Pair\.b"):
            del net.b
        with pytest.raises(RuntimeError, match=r"cannot assign Hidden\.extra"):
            node.extra = Port(send=h())  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(RuntimeError, match=r"cannot delete Hidden\.prev"):
            del node.prev
        assert list(children_of(net)) == ["a", "b"] and net.a is node
        assert list(ports_of(net)) == ["prev", "next"]
        assert list(ports_of(node)) == ["prev", "next"]

    def test_ordinary_attributes_stay_free(self):
        net = Pair()
        net.build()
        net.note = "built"  # pyright: ignore[reportAttributeAccessIssue]
        net.a.count = 1  # pyright: ignore[reportAttributeAccessIssue]
        del net.note  # pyright: ignore[reportAttributeAccessIssue]
