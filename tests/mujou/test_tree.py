from typing import Any, override

import pytest
import torch
from torch import nn

from mujou.network import Group, Network
from mujou.node import Node
from mujou.port import MultiPort, Port
from mujou.spec import TensorSpec
from mujou.tree import walk
from tests.mujou.nodes import Hidden


def h(n: int = 3) -> TensorSpec:
    return TensorSpec((2, n))


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


class Holder(Node):
    """Holds what it is given under ``held``."""

    def __init__(self, held: object) -> None:
        super().__init__()
        self.held = held


class Holds(Network):
    """Holds a node, and what it is given under ``held``."""

    def __init__(self, held: object) -> None:
        super().__init__()
        self.node = Hidden()
        self.held = held


class TestPaths:
    def test_are_depth_first_in_the_order_of_assignment(self):
        net = Classifier()
        tree = walk(net)
        assert [m.path for m in tree.members] == ["body.l1", "body.l2", "head"]
        assert [m.node for m in tree.members] == [net.body.l1, net.body.l2, net.head]
        assert dict(tree.members[2].ports) == {
            "prev": net.head.prev,
            "teacher": net.head.teacher,
        }

    def test_a_port_is_at_the_path_of_its_node(self):
        net = Classifier()
        tree = walk(net)
        assert tree.path(net.body.l1.next) == "body.l1.next"
        # An exposed port is the port of its node.
        assert tree.path(net.prev) == "body.l1.prev"
        assert tree.label(net.teacher) == "head.teacher"

    def test_a_node_that_is_the_root_is_at_the_empty_path(self):
        node = Hidden()
        tree = walk(node)
        assert [m.path for m in tree.members] == [""]
        assert tree.name("") == "Hidden"
        assert tree.path(node.next) == "next"
        # The path alone would not say which node the port is of.
        assert tree.label(node.next) == "Hidden.next"

    def test_names_a_port_outside_the_tree_by_its_repr(self):
        port = Port(send=h())
        assert walk(Hidden()).label(port) == repr(port)

    def test_follow_assignments_after_the_constructor(self):
        net = Stack()
        net.extra = Hidden()  # pyright: ignore[reportAttributeAccessIssue]
        net.l2 = Hidden()  # replaced: keeps its place
        del net.l1
        assert [m.path for m in walk(net).members] == ["l2", "extra"]

    def test_a_network_without_nodes_has_no_members(self):
        class Empty(Network):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Network()

        assert walk(Empty()).members == ()


class TestChecks:
    def test_a_node_that_did_not_call_the_base_init(self):
        class Lazy(Node):
            def __init__(self) -> None:
                self.out = Port(send=h())  # an assignment before it is fine

        class Late(Node):
            def __init__(self) -> None:
                self.out = Port(send=h())
                super().__init__()

        walk(Late())
        with pytest.raises(
            TypeError, match=r"^Lazy\.__init__ did not call super\(\)\.__init__\(\)$"
        ):
            walk(Group(node=Lazy()))

    @pytest.mark.parametrize("make", [Hidden, Stack])
    def test_a_node_holding_a_node_or_a_network(self, make: type[Node | Network]):
        with pytest.raises(
            TypeError, match=rf"^node\.held is a {make.__name__}: a node holds no"
        ):
            walk(Group(node=Holder(make())))

    @pytest.mark.parametrize(
        "held",
        [
            [Port(send=h())],
            (MultiPort(recv=h()),),
            {"x": Port(send=h())},
            [Hidden()],
        ],
        ids=["list", "tuple", "dict", "node"],
    )
    @pytest.mark.parametrize("holder", [Holder, Holds])
    def test_parts_in_a_list_a_tuple_or_a_dict(
        self, holder: type[Holder | Holds], held: object
    ):
        with pytest.raises(
            TypeError,
            match=rf"^part\.held holds ports, nodes or networks in a "
            rf"{type(held).__name__}, which a run does not read",
        ):
            walk(Group(part=holder(held)))

    def test_takes_lists_of_other_values(self):
        walk(Holds([1, torch.zeros(1)]))
        walk(Holder({"x": torch.zeros(1)}))

    def test_the_same_node_at_two_places(self):
        class Twice(Network):
            def __init__(self) -> None:
                super().__init__()
                self.a = Hidden()
                self.b = self.a

        with pytest.raises(
            ValueError, match=r"^the Hidden at 'a' is also at 'b'; a node, a network"
        ):
            walk(Twice())

    def test_the_same_network_at_two_places(self):
        stack = Stack()
        with pytest.raises(ValueError, match="'x' is also at 'y'"):
            walk(Group(x=stack, y=stack))

    def test_a_port_held_by_two_nodes(self):
        class Net(Network):
            def __init__(self) -> None:
                super().__init__()
                self.a = Hidden()
                self.b = Holder(self.a.next)

        with pytest.raises(ValueError, match=r"'a\.next' is also at 'b\.held'"):
            walk(Net())

    def test_a_port_connected_to_one_outside_the_tree(self):
        net = Classifier()
        Hidden().next.connect(net.prev)
        with pytest.raises(
            ValueError,
            match=r"^body\.l1\.prev is connected to MultiPort\(.*\), whose node is "
            r"not in the tree",
        ):
            walk(net)

    @pytest.mark.parametrize(
        "make",
        [lambda: torch.zeros(2), lambda: nn.Linear(2, 2)],
        ids=["tensor", "module"],
    )
    def test_two_nodes_holding_the_same_state(self, make: Any):
        shared = make()
        a, b = Holder(shared), Holder(shared)
        with pytest.raises(
            ValueError,
            match=r"^a\.held and b\.held hold the same \w+: a state has one writer",
        ):
            walk(Group(a=a, b=b))

    def test_two_nodes_holding_the_same_parameter(self):
        a, b = nn.Linear(2, 2), nn.Linear(2, 2)
        b.weight = a.weight  # tied
        with pytest.raises(
            ValueError,
            match=r"^a\.held\.weight and b\.held\.weight hold the same Parameter",
        ):
            walk(Group(a=Holder(a), b=Holder(b)))
        with pytest.raises(
            ValueError, match=r"^a\.held and b\.held\.weight hold the same Parameter"
        ):
            walk(Group(a=Holder(a.weight), b=Holder(b)))

    def test_two_modules_may_have_the_same_extra_state(self):
        class Counted(nn.Linear):
            @override
            def get_extra_state(self) -> object:
                return 0  # the same int object in every module

            @override
            def set_extra_state(self, state: object) -> None:
                pass

        walk(Group(a=Holder(Counted(2, 2)), b=Holder(Counted(2, 2))))

    def test_one_node_may_hold_its_state_twice(self):
        node = Holder(torch.zeros(2))
        node.alias = node.held  # pyright: ignore[reportAttributeAccessIssue]
        walk(node)

    def test_two_nodes_may_hold_the_same_plain_value(self):
        a, b = Holder(0), Holder(0)
        a.register_state("held")
        b.register_state("held")
        walk(Group(a=a, b=b))

    def test_a_network_holds_ports_as_references(self):
        class Exposes(Network):
            def __init__(self, port: Port[Any, Any]) -> None:
                super().__init__()
                self.node = Hidden()
                self.out = port

        # Even a port of no node of the tree: only the nodes own ports.
        tree = walk(Exposes(Port(send=h())))
        assert len(tree.owners) == 2
