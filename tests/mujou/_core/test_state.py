import io
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

from mujou import Group, MultiPort, NetworkError, Node, Port, TensorSpec
from tests.mujou.nodes import F64, Chain, Hidden, Probe, assert_same_tree, vec

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual

H = 1e-3


def u(k: float) -> torch.Tensor:
    return torch.tensor([k, -k], dtype=F64)


class Learned(Node):
    """Holds a module, a tensor and a registered count."""

    def __init__(self) -> None:
        super().__init__(seed=3)
        self.out = Port(send=vec())
        self.linear = nn.Linear(2, 2)
        self.x = torch.zeros(2, dtype=F64)
        self.count = 0
        self.register_state("count")


class TestShape:
    def test_holds_the_timer_and_each_node_by_its_path(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net, inputs={net.prev: u(1)})
        run.step(H)
        tree = run.state_dict()
        assert list(tree) == ["timer", "nodes"]
        assert tree["timer"] == {"time_ns": 1_000_000, "time_scale": 1.0}
        assert list(tree["nodes"]) == ["a", "b"]
        a = tree["nodes"]["a"]
        assert list(a) == ["state", "last_update_ns", "ports"]
        assert a["last_update_ns"] == 1_000_000
        assert a["state"]["updates"] == 1
        assert_same_tree(
            a["ports"],
            {
                "prev": {"in": {"value": u(1), "time_ns": 0}},
                "next": {
                    "out": {"value": net.a.x, "time_ns": 1_000_000},
                    "to": {"b.prev": {"value": net.a.x, "time_ns": 1_000_000}},
                },
            },
        )
        assert_same_tree(
            tree["nodes"]["b"]["ports"],
            {
                "prev": {},
                "next": {"out": {"value": net.b.x, "time_ns": 1_000_000}, "to": {}},
            },
        )

    def test_a_slot_not_written_yet_has_no_time(self, start_manual: StartManual):
        tree = start_manual(Chain()).state_dict()
        assert tree["nodes"]["a"]["ports"]["prev"]["in"]["time_ns"] is None
        assert tree["nodes"]["b"]["ports"]["next"]["out"]["time_ns"] is None

    def test_send_to_writes_only_its_peer(self, start_manual: StartManual):
        spec = TensorSpec((2, 3))
        port = MultiPort(send=spec, recv=spec)
        b, c = Hidden(), Hidden()
        channel = port.connect(b.prev)
        port.connect(c.prev)
        node = Probe(lambda: port.send_to(channel, torch.ones(2, 3)), next=port)
        run = start_manual(Group(a=node, b=b, c=c))
        run.step(H, node)
        to = run.state_dict()["nodes"]["a"]["ports"]["next"]["to"]
        assert to["b.prev"]["time_ns"] == 1_000_000
        assert to["c.prev"]["time_ns"] is None

    def test_a_node_state_is_the_default_one(self, start_manual: StartManual):
        node = Learned()
        tree = start_manual(Group(node=node)).state_dict()
        state = tree["nodes"]["node"]["state"]
        assert list(state) == ["generator", "linear", "x", "count"]
        assert_same_tree(state["linear"], node.linear.state_dict())


class TestCopies:
    @pytest.mark.parametrize("ended", [False, True])
    def test_share_no_memory_with_the_run(self, start_manual: StartManual, ended: bool):
        node = Learned()
        run = start_manual(Group(node=node))
        if ended:
            run.shutdown()
        tree = run.state_dict()
        node.x += 1.0  # in place, as a node updates its state
        with torch.no_grad():
            node.linear.weight.add_(1.0)
        state = tree["nodes"]["node"]["state"]
        assert torch.equal(state["x"], torch.zeros(2, dtype=F64))
        assert not torch.equal(state["linear"]["weight"], node.linear.weight)

    def test_torch_load_reads_it_with_weights_only(self, start_manual: StartManual):
        node = Learned()
        tree = start_manual(Group(node=node)).state_dict()
        buffer = io.BytesIO()
        torch.save(tree, buffer)
        buffer.seek(0)
        assert_same_tree(torch.load(buffer, weights_only=True), tree)


class TestRestore:
    def test_goes_on_from_the_tree(self, start_manual: StartManual):
        first = Learned()
        run = start_manual(Group(node=first))
        first.count = 5
        # Moved on from the seed, which a new node starts from.
        torch.rand(3, generator=first.generator)
        run.step(H)
        tree = run.state_dict()
        run.shutdown()
        again = Learned()
        with torch.no_grad():
            again.linear.weight.fill_(7.0)
        restored = start_manual(Group(node=again), state=tree)
        assert again.count == 5
        assert torch.equal(again.linear.weight, first.linear.weight)
        # The generator goes on with the same sequence.
        assert torch.equal(
            torch.rand(3, generator=again.generator),
            torch.rand(3, generator=first.generator),
        )
        assert_same_tree(
            restored.state_dict()["nodes"]["node"]["ports"],
            tree["nodes"]["node"]["ports"],
        )

    def test_a_node_state_that_does_not_match_stops_the_run(
        self, start_manual: StartManual
    ):
        source = start_manual(Group(node=Learned()))
        tree = source.state_dict()
        source.shutdown()
        del tree["nodes"]["node"]["state"]["x"]
        with pytest.raises(
            NetworkError, match=r"node 'node' raised in load_state_dict"
        ) as raised:
            start_manual(Group(node=Learned()), state=tree)
        assert isinstance(raised.value.__cause__, ValueError)
        assert "missing ['x']" in str(raised.value.__cause__)

    def test_lists_what_does_not_match(self, start_manual: StartManual):
        source = start_manual(Chain())
        tree: dict[str, Any] = source.state_dict()
        source.shutdown()
        del tree["nodes"]["b"]
        del tree["nodes"]["a"]["ports"]["next"]["to"]["b.prev"]
        tree["nodes"]["a"]["ports"]["extra"] = {}
        net = Chain()
        with pytest.raises(ValueError) as raised:
            start_manual(net, state=tree)
        assert str(raised.value) == (
            "the state tree does not match this run: "
            "missing nodes['a']['ports']['next']['to']['b.prev'], "
            "unexpected nodes['a']['ports']['extra'], missing nodes['b']"
        )
        assert not net.a.started

    def test_lists_what_its_top_does_not_match(self, start_manual: StartManual):
        source = start_manual(Chain())
        tree: dict[str, Any] = source.state_dict()
        source.shutdown()
        del tree["timer"]["time_scale"]
        tree["extra"] = {}
        net = Chain()
        with pytest.raises(ValueError) as raised:
            start_manual(net, state=tree)
        assert str(raised.value) == (
            "the state tree does not match this run: "
            "missing timer['time_scale'], unexpected extra"
        )
        assert not net.a.started
