import pytest
import torch

from mujou._core.network import Group, Network
from mujou._core.port import MultiPort, Port
from mujou._core.spec import TensorSpec
from mujou._core.tree import walk
from mujou._core.wiring import Held, wire
from mujou.time import ReadOnlyTimer, VirtualTimer
from tests.mujou.nodes import F64, Chain, Hidden, Probe, vec


def wired(root: Network) -> dict[str, dict[str, dict[str, object]]]:
    """The keys of the slots of each port of each node, and the names of the
    slots."""
    wiring = wire(walk(root), ReadOnlyTimer(VirtualTimer()))
    return {
        path: {
            attribute: {
                key: held.slot.name
                if isinstance(held, Held)
                else {peer: h.slot.name for peer, h in held.items()}
                for key, held in slots.items()
            }
            for attribute, slots in ports.items()
        }
        for path, ports in wiring.slots.items()
    }


class TestSlots:
    def test_by_port_as_the_state_tree_keys_them(self):
        assert wired(Chain()) == {
            "a": {
                "prev": {"in": "a.prev/in"},
                "next": {"out": "a.next/out", "to": {"b.prev": "a.next/ch0"}},
            },
            "b": {"prev": {}, "next": {"out": "b.next/out", "to": {}}},
        }

    def test_a_port_of_the_root_node_is_named_by_its_class(self):
        node = Probe(out=Port(send=vec()), inp=Port(recv=vec()))
        wiring = wire(walk(node), ReadOnlyTimer(VirtualTimer()))
        assert wiring.outlets[node.out].outbox.name == "Probe.out/out"  # pyright: ignore[reportAttributeAccessIssue]
        assert wiring.inputs[node.inp].name == "Probe.inp/in"  # pyright: ignore[reportAttributeAccessIssue]

    def test_a_port_that_only_sends_has_a_slot_for_each_peer(self):
        source = Probe(out=Port(send=vec()))
        sinks = [Probe(inp=Port(recv=vec())) for _ in range(2)]
        for sink in sinks:
            source.out.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        net = Group(source=source, x=sinks[0], y=sinks[1])
        assert wired(net)["source"]["out"] == {
            "out": "source.out/out",
            "to": {"x.inp": "source.out/ch0", "y.inp": "source.out/ch1"},
        }

    def test_a_multiport_has_a_slot_for_each_peer_and_none_for_the_host(self):
        a, b, c = Hidden(), Hidden(), Hidden()
        a.next.connect(b.prev)
        a.next.connect(c.prev)
        slots = wired(Group(a=a, b=b, c=c))
        assert slots["a"]["next"]["to"] == {
            "b.prev": "a.next/ch0",
            "c.prev": "a.next/ch1",
        }
        # An unconnected MultiPort reads nothing of the host.
        assert "in" not in slots["c"]["next"]

    def test_a_port_connected_to_itself_has_a_slot_for_itself(self):
        node = Probe(loop=MultiPort(send=vec(), recv=vec()))
        node.loop.connect(node.loop)  # pyright: ignore[reportAttributeAccessIssue]
        assert wired(Group(node=node))["node"]["loop"] == {
            "out": "node.loop/out",
            "to": {"node.loop": "node.loop/ch0"},
        }


class TestInitial:
    def test_is_in_every_slot_of_the_port_without_a_time(self):
        node = Probe(out=Port(send=vec(), initial=torch.ones(2, dtype=F64)))
        sink = Probe(inp=Port(recv=vec()))
        node.out.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        wiring = wire(walk(Group(node=node, sink=sink)), ReadOnlyTimer(VirtualTimer()))
        outlet = wiring.outlets[node.out]  # pyright: ignore[reportAttributeAccessIssue]
        for slot in (outlet.outbox, outlet.channel(0)):
            message = slot.read()
            assert message.timestamp is None
            assert torch.equal(
                torch.from_dlpack(message.values[0]), torch.ones(2, dtype=F64)
            )

    def test_a_value_that_does_not_match_names_the_port_by_its_path(self):
        net = Group(node=Probe(out=Port(send=vec(), initial=torch.ones(3, dtype=F64))))
        with pytest.raises(ValueError, match=r"^slot 'node\.out/out': expected shape"):
            wire(walk(net), ReadOnlyTimer(VirtualTimer()))
        wrong = Group(node=Probe(out=Port(send=vec(), initial=[1.0])))  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match=r"^node\.out: expected a torch\.Tensor"):
            wire(walk(wrong), ReadOnlyTimer(VirtualTimer()))

    def test_is_read_when_the_run_starts(self):
        value = torch.zeros(2, dtype=F64)
        node = Probe(out=Port(send=vec(), initial=value))
        value.fill_(3.0)  # after the port was made
        wiring = wire(walk(node), ReadOnlyTimer(VirtualTimer()))
        message = wiring.outlets[node.out].outbox.read()  # pyright: ignore[reportAttributeAccessIssue]
        assert torch.equal(torch.from_dlpack(message.values[0]), value)


class TestWrite:
    def test_names_a_port_the_host_does_not_write(self):
        net = Chain()
        wiring = wire(walk(net), ReadOnlyTimer(VirtualTimer()))
        for port, name in ((net.b.prev, "b.prev"), (net.next, "b.next")):
            with pytest.raises(
                ValueError,
                match=rf"^the host does not write {name}: it writes an unconnected "
                "Port of this run that receives$",
            ):
                wiring.write({port: torch.zeros(2, dtype=F64)})

    def test_a_scalar_spec(self):
        node = Probe(inp=Port(recv=TensorSpec(())))
        wiring = wire(walk(node), ReadOnlyTimer(VirtualTimer()))
        wiring.write({node.inp: torch.tensor(1.0)})  # pyright: ignore[reportAttributeAccessIssue]
        (value,) = wiring.inputs[node.inp].read().values  # pyright: ignore[reportAttributeAccessIssue]
        assert torch.equal(torch.from_dlpack(value), torch.tensor(1.0))
