import pytest
import torch

from mujou._native import BufferView
from mujou.network import Group, Network
from mujou.port import MultiPort, Port
from mujou.spec import TensorSpec
from mujou.time import ReadOnlyTimer, VirtualTimer
from mujou.tree import walk
from mujou.wiring import Wiring, wire
from tests.mujou.nodes import F64, Chain, Hidden, Probe, probe_on, vec


def wired(root: Network | Probe) -> Wiring:
    return wire(walk(root), ReadOnlyTimer(VirtualTimer()))


class TestSlots:
    def test_are_named_by_the_path_of_the_port(self):
        net = Chain()
        wiring = wired(net)
        assert wiring.inputs[net.a.prev].name == "a.prev/in"
        assert wiring.outlets[net.a.next].outbox.name == "a.next/out"
        assert [slot.name for slot in wiring.channels(net.a.next)] == ["a.next/ch0"]
        assert wiring.channels(net.b.next) == []
        # A connected port reads its peer, not the host.
        assert net.b.prev not in wiring.inputs

    def test_a_port_of_the_root_node_is_named_by_its_class(self):
        node = Probe(out=Port(send=vec()), inp=Port(recv=vec()))
        wiring = wired(node)
        assert wiring.outlets[node.out].outbox.name == "Probe.out/out"  # pyright: ignore[reportAttributeAccessIssue]
        assert wiring.inputs[node.inp].name == "Probe.inp/in"  # pyright: ignore[reportAttributeAccessIssue]

    def test_a_port_that_only_sends_has_a_slot_for_each_peer(self):
        source = Probe(out=Port(send=vec()))
        sinks = [Probe(inp=Port(recv=vec())) for _ in range(2)]
        for sink in sinks:
            source.out.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        wiring = wired(Group(source=source, x=sinks[0], y=sinks[1]))
        names = [slot.name for slot in wiring.channels(source.out)]  # pyright: ignore[reportAttributeAccessIssue]
        assert names == ["source.out/ch0", "source.out/ch1"]

    def test_an_unconnected_multiport_reads_nothing_of_the_host(self):
        a, b = Hidden(), Hidden()
        a.next.connect(b.prev)
        wiring = wired(Group(a=a, b=b))
        assert [slot.name for slot in wiring.channels(a.next)] == ["a.next/ch0"]
        assert b.next not in wiring.inputs

    def test_a_port_connected_to_itself_has_a_slot_for_itself(self):
        node = Probe(loop=MultiPort(send=vec(), recv=vec()))
        node.loop.connect(node.loop)  # pyright: ignore[reportAttributeAccessIssue]
        wiring = wired(Group(node=node))
        assert [slot.name for slot in wiring.channels(node.loop)] == ["node.loop/ch0"]  # pyright: ignore[reportAttributeAccessIssue]


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


class TestDevices:
    def test_nodes_on_the_cpu_read_no_cuda_device(self):
        assert wired(Chain()).read_devices == (frozenset(), frozenset())

    @pytest.mark.cuda(2)
    def test_a_node_reads_the_cuda_devices_of_its_peers_but_its_own(self):
        devices = ("cuda:0", "cuda:1", "cpu")
        sources = [probe_on(d, out=Port(send=vec())) for d in devices]
        sink = probe_on("cuda:1", inp=MultiPort(recv=vec()))
        on_cpu = probe_on("cpu", inp=Port(recv=vec()))
        for source in sources:
            source.out.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        sources[0].out.connect(on_cpu.inp)  # pyright: ignore[reportAttributeAccessIssue]
        a, b, c = sources
        net = Group(a=a, b=b, c=c, sink=sink, on_cpu=on_cpu)
        cuda0 = frozenset({torch.device("cuda", 0)})
        assert wired(net).read_devices == (frozenset(),) * 3 + (cuda0, cuda0)

    @pytest.mark.cuda
    def test_what_the_host_writes_is_in_host_memory(self):
        # The host's thread may have another current device and stream.
        node = probe_on("cuda:0", inp=Port(recv=vec()))
        wiring = wired(Group(node=node))
        x = torch.ones(2, dtype=F64, device="cuda:0")
        wiring.write({node.inp: x})  # pyright: ignore[reportAttributeAccessIssue]
        (value,) = wiring.inputs[node.inp].read().values  # pyright: ignore[reportAttributeAccessIssue]
        assert isinstance(value, BufferView)
        assert torch.equal(torch.from_dlpack(value), x.cpu())

    @pytest.mark.cuda
    def test_the_outbox_and_every_channel_hold_one_clone_on_the_device(self):
        out = Port(send=vec())
        source = probe_on("cuda:0", out=out)
        sinks = {f"s{k}": Probe(inp=Port(recv=vec())) for k in range(3)}
        for sink in sinks.values():
            out.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        wiring = wired(Group(source=source, **sinks))
        x = torch.ones(2, dtype=F64, device="cuda:0")
        wiring.wires[0][out].send(x)
        outlet = wiring.outlets[out]
        (held,) = outlet.outbox.read().values
        assert isinstance(held, torch.Tensor)
        assert held.device == torch.device("cuda", 0)
        assert held.data_ptr() != x.data_ptr()
        for k in range(3):
            assert outlet.channel(k).read().values[0] is held
