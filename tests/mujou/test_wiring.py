from typing import TypedDict

import pytest
import torch

from mujou._native import BufferView
from mujou.codec import peek
from mujou.network import Group, Network
from mujou.port import MultiPort, Port
from mujou.spec import DictSpec, TensorSpec
from mujou.time import ReadOnlyTimer, VirtualTimer
from mujou.tree import walk
from mujou.wiring import Wiring, wire
from tests.mujou.nodes import F64, Chain, Probe, probe_on, vec


class Both(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


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
        # The root node's ports are named by its class.
        node = Probe(out=Port(send=vec()), inp=Port(recv=vec()))
        wiring = wired(node)
        assert wiring.outlets[node.out].outbox.name == "Probe.out/out"  # pyright: ignore[reportAttributeAccessIssue]
        assert wiring.inputs[node.inp].name == "Probe.inp/in"  # pyright: ignore[reportAttributeAccessIssue]


class TestWrite:
    def test_a_cpu_tensor_is_held_in_host_memory(self):
        node = Probe(inp=Port(recv=vec()), scalar=Port(recv=TensorSpec(())))
        wiring = wired(Group(node=node))
        x = torch.ones(2, dtype=F64)
        wiring.write({node.inp: x, node.scalar: torch.tensor(1.0)})  # pyright: ignore[reportAttributeAccessIssue]
        for port, expected in ((node.inp, x), (node.scalar, torch.tensor(1.0))):  # pyright: ignore[reportAttributeAccessIssue]
            (value,) = wiring.inputs[port].read().values
            assert isinstance(value, BufferView)
            assert torch.equal(torch.from_dlpack(value), expected)


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

    @pytest.mark.cuda(2)
    def test_what_the_host_writes_stays_only_on_the_device_of_its_node(self):
        # One write to the ports of a node on cuda:0 and of a node on the CPU,
        # with tensors on two devices, while the host's current device is
        # cuda:1. Only a tensor on cuda:0 for the node on cuda:0 stays.
        spec = DictSpec(Both, x=vec(), y=vec())
        on_cuda = probe_on("cuda:0", a=Port(recv=vec()), b=Port(recv=spec))
        on_cpu = Probe(c=Port(recv=vec()))
        wiring = wired(Group(on_cuda=on_cuda, on_cpu=on_cpu))
        a = torch.ones(2, dtype=F64, device="cuda:0")
        b = {
            "x": torch.full((2,), 2.0, dtype=F64, device="cuda:0"),
            "y": torch.full((2,), 3.0, dtype=F64, device="cuda:1"),
        }
        c = torch.full((2,), 4.0, dtype=F64, device="cuda:0")
        ports = (on_cuda.a, on_cuda.b, on_cpu.c)  # pyright: ignore[reportAttributeAccessIssue]
        with torch.cuda.device(1):
            wiring.write(dict(zip(ports, (a, b, c), strict=True)))
        read_a, read_b, read_c = (
            peek(port.recv_spec, wiring.inputs[port].read().values) for port in ports
        )
        assert isinstance(read_a, torch.Tensor) and isinstance(read_c, torch.Tensor)
        assert isinstance(read_b, dict)
        for read, sent, stays in (
            (read_a, a, True),
            (read_b["x"], b["x"], True),
            (read_b["y"], b["y"], False),
            (read_c, c, False),
        ):
            assert read.device == (sent.device if stays else torch.device("cpu"))
            assert read.data_ptr() != sent.data_ptr()
            assert torch.equal(read.cpu(), sent.cpu())

    @pytest.mark.cuda
    def test_only_a_port_that_a_node_on_cuda_reads_keeps_the_message_there(self):
        # One wiring: a port without peers (only the host reads it), a port that
        # only a node on the CPU reads, and one that a node on CUDA reads too.
        lonely, to_cpu, mixed = Port(send=vec()), Port(send=vec()), Port(send=vec())
        source = probe_on("cuda:0", lonely=lonely, to_cpu=to_cpu, mixed=mixed)
        on_cpu = Probe(inp=Port(recv=vec()))
        to_cpu.connect(on_cpu.inp)  # pyright: ignore[reportAttributeAccessIssue]
        # One reader on CUDA among readers on the CPU is enough.
        sinks = {f"s{k}": Probe(inp=Port(recv=vec())) for k in range(2)}
        sinks["s2"] = probe_on("cuda:0", inp=Port(recv=vec()))
        for sink in sinks.values():
            mixed.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        wiring = wired(Group(source=source, on_cpu=on_cpu, **sinks))
        wires = wiring.wires[0]
        assert wires[lonely].cuda is None and wires[to_cpu].cuda is None
        # The node on the CPU reads its message in host memory, not on cuda:0.
        assert wiring.read_devices[1] == frozenset()
        x = torch.ones(2, dtype=F64, device="cuda:0")
        wires[to_cpu].send(x)
        for slot in (wiring.outlets[to_cpu].outbox, wiring.outlets[to_cpu].channel(0)):
            assert isinstance(slot.read().values[0], BufferView)
        # The outbox and every channel hold one clone on the device.
        wires[mixed].send(x)
        outlet = wiring.outlets[mixed]
        (held,) = outlet.outbox.read().values
        assert isinstance(held, torch.Tensor)
        assert held.device == torch.device("cuda", 0)
        assert held.data_ptr() != x.data_ptr()
        for k in range(3):
            assert outlet.channel(k).read().values[0] is held
