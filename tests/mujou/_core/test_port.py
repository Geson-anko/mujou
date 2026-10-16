from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, TypedDict, assert_type, cast, override

import pytest
import torch
from mujou._network import Network
from mujou._node import Node
from mujou._port import Channel, Multi, NextPort, Port, PrevPort, Single
from mujou._spec import DictSpec, TensorSpec

from mujou._core import GlobalTimer

if TYPE_CHECKING:
    from tests.mujou.conftest import RunNetwork


class Feedback(TypedDict):
    error: torch.Tensor


class Other(TypedDict):
    error: torch.Tensor


def h() -> TensorSpec:
    return TensorSpec((2, 3))


def fb(typed_dict: type[Feedback] = Feedback) -> DictSpec[Feedback]:
    return DictSpec(typed_dict, error=TensorSpec((2, 3)))


class TestPortTypes:
    def test_peer_count_fixes_the_recv_type_statically(self):
        # Checked by pyright (assert_type); at runtime only the defaults are seen.
        single = Port(send=h(), recv=h())
        multi = Port(send=h(), recv=fb(), multi=True)
        prev = PrevPort(recv=h())
        prev_multi = PrevPort(recv=h(), multi=True)
        nxt = NextPort(send=h(), recv=fb())
        nxt_single = NextPort(send=h(), multi=False)
        assert_type(single, Port[torch.Tensor, torch.Tensor, Single])
        assert_type(multi, Port[torch.Tensor, Feedback, Multi])
        assert_type(prev, PrevPort[Never, torch.Tensor, Single])
        assert_type(prev_multi, PrevPort[Never, torch.Tensor, Multi])
        assert_type(nxt, NextPort[torch.Tensor, Feedback, Multi])
        assert_type(nxt_single, NextPort[torch.Tensor, Never, Single])

        def recv_types() -> None:
            # Only for the type checker: recv raises before a run.
            assert_type(single.recv(), torch.Tensor)
            assert_type(multi.recv(), dict[Channel, Feedback])
            assert_type(nxt.recv_with_time(), dict[Channel, tuple[Feedback, int]])
            assert_type(prev.recv_with_time(), tuple[torch.Tensor, int])

        assert not single.multi and not prev.multi and not nxt_single.multi
        assert multi.multi and prev_multi.multi and nxt.multi


class TestPortSpecs:
    @pytest.mark.parametrize("spec", [(2, 3), torch.zeros(3), TensorSpec])
    def test_rejects_what_is_not_a_spec(self, spec: object):
        with pytest.raises(TypeError, match="TensorSpec, a DictSpec or None"):
            Port(send=spec)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]
        with pytest.raises(TypeError, match="PrevPort recv"):
            PrevPort(recv=spec)  # pyright: ignore[reportArgumentType, reportUnknownVariableType]


class TestConnect:
    def test_registers_the_channel_at_both_ends_in_order(self):
        a = NextPort(send=h(), recv=fb())
        b = PrevPort(send=fb(), recv=h())
        c = PrevPort(send=fb(), recv=h())
        ab = a.connect(b)
        ca = c.connect(a)
        assert a.channels == (ab, ca)
        assert b.channels == (ab,)
        assert c.channels == (ca,)
        assert ab.ends == (a, b)
        assert ab.other(a) is b and ab.other(b) is a
        assert ca.ends == (c, a)

    def test_two_channels_between_the_same_ports_are_distinct(self):
        a = Port(send=h(), multi=True)
        b = Port(recv=h(), multi=True)
        first, second = a.connect(b), a.connect(b)
        assert first is not second
        assert {first: 1, second: 2}[first] == 1
        assert a.channels == (first, second)

    def test_one_way_ports_connect(self):
        sender = Port(send=h())
        receiver = Port(recv=h())
        ch = sender.connect(receiver)
        assert ch.ends == (sender, receiver)
        assert sender.recv_spec is None and receiver.send_spec is None

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # shape
            (Port(send=h()), Port(recv=TensorSpec((3, 2)))),
            # dtype
            (Port(send=h()), Port(recv=TensorSpec((2, 3), torch.float64))),
            # message type
            (Port(send=h()), Port(recv=fb())),
            # TypedDict type
            (Port(send=fb()), Port(recv=fb(Other))),  # pyright: ignore[reportArgumentType]
            # only one direction matches
            (Port(send=h(), recv=h()), Port(recv=h())),
            # same direction on both sides
            (Port(send=h()), Port(send=h())),
        ],
    )
    def test_rejects_specs_that_are_not_reversed(
        self, a: Port[Any, Any, Any], b: Port[Any, Any, Any]
    ):
        with pytest.raises(TypeError, match="do not match"):
            a.connect(b)
        assert a.channels == () and b.channels == ()

    def test_rejects_ports_without_specs(self):
        with pytest.raises(TypeError, match="nothing would flow"):
            Port().connect(Port())

    def test_rejects_itself(self):
        a = Port(send=h(), recv=h(), multi=True)
        with pytest.raises(ValueError, match="itself"):
            a.connect(a)

    @pytest.mark.parametrize("second_on_single", [True, False])
    def test_rejects_a_second_peer_on_a_single_port(self, second_on_single: bool):
        single = PrevPort(recv=h())
        single.connect(NextPort(send=h()))
        other = NextPort(send=h())
        with pytest.raises(ValueError, match="already connected"):
            if second_on_single:
                single.connect(other)
            else:
                other.connect(single)
        assert len(single.channels) == 1 and other.channels == ()

    def test_next_port_takes_many_peers_by_default(self):
        a = NextPort(send=h())
        for _ in range(3):
            a.connect(PrevPort(recv=h()))
        assert len(a.channels) == 3

    def test_multi_prev_port_takes_many_peers(self):
        merge = PrevPort(recv=h(), multi=True)
        merge.connect(NextPort(send=h()))
        merge.connect(NextPort(send=h()))
        assert len(merge.channels) == 2


class TestChannel:
    def test_other_rejects_a_port_that_is_not_an_end(self):
        ch = Port(send=h()).connect(Port(recv=h()))
        with pytest.raises(ValueError, match="not an end"):
            ch.other(Port(send=h()))


class TestRecvSendBeforeRun:
    def test_raise_until_the_network_runs(self):
        a = Port(send=h(), recv=h())
        ch = a.connect(Port(send=h(), recv=h()))
        for call in (
            a.recv,
            a.recv_with_time,
            lambda: a.send(torch.zeros(2, 3)),
            lambda: a.send_to(ch, torch.zeros(2, 3)),
        ):
            with pytest.raises(RuntimeError, match="not bound"):
                call()


H = 0.001  # The step of the virtual clock, in seconds.


class Probe(Node):
    """Holds the given ports and, in each ``on_update``, keeps what ``update``
    returns (or the exception it raised)."""

    def __init__(
        self,
        update: Callable[[], object] = lambda: None,
        **ports: Port[Any, Any, Any],
    ) -> None:
        for name, port in ports.items():
            setattr(self, name, port)
        self.update = update
        self.results: list[object] = []

    @override
    def on_update(self, dt: float) -> None:
        try:
            self.results.append(self.update())
        except Exception as e:  # noqa: BLE001 - kept for the test to inspect
            self.results.append(e)


class Net(Network):
    def __init__(self, **children: Node) -> None:
        for name, child in children.items():
            setattr(self, name, child)


def step(net: Network, timer: GlobalTimer, *order: Node) -> None:
    """Advance the virtual clock by ``H`` and update every node once."""
    timer.advance(H)
    net._step(order or None)  # pyright: ignore[reportPrivateUsage]


class TestRecv:
    def test_unconnected_ports_give_zeros_or_nothing(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        single, multi = Port(recv=h()), Port(recv=h(), multi=True)
        node = Probe(lambda: (single.recv(), multi.recv()), single=single, multi=multi)
        net = run_network(Net(node=node), _manual=True)
        step(net, virtual_timer)
        ((value, peers),) = cast(list[tuple[torch.Tensor, object]], node.results)
        assert torch.equal(value, torch.zeros(2, 3))
        assert peers == {}

    def test_many_peers_are_keyed_by_channel_in_connection_order(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        merge = PrevPort(recv=h(), multi=True)
        senders = [NextPort(send=h()) for _ in range(3)]
        channels = [s.connect(merge) for s in senders[::-1]]  # connect 2, 1, 0
        nodes = {
            f"s{k}": Probe(lambda s=s, k=k: s.send(torch.full((2, 3), float(k))), p=s)
            for k, s in enumerate(senders)
        }
        sink = Probe(merge.recv, prev=merge)
        net = run_network(Net(**nodes, sink=sink), _manual=True)
        step(net, virtual_timer, *nodes.values(), sink)
        (received,) = sink.results
        received = cast(dict[Channel, torch.Tensor], received)
        assert list(received) == channels == list(merge.channels)
        for channel, k in zip(channels, (2, 1, 0), strict=True):
            assert torch.equal(received[channel], torch.full((2, 3), float(k)))

    def test_gives_new_tensors_without_gradients(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out, inp, other = NextPort(send=h()), PrevPort(recv=h()), PrevPort(recv=h())
        out.connect(inp)
        out.connect(other)
        sent = torch.ones(2, 3, requires_grad=True)
        source = Probe(lambda: out.send(sent), next=out)

        def change_what_was_received() -> tuple[torch.Tensor, torch.Tensor]:
            first = inp.recv()
            first.add_(10.0)  # in place: must not reach the slot
            return first, inp.recv()

        sink = Probe(change_what_was_received, prev=inp)
        neighbour = Probe(other.recv, prev=other)
        net = run_network(Net(source=source, sink=sink, nb=neighbour), _manual=True)
        step(net, virtual_timer, source, sink, neighbour)
        ((first, again),) = cast(list[tuple[torch.Tensor, torch.Tensor]], sink.results)
        assert not first.requires_grad and not again.requires_grad
        assert first.device == torch.device("cpu")
        assert torch.equal(again, torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, neighbour.results[0]), torch.ones(2, 3))
        assert torch.equal(net.read(out)[0], torch.ones(2, 3))

    def test_with_time_gives_the_time_of_the_write(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)
        sink = Probe(inp.recv_with_time, prev=inp)
        net = run_network(Net(source=source, sink=sink), _manual=True)
        step(net, virtual_timer, source, sink)
        ((value, time_ns),) = cast(list[tuple[torch.Tensor, int]], sink.results)
        assert torch.equal(value, torch.ones(2, 3))
        assert time_ns == virtual_timer.time_ns()

    def test_only_the_owning_node_receives(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        mine, theirs = Port(recv=h()), Port(recv=h())
        owner = Probe(mine=mine)
        other = Probe(theirs.recv, theirs=Port(recv=h()))  # reads another's port
        net = run_network(
            Net(owner=Probe(mine=theirs), node=owner, other=other), _manual=True
        )
        step(net, virtual_timer)
        assert isinstance(other.results[0], RuntimeError)
        with pytest.raises(RuntimeError, match="only the node that owns"):
            mine.recv()  # from the host

    def test_rejects_a_port_that_receives_nothing(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out = Port(send=h())
        node = Probe(out.recv, out=out)
        net = run_network(Net(node=node), _manual=True)
        step(net, virtual_timer)
        assert isinstance(node.results[0], TypeError)


class TestSend:
    def test_copies_the_values(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out = Port(send=h())
        value = torch.zeros(2, 3)

        def send_then_change() -> None:
            out.send(value)
            value.add_(1.0)

        net = run_network(Net(node=Probe(send_then_change, out=out)), _manual=True)
        step(net, virtual_timer)
        assert torch.equal(net.read(out)[0], torch.zeros(2, 3))

    def test_to_one_channel_changes_only_that_peer(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out, a, b = NextPort(send=h()), PrevPort(recv=h()), PrevPort(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.recv, prev=a), Probe(b.recv, prev=b)
        net = run_network(Net(source=source, a=sink_a, b=sink_b), _manual=True)
        step(net, virtual_timer, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), torch.zeros(2, 3))
        # The output the host reads changes only with send to every peer.
        assert torch.equal(net.read(out)[0], torch.zeros(2, 3))

    def test_to_a_channel_of_another_port(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        foreign = Port(send=h()).connect(Port(recv=h()))
        out.connect(inp)
        node = Probe(lambda: out.send_to(foreign, torch.ones(2, 3)), next=out)
        net = run_network(Net(node=node, sink=Probe(prev=inp)), _manual=True)
        step(net, virtual_timer)
        assert isinstance(node.results[0], ValueError)

    def test_rejects_a_port_that_sends_nothing(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        inp = Port(recv=h())
        # The type checker rejects it too: the port's message type is Never.
        node = Probe(lambda: inp.send(torch.ones(2, 3)), inp=inp)  # pyright: ignore[reportArgumentType]
        net = run_network(Net(node=node), _manual=True)
        step(net, virtual_timer)
        assert isinstance(node.results[0], TypeError)

    def test_only_the_owning_node_sends(self, run_network: RunNetwork):
        out = Port(send=h())
        run_network(Net(node=Probe(out=out)), _manual=True)
        with pytest.raises(RuntimeError, match="only the node that owns"):
            out.send(torch.ones(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestCuda:
    def test_receives_on_the_device_and_sends_to_the_host(
        self, virtual_timer: GlobalTimer, run_network: RunNetwork
    ):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        out.connect(inp)

        class OnCuda(Probe):
            def __init__(self, update: Callable[[], object], **ports: Any) -> None:
                super().__init__(update, **ports)
                Node.__init__(self, device="cuda")

        source = OnCuda(lambda: out.send(torch.ones(2, 3, device="cuda")), next=out)
        sink = OnCuda(inp.recv, prev=inp)
        net = run_network(Net(source=source, sink=sink), _manual=True)
        step(net, virtual_timer, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        assert received.device.type == "cuda"
        assert torch.equal(received.cpu(), torch.ones(2, 3))
        assert torch.equal(net.read(out)[0], torch.ones(2, 3))
