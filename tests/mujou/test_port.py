import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, TypedDict, cast, override

import pytest
import torch

from mujou.network import Group
from mujou.node import Node
from mujou.port import Channel, MultiPort, Port
from mujou.spec import DictSpec, IntSpec, TensorSpec
from mujou.time import Timestamp, VirtualTimer
from tests.mujou.nodes import Probe, probe_on

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual

H = 1e-3  # seconds per step
# The error of a port, or a node's timer, used outside its owner's hooks (K12).
OUTSIDE = "is used only in the hooks of its node while it runs"

type AnyPort = Port[Any, Any] | MultiPort[Any, Any]


class Feedback(TypedDict):
    error: torch.Tensor


class Other(TypedDict):
    error: torch.Tensor


class Tagged(TypedDict):
    x: torch.Tensor
    k: int


def h() -> TensorSpec:
    return TensorSpec((2, 3))


def fb(typed_dict: type[Feedback] = Feedback) -> DictSpec[Feedback]:
    return DictSpec(typed_dict, error=TensorSpec((2, 3)))


class TestConnect:
    def test_registers_the_channel_at_both_ends_in_order(self):
        a = MultiPort(send=h(), recv=fb())
        b = Port(send=fb(), recv=h())
        c = Port(send=fb(), recv=h())
        ab = a.connect(b)
        ca = c.connect(a)
        assert a.channels == (ab, ca)
        assert b.channels == (ab,)
        assert c.channels == (ca,)
        assert ab.other(a) is b and ab.other(b) is a
        assert ca.other(c) is a and ca.other(a) is c

    @pytest.mark.parametrize("make", [Port, MultiPort])
    def test_the_same_peer_again_gives_the_same_channel(self, make: type[AnyPort]):
        a, b = make(send=h(), recv=h()), make(send=h(), recv=h())
        ab = a.connect(b)
        assert a.connect(b) is ab
        assert b.connect(a) is ab
        assert a.channels == (ab,) and b.channels == (ab,)
        assert ab.other(a) is b

    @pytest.mark.parametrize("make", [Port, MultiPort])
    def test_a_port_connects_to_itself_once(self, make: type[AnyPort]):
        a = make(send=h(), recv=h())
        aa = a.connect(a)
        assert a.connect(a) is aa
        assert a.channels == (aa,)
        assert aa.other(a) is a

    def test_one_way_ports_connect(self):
        sender = Port(send=h())
        receiver = Port(recv=h())
        ch = sender.connect(receiver)
        assert ch.other(sender) is receiver
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
            # a MultiPort is checked alike
            (MultiPort(send=h()), Port(recv=fb())),
        ],
    )
    def test_rejects_specs_that_are_not_reversed(self, a: AnyPort, b: AnyPort):
        with pytest.raises(TypeError, match="do not match"):
            a.connect(b)
        assert a.channels == () and b.channels == ()

    @pytest.mark.parametrize("from_the_receiver", [True, False])
    def test_a_port_that_receives_takes_one_peer(self, from_the_receiver: bool):
        receiver = Port(recv=h())
        receiver.connect(Port(send=h()))
        other = MultiPort(send=h())
        with pytest.raises(ValueError, match="use a MultiPort to receive from several"):
            if from_the_receiver:
                receiver.connect(other)
            else:
                other.connect(receiver)
        assert len(receiver.channels) == 1 and other.channels == ()

    def test_a_port_that_only_sends_takes_any_number_of_peers(self):
        sender = Port(send=h())
        for _ in range(3):
            sender.connect(Port(recv=h()))
        assert len(sender.channels) == 3

    def test_a_multi_port_takes_any_number_of_peers(self):
        merge = MultiPort(recv=h(), send=h())
        merge.connect(Port(send=h(), recv=h()))
        merge.connect(MultiPort(send=h(), recv=h()))
        assert len(merge.channels) == 2


class TestPortNames:
    def test_a_port_is_shown_by_its_specs(self):
        assert repr(Port(send=h())) == f"Port(send={h()!r}, recv=None)"
        assert repr(MultiPort(recv=h())) == f"MultiPort(send=None, recv={h()!r})"

    def test_connect_errors_name_the_two_ends(self):
        a, b = Port(recv=h()), Port(recv=h())
        with pytest.raises(TypeError) as raised:
            a.connect(b)  # pyright: ignore[reportArgumentType]
        assert str(raised.value) == (
            f"port specs do not match: this port sends nothing, but the other "
            f"receives {h()!r}; the other sends nothing, but this port receives "
            f"{h()!r}"
        )
        Port(send=h()).connect(b)
        with pytest.raises(
            ValueError,
            match=r"^the other is a Port that receives and has a peer already; use a "
            "MultiPort to receive from several$",
        ):
            Port(send=h()).connect(b)


class TestChannel:
    def test_names_its_ends(self):
        a, b = Port(send=h()), Port(recv=h())
        assert repr(a.connect(b)) == f"Channel({a!r} <-> {b!r})"


def raises_outside(call: Callable[[], object]) -> RuntimeError | None:
    """What ``call`` raised if it was the error of K12, else ``None``."""
    try:
        call()
    except RuntimeError as e:
        return e if OUTSIDE in str(e) else None
    return None


class TestOutsideTheHooks:
    def test_before_a_run(self):
        a = MultiPort(send=h(), recv=h())
        ch = a.connect(Port(send=h(), recv=h()))
        for call in (
            a.recv,
            a.recv_with_time,
            lambda: a.send(torch.zeros(2, 3)),
            lambda: a.send_to(ch, torch.zeros(2, 3)),
        ):
            assert raises_outside(call) is not None

    def test_names_the_port(self):
        port = Port(recv=h())
        with pytest.raises(RuntimeError, match=rf"^{re.escape(repr(port))} {OUTSIDE}"):
            port.recv()

    def test_from_the_host_while_it_runs(self, start_manual: StartManual):
        inp, out = Port(recv=h()), Port(send=h())
        start_manual(Group(node=Probe(inp=inp, out=out)))
        assert raises_outside(inp.recv) is not None
        assert raises_outside(lambda: out.send(torch.ones(2, 3))) is not None

    def test_from_another_node(self, start_manual: StartManual):
        theirs = Port(recv=h())
        owner = Probe(theirs=theirs)
        other = Probe(lambda: raises_outside(theirs.recv), own=Port(recv=h()))
        run = start_manual(Group(owner=owner, other=other))
        run.step(H)
        assert isinstance(other.results[0], RuntimeError)

    def test_a_port_made_in_a_hook(self, start_manual: StartManual):
        class Late(Node):
            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                self.error = raises_outside(Port(send=h()).recv_with_time)

        node = Late()
        run = start_manual(Group(node=node))
        run.step(H)
        assert isinstance(node.error, RuntimeError)

    def test_after_the_run(self, start_manual: StartManual):
        out = Port(send=h())
        run = start_manual(Group(node=Probe(out=out)))
        run.shutdown()
        assert raises_outside(lambda: out.send(torch.ones(2, 3))) is not None


class Signalling(Probe):
    """A ``Probe`` that also signals each update it finished."""

    def __init__(
        self,
        update: Callable[[], object] = lambda: None,
        **ports: AnyPort,
    ) -> None:
        super().__init__(update, **ports)
        self.updated: queue.Queue[None] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.updated.put(None)


class TestOwnerOnAWorker:
    def test_only_the_owning_node_reads_and_writes_on_a_shared_worker(
        self, virtual: VirtualTimer, start: Start
    ):
        # specs: the owner is "the node the thread runs now", not the thread: one
        # worker runs all three nodes, one after another.
        inp, out = Port(recv=h()), Port(send=h())
        owner = Signalling(inp.recv, inp=inp, out=out)
        reader = Signalling(inp.recv, own=Port(recv=h()))
        writer = Signalling(lambda: out.send(torch.ones(2, 3)), own=Port(send=h()))
        run = start(
            Group(owner=owner, reader=reader, writer=writer), timer=virtual, workers=1
        )
        for _ in range(2):
            virtual.advance(0.01)
            for node in (owner, reader, writer):
                node.updated.get(timeout=5.0)
        run.shutdown()
        assert all(
            torch.equal(cast(torch.Tensor, r), torch.zeros(2, 3)) for r in owner.results
        )
        for node in (reader, writer):
            assert len(node.results) == 2
            for result in node.results:
                assert isinstance(result, RuntimeError)
                assert OUTSIDE in str(result)
        assert torch.equal(run.read(out), torch.zeros(2, 3))
        assert len(owner.results) == 2


class TestRecv:
    def test_unconnected_ports_give_zeros_or_nothing(self, start_manual: StartManual):
        single, multi = Port(recv=h()), MultiPort(recv=h())
        node = Probe(lambda: (single.recv(), multi.recv()), single=single, multi=multi)
        run = start_manual(Group(node=node))
        run.step(H)
        ((value, peers),) = cast(list[tuple[torch.Tensor, object]], node.results)
        assert torch.equal(value, torch.zeros(2, 3))
        assert peers == {}

    def test_many_peers_are_keyed_by_channel_in_connection_order(
        self, start_manual: StartManual
    ):
        merge = MultiPort(recv=h())
        senders = [Port(send=h()) for _ in range(3)]
        channels = [s.connect(merge) for s in senders[::-1]]  # connect 2, 1, 0
        nodes = {
            f"s{k}": Probe(lambda s=s, k=k: s.send(torch.full((2, 3), float(k))), p=s)
            for k, s in enumerate(senders)
        }
        sink = Probe(merge.recv, prev=merge)
        run = start_manual(Group(**nodes, sink=sink))
        run.step(H, *nodes.values(), sink)
        (received,) = sink.results
        received = cast(dict[Channel, torch.Tensor], received)
        assert list(received) == channels == list(merge.channels)
        for channel, k in zip(channels, (2, 1, 0), strict=True):
            assert torch.equal(received[channel], torch.full((2, 3), float(k)))

    def test_gives_new_tensors_without_gradients(self, start_manual: StartManual):
        out, inp, other = Port(send=h()), Port(recv=h()), Port(recv=h())
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
        run = start_manual(Group(source=source, sink=sink, nb=neighbour))
        run.step(H, source, sink, neighbour)
        ((first, again),) = cast(list[tuple[torch.Tensor, torch.Tensor]], sink.results)
        assert not first.requires_grad and not again.requires_grad
        assert first.device == torch.device("cpu")
        assert torch.equal(again, torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, neighbour.results[0]), torch.ones(2, 3))
        assert torch.equal(run.read(out), torch.ones(2, 3))

    def test_with_time_gives_the_time_of_the_write(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        out, inp = Port(send=h()), Port(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)
        sink = Probe(inp.recv_with_time, prev=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        ((value, written_at),) = cast(
            list[tuple[torch.Tensor, Timestamp]], sink.results
        )
        assert torch.equal(value, torch.ones(2, 3))
        assert written_at == virtual.now()

    def test_a_port_connected_to_itself_receives_what_it_sent(
        self, start_manual: StartManual
    ):
        loop = Port(send=h(), recv=h())
        loop.connect(loop)

        def count() -> torch.Tensor:
            x = loop.recv()
            loop.send(x + 1)
            return x

        node = Probe(count, loop=loop)
        run = start_manual(Group(node=node))
        for _ in range(3):
            run.step(H)
        firsts = [float(cast(torch.Tensor, x)[0, 0]) for x in node.results]
        assert firsts == [0.0, 1.0, 2.0]


def written(port: Port[Any, torch.Tensor]) -> bool:
    """Whether the writer of ``port`` has written what ``recv`` returns."""
    _, time = port.recv_with_time()
    return time is not None


def written_by_channel(port: MultiPort[Any, torch.Tensor]) -> dict[Channel, bool]:
    return {ch: time is not None for ch, (_, time) in port.recv_with_time().items()}


class TestNoTimeUntilWritten:
    def test_the_initial_value_has_no_time_until_sent(self, start_manual: StartManual):
        out = Port(send=h(), initial=torch.ones(2, 3))
        inp = Port(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.zeros(2, 3)), next=out)
        sink = Probe(lambda: written(inp), prev=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, sink)
        run.step(H, source, sink)  # sends the zeros: a value equal to the initial one
        assert sink.results == [False, True]

    def test_many_peers_are_keyed_by_channel_as_recv_is(
        self, start_manual: StartManual
    ):
        merge = MultiPort(recv=h())
        quiet, talker = Port(send=h()), Port(send=h())
        to_quiet = quiet.connect(merge)
        to_talker = talker.connect(merge)
        nodes = {
            "quiet": Probe(next=quiet),
            "talker": Probe(lambda: talker.send(torch.ones(2, 3)), next=talker),
        }
        sink = Probe(lambda: written_by_channel(merge), prev=merge)
        unconnected = MultiPort(recv=h())
        lonely = Probe(lambda: written_by_channel(unconnected), prev=unconnected)
        run = start_manual(Group(**nodes, sink=sink, lonely=lonely))
        run.step(H, *nodes.values(), sink, lonely)
        assert sink.results == [{to_quiet: False, to_talker: True}]
        assert lonely.results == [{}]

    def test_send_to_gives_a_time_only_for_that_peer(self, start_manual: StartManual):
        out = MultiPort(send=h())
        a, b = Port(recv=h()), Port(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a = Probe(lambda: written(a), prev=a)
        sink_b = Probe(lambda: written(b), prev=b)
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        run.step(H, source, sink_a, sink_b)
        assert (sink_a.results, sink_b.results) == ([True], [False])

    def test_what_the_host_writes_has_its_time(self, start_manual: StartManual):
        given, later = Port(recv=h()), Port(recv=h())
        node = Probe(
            lambda: (given.recv_with_time()[1], later.recv_with_time()[1]),
            given=given,
            later=later,
        )
        run = start_manual(Group(node=node), inputs={given: torch.ones(2, 3)})
        run.step(H)
        written_at = run.write({later: torch.ones(2, 3)})
        run.step(H)
        start = Timestamp(ns=0)
        assert node.results == [(start, None), (start, written_at)]

    @pytest.mark.parametrize("with_inputs", [False, True])
    def test_a_restored_run_keeps_what_was_not_written(
        self, start_manual: StartManual, with_inputs: bool
    ):
        def build() -> tuple[Group, Probe, Probe, Port[Never, torch.Tensor]]:
            out = MultiPort(send=h(), initial=torch.ones(2, 3))
            a, b = Port(recv=h()), Port(recv=h())
            to_a = out.connect(a)
            out.connect(b)
            host = Port(recv=h())
            source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
            sink = Probe(
                lambda: (written(a), written(b), written(host)),
                a=a,
                b=b,
                host=host,
            )
            return Group(source=source, sink=sink), source, sink, host

        net, source, _, _ = build()
        run = start_manual(net)
        run.step(H, source)  # sends to a only; nothing is written to host
        tree = run.state_dict()
        run.shutdown()

        restored, _, sink, host = build()
        inputs = {host: torch.ones(2, 3)} if with_inputs else None
        run = start_manual(restored, state=tree, inputs=inputs)
        run.step(H, sink)
        assert sink.results == [(True, False, with_inputs)]


class TestSend:
    def test_copies_the_values(self, start_manual: StartManual):
        out = Port(send=h())
        value = torch.zeros(2, 3)

        def send_then_change() -> None:
            out.send(value)
            value.add_(1.0)

        run = start_manual(Group(node=Probe(send_then_change, out=out)))
        run.step(H)
        assert torch.equal(run.read(out), torch.zeros(2, 3))

    @pytest.mark.parametrize("make", [Port, MultiPort])
    def test_reaches_every_peer(self, start_manual: StartManual, make: type[AnyPort]):
        out = make(send=h())
        sinks = {f"s{k}": Port(recv=h()) for k in range(3)}
        for inp in sinks.values():
            out.connect(inp)
        nodes = {name: Probe(inp.recv, prev=inp) for name, inp in sinks.items()}
        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)
        run = start_manual(Group(source=source, **nodes))
        run.step(H)
        for node in nodes.values():
            assert torch.equal(cast(torch.Tensor, node.results[0]), torch.ones(2, 3))
        assert torch.equal(run.read(out), torch.ones(2, 3))

    def test_to_one_channel_changes_only_that_peer(self, start_manual: StartManual):
        out, a, b = MultiPort(send=h()), Port(recv=h()), Port(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.recv, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        run.step(H, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), torch.zeros(2, 3))
        # The output the host reads changes only with send to every peer.
        assert torch.equal(run.read(out), torch.zeros(2, 3))

    def test_rejects_a_tensor_that_is_not_dense(self, start_manual: StartManual):
        out = Port(send=TensorSpec((2, 2)))
        node = Probe(lambda: out.send(torch.eye(2).to_sparse()), out=out)
        run = start_manual(Group(node=node))
        run.step(H)
        assert isinstance(node.results[0], BufferError)
        assert torch.equal(run.read(out), torch.zeros(2, 2))

    def test_to_a_channel_of_another_port(self, start_manual: StartManual):
        out, inp = MultiPort(send=h()), Port(recv=h())
        foreign = Port(send=h()).connect(Port(recv=h()))
        out.connect(inp)
        node = Probe(lambda: out.send_to(foreign, torch.ones(2, 3)), next=out)
        run = start_manual(Group(node=node, sink=Probe(prev=inp)))
        run.step(H)
        assert isinstance(node.results[0], KeyError)


class TestInitial:
    def test_every_peer_and_the_host_read_it_until_the_first_send(
        self, start_manual: StartManual
    ):
        twos = torch.full((2, 3), 2.0)
        out = Port(send=h(), initial=twos)
        a, b = Port(recv=h()), Port(recv=h())
        out.connect(a)
        out.connect(b)
        sink_a, sink_b = Probe(a.recv_with_time, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Group(source=Probe(next=out), a=sink_a, b=sink_b))
        run.step(H, sink_a, sink_b)  # the source does not send
        ((value, written_at),) = cast(
            list[tuple[torch.Tensor, Timestamp | None]], sink_a.results
        )
        assert torch.equal(value, twos)
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), twos)
        assert written_at is None  # as the zeros: the port has not sent it
        assert torch.equal(run.read(out), twos)

    def test_send_to_replaces_it_for_that_peer_only(self, start_manual: StartManual):
        twos = torch.full((2, 3), 2.0)
        out = MultiPort(send=h(), initial=twos)
        a, b = Port(recv=h()), Port(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.recv, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        run.step(H, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), twos)
        assert torch.equal(run.read(out), twos)

    def test_of_a_dict_message(self, start_manual: StartManual):
        out = Port(send=fb(), initial={"error": torch.ones(2, 3)})
        inp = Port(recv=fb())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Group(source=Probe(out=out), sink=sink))
        run.step(H, sink)
        received = cast(Feedback, sink.results[0])
        assert torch.equal(received["error"], torch.ones(2, 3))

    def test_is_copied_when_the_run_starts(self, start_manual: StartManual):
        value = torch.ones(2, 3, requires_grad=True)
        out = Port(send=h(), initial=value)
        run = start_manual(Group(node=Probe(out=out)))
        with torch.no_grad():
            value.add_(1.0)  # after the start: must not reach the slots
        received = run.read(out)
        assert torch.equal(received, torch.ones(2, 3))
        assert not received.requires_grad

    def test_a_restored_run_takes_the_value_from_the_state(
        self, start_manual: StartManual
    ):
        def build(initial: float) -> tuple[Group, Probe, Probe]:
            out = Port(send=h(), initial=torch.full((2, 3), initial))
            inp = Port(recv=h())
            out.connect(inp)
            source = Probe(lambda: out.send(torch.full((2, 3), 5.0)), next=out)
            sink = Probe(inp.recv, prev=inp)
            return Group(source=source, sink=sink), source, sink

        net, source, _ = build(2.0)
        run = start_manual(net)
        run.step(H, source)  # the source sends 5
        tree = run.state_dict()
        run.shutdown()

        restored, _, sink = build(7.0)
        run = start_manual(restored, state=tree)
        run.step(H, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]), torch.full((2, 3), 5.0))

    def test_a_run_rejects_a_value_that_is_not_a_message_by_its_path(
        self, start_manual: StartManual
    ):
        tensor = Probe(next=Port(send=h(), initial=[[0.0] * 3] * 2))  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match=r"^node\.next: expected a torch\.Tensor"):
            start_manual(Group(node=tensor))
        keys = Probe(next=Port(send=fb(), initial={"loss": torch.zeros(2, 3)}))  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match=r"^node\.next: expected a dict with"):
            start_manual(Group(node=keys))

    @pytest.mark.parametrize(
        ("value", "what"),
        [
            (torch.zeros(3, 2), "shape"),
            (torch.zeros(2, 3, dtype=torch.float64), "dtype"),
        ],
    )
    def test_a_run_rejects_a_value_of_another_shape_or_dtype_by_its_path(
        self, start_manual: StartManual, value: torch.Tensor, what: str
    ):
        # The core checks the shape and the dtype when the run writes the value
        # to the port's slots, before any node starts.
        node = Probe(next=Port(send=h(), initial=value))
        with pytest.raises(
            ValueError, match=rf"^slot 'node\.next/out': expected {what}"
        ):
            start_manual(Group(node=node))


# Steps a node on argv[1] that sends to a node on argv[2] under torch.profiler,
# and prints the name of each copy on a GPU.
_PROFILED = """
import sys
import threading
from typing import override

import torch
from torch.profiler import ProfilerActivity, profile

import mujou.testing
from mujou import Group, Node, Port, TensorSpec

SPEC = TensorSpec((1024,))


class Source(Node):
    def __init__(self) -> None:
        super().__init__(device=sys.argv[1])
        self.out = Port(send=SPEC)

    @override
    def on_update(self, dt: float) -> None:
        self.out.send(torch.ones(1024, device=self.device))


class Sink(Node):
    def __init__(self) -> None:
        super().__init__(device=sys.argv[2])
        self.inp = Port(recv=SPEC)

    @override
    def on_update(self, dt: float) -> None:
        self.inp.recv()


source, sink = Source(), Sink()
source.out.connect(sink.inp)
run = mujou.testing.run(Group(source=source, sink=sink))
run.step(0.001, source, sink)
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        run.step(0.001, source, sink)
run.shutdown()
for event in prof.events():
    if event.name.startswith("Memcpy"):
        print(event.name)
"""


def profiled_copies(source: str, sink: str) -> list[str]:
    """The copies on a GPU while a node on ``source`` sends to one on ``sink``,
    in a new interpreter: the profiler loads modules that enable the GIL."""
    done = subprocess.run(
        [sys.executable, "-c", _PROFILED, source, sink],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return done.stdout.splitlines()


BIG = TensorSpec((1 << 20,))


def delay_the_stream() -> None:
    """Keep the current stream busy for about 0.1 s of GPU time, so that what
    is queued after it is still pending when the CPU moves on. ``_sleep`` is a
    private API of torch, which its own tests use for this."""
    torch.cuda._sleep(250_000_000)  # pyright: ignore[reportPrivateUsage]


class OnCuda0(Node):
    """A node on ``cuda:0`` on a thread of its own that, in ``on_start``, runs
    each kind of kernel the tests run: CUDA loads the code of a kernel when it
    first runs, and waits for the whole device to load it, which would hide a
    missing wait for a stream."""

    def __init__(self) -> None:
        super().__init__(device="cuda:0", dedicated_thread=True)

    @override
    def on_start(self) -> None:
        x = torch.empty(BIG.shape, device=self.device)
        x.fill_(1.0)
        x.clone().to(self.device, copy=True).cpu()
        torch.cuda._sleep(1)  # pyright: ignore[reportPrivateUsage]
        torch.cuda.current_stream().synchronize()


class Late(OnCuda0):
    """Sends 7s that its stream fills late, once, and then sets ``sent``."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=BIG)
        self.sent = threading.Event()

    @override
    def on_update(self, dt: float) -> None:
        if self.sent.is_set():
            return
        x = torch.empty(BIG.shape, device=self.device)
        delay_the_stream()
        x.fill_(7.0)
        self.out.send(x)
        self.sent.set()


class Resends(OnCuda0):
    """Sends 1s and sets ``first``, then, once ``read`` is set, sends 2s, 3s and
    4s and sets ``again``: once."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=BIG)
        self.first, self.read, self.again = (threading.Event() for _ in range(3))

    @override
    def on_update(self, dt: float) -> None:
        if self.first.is_set():
            return
        self.out.send(torch.ones(BIG.shape, device=self.device))
        self.first.set()
        assert self.read.wait(10.0)
        for k in (2.0, 3.0, 4.0):
            self.out.send(torch.full(BIG.shape, k, device=self.device))
        self.again.set()


class Reads(OnCuda0):
    """Once ``ready`` is set: calls ``before``, receives one message, calls
    ``between`` and keeps the message on the CPU in ``received``. Once."""

    def __init__(
        self, ready: threading.Event, before: Callable[[], object] = lambda: None
    ) -> None:
        super().__init__()
        self.inp = Port(recv=BIG)
        self.ready = ready
        self.before = before
        self.between: Callable[[], object] = lambda: None
        self.received: torch.Tensor | None = None
        self.done = threading.Event()

    @override
    def on_update(self, dt: float) -> None:
        if self.done.is_set():
            return
        assert self.ready.wait(10.0)
        self.before()
        received = self.inp.recv()
        self.between()
        self.received = received.cpu()
        self.done.set()


@pytest.mark.cuda
class TestCuda:
    def test_receives_on_the_device_and_sends_to_the_host(
        self, start_manual: StartManual
    ):
        out, inp, back = Port(send=h()), Port(recv=h()), Port(send=h())
        out.connect(inp)

        def update() -> torch.Tensor:
            x = inp.recv()
            back.send(2 * x)
            return x

        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)

        class OnCuda(Probe):
            on_device = "cuda"

        sink = OnCuda(update, prev=inp, back=back)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        # The index-less device is fixed, so the message compares equal to it.
        assert received.device == sink.device == torch.device("cuda", 0)
        assert torch.equal(received.cpu(), torch.ones(2, 3))
        sent = run.read(back)
        assert sent.device == torch.device("cpu")
        assert torch.equal(sent, torch.full((2, 3), 2.0))

    @pytest.mark.parametrize(
        "reader", ["cuda:0", pytest.param("cuda:1", marks=pytest.mark.cuda(2)), "cpu"]
    )
    def test_a_message_of_a_cuda_node_arrives_on_the_readers_device(
        self, start_manual: StartManual, reader: str
    ):
        out, inp = Port(send=h()), Port(recv=h())
        out.connect(inp)
        x = torch.arange(6.0).reshape(3, 2).t()  # (2, 3), not contiguous

        def send() -> None:
            w = torch.ones((), device="cuda:0", requires_grad=True)
            out.send(x.to("cuda:0") * w)

        source = probe_on("cuda:0", send, out=out)
        sink = probe_on(reader, inp.recv, inp=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        assert received.device == torch.device(reader)
        assert received.is_contiguous() and not received.requires_grad
        assert torch.equal(received.cpu(), x)
        assert torch.equal(run.read(out), x)

    def test_a_dict_mixes_a_tensor_on_the_device_and_scalars(
        self, start_manual: StartManual
    ):
        spec = DictSpec(Tagged, x=h(), k=IntSpec())
        out, inp = Port(send=spec), Port(recv=spec)
        out.connect(inp)
        msg = {"x": torch.ones(2, 3, device="cuda:0"), "k": 5}
        source = probe_on("cuda:0", lambda: out.send(msg), out=out)  # pyright: ignore[reportArgumentType]
        sink = probe_on("cuda:0", inp.recv, inp=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        received = cast(dict[str, Any], sink.results[0])
        assert received["k"] == 5
        assert received["x"].device == torch.device("cuda", 0)
        assert torch.equal(received["x"].cpu(), torch.ones(2, 3))

    def test_a_reader_changing_its_message_changes_no_one_elses(
        self, start_manual: StartManual
    ):
        out, a, b = Port(send=h()), Port(recv=h()), Port(recv=h())
        out.connect(a)
        out.connect(b)

        def send() -> None:
            x = torch.ones(2, 3, device="cuda:0")
            out.send(x)
            x.add_(1.0)  # after the send: must not reach the message

        source = probe_on("cuda:0", send, out=out)
        changes = probe_on("cuda:0", lambda: a.recv().add_(1.0), inp=a)
        reads = probe_on("cuda:0", b.recv, inp=b)
        run = start_manual(Group(source=source, changes=changes, reads=reads))
        run.step(H, source, changes, reads)
        ones = torch.ones(2, 3)
        assert torch.equal(cast(torch.Tensor, changes.results[0]).cpu(), 2 * ones)
        assert torch.equal(cast(torch.Tensor, reads.results[0]).cpu(), ones)
        assert torch.equal(run.read(out), ones)

    def test_send_to_reaches_only_its_peer(self, start_manual: StartManual):
        out, a, b = MultiPort(send=h()), Port(recv=h()), Port(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        ones = torch.ones(2, 3, device="cuda:0")
        source = probe_on("cuda:0", lambda: out.send_to(to_a, ones), out=out)
        sink_a = probe_on("cuda:0", a.recv, inp=a)
        sink_b = probe_on("cuda:0", b.recv, inp=b)
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        run.step(H, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]).cpu(), ones.cpu())
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]).cpu(), 0 * ones.cpu())
        assert torch.equal(run.read(out), torch.zeros(2, 3))

    def test_rejects_a_tensor_that_is_not_dense(self, start_manual: StartManual):
        out = Port(send=TensorSpec((2, 2)))
        sparse = torch.eye(2, device="cuda:0").to_sparse()
        node = probe_on("cuda:0", lambda: out.send(sparse), out=out)
        run = start_manual(Group(node=node))
        run.step(H)
        assert isinstance(node.results[0], BufferError)

    @pytest.mark.cuda(2)
    def test_a_tensor_on_another_device_arrives(self, start_manual: StartManual):
        out, inp = Port(send=h()), Port(recv=h())
        out.connect(inp)
        x = torch.ones(2, 3, device="cuda:1")
        source = probe_on("cuda:0", lambda: out.send(x), out=out)
        sink = probe_on("cuda:0", inp.recv, inp=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]).cpu(), x.cpu())

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("source", "sink", "via_the_host"),
        [
            ("cuda:0", "cuda:0", False),
            pytest.param("cuda:0", "cuda:1", False, marks=pytest.mark.cuda(2)),
            # The control: the oracle sees a copy from host memory.
            ("cpu", "cuda:0", True),
        ],
    )
    def test_a_message_between_cuda_nodes_skips_host_memory(
        self, source: str, sink: str, via_the_host: bool
    ):
        copies = profiled_copies(source, sink)
        assert copies, "the profiler saw no copy"
        # "Memcpy HtoD (Pageable -> Device)": the ends are in the parentheses. A
        # copy between GPUs without peer access is "Memcpy DtoH (Device ->
        # Device)", which the driver may stage through the host by itself.
        ends = {
            end for name in copies for end in re.findall(r"\w+", name.split("(")[-1])
        }
        assert bool(ends & {"Pageable", "Pinned"}) == via_the_host, copies

    @pytest.mark.slow
    def test_a_reader_gets_what_a_late_stream_wrote(
        self, virtual: VirtualTimer, start: Start
    ):
        # The reader copies the message the moment the writer has sent it,
        # while the writer is still in its on_update: send waits for the stream.
        writer = Late()
        reader = Reads(writer.sent)
        writer.out.connect(reader.inp)
        start(Group(writer=writer, reader=reader), timer=virtual)
        virtual.advance(0.01)
        assert reader.done.wait(10.0)
        assert torch.equal(
            cast(torch.Tensor, reader.received), torch.full(BIG.shape, 7.0)
        )

    @pytest.mark.slow
    def test_a_copy_still_queued_keeps_its_message_from_reuse(
        self, virtual: VirtualTimer, start: Start
    ):
        # The reader's copy waits on its stream while the writer replaces the
        # message: the memory of the 1s goes back to the caching allocator, and
        # the next message of its size would take it, before the copy has run.
        writer = Resends()
        reader = Reads(writer.first, before=delay_the_stream)
        reader.between = lambda: (writer.read.set(), writer.again.wait(10.0))
        writer.out.connect(reader.inp)
        start(Group(writer=writer, reader=reader), timer=virtual)
        virtual.advance(0.01)
        assert reader.done.wait(10.0)
        assert torch.equal(cast(torch.Tensor, reader.received), torch.ones(BIG.shape))

    def test_takes_an_initial_value_on_the_device(self, start_manual: StartManual):
        out = Port(send=h(), initial=torch.ones(2, 3, device="cuda:0"))
        inp = Port(recv=h())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Group(source=Probe(out=out), sink=sink))
        run.step(H, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]), torch.ones(2, 3))
