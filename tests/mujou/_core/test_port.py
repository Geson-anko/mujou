from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, TypedDict, assert_type, cast, override

import pytest
import torch

from mujou._core.network import Compose, Network
from mujou._core.node import Node
from mujou._core.port import Channel, Multi, NextPort, Port, PrevPort, Single
from mujou._core.spec import DictSpec, TensorSpec
from mujou.testing import VirtualTimer
from mujou.time import Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual, Step


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
            assert_type(nxt.recv_with_time(), dict[Channel, tuple[Feedback, Timestamp]])
            assert_type(prev.recv_with_time(), tuple[torch.Tensor, Timestamp])
            assert_type(single.has_received(), bool)
            assert_type(multi.has_received(), dict[Channel, bool])

        assert not single.multi and not prev.multi and not nxt_single.multi
        assert multi.multi and prev_multi.multi and nxt.multi

    def test_an_explicit_none_spec_is_a_static_error(self):
        # Omit a direction instead: an explicit None leaves its type unknown. Without
        # the ignore pyright reports the call; at runtime None is still accepted.
        port = Port(send=None)  # pyright: ignore[reportArgumentType]
        assert port.send_spec is None


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


class TestPortNames:
    class Shift(Node):
        def __init__(self) -> None:
            self.prev = PrevPort(recv=h())
            self.next = NextPort(send=h())
            self.alias = self.next  # a second name does not rename the port

    def test_a_bare_port_is_shown_by_its_specs(self):
        assert repr(Port(send=h())) == f"Port(send={h()!r}, recv=None, multi=False)"

    def test_a_port_of_a_node_is_named_by_its_first_owner_and_attribute(self):
        shift = self.Shift()

        class Exposing(Network):
            def __init__(self) -> None:
                self.node = shift
                self.next = shift.next

        Exposing()
        assert repr(shift.next) == (
            f"NextPort(Shift.next, send={h()!r}, recv=None, multi=True)"
        )

    def test_connect_errors_name_the_ports(self):
        a, b = self.Shift(), self.Shift()
        with pytest.raises(
            TypeError,
            match=r"^port specs do not match: Shift\.prev sends nothing, but "
            r"Shift\.prev receives TensorSpec",
        ):
            a.prev.connect(b.prev)
        with pytest.raises(ValueError, match=r"to itself: Shift\.next$"):
            a.next.connect(a.next)
        b.prev.connect(NextPort(send=h()))
        with pytest.raises(ValueError, match=r"^Shift\.prev takes one peer"):
            a.next.connect(b.prev)

    def test_the_error_before_a_run_names_the_port(self):
        shift = self.Shift()
        with pytest.raises(
            RuntimeError,
            match=r"^Shift\.prev can only send and receive in its owning node's hooks "
            r"while the network runs \(not in __init__\)$",
        ):
            shift.prev.recv()


class TestChannel:
    def test_is_made_only_by_connect(self):
        with pytest.raises(TypeError, match=r"Port\.connect"):
            Channel()

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
            with pytest.raises(RuntimeError, match="while the network runs"):
                call()


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


class TestRecv:
    def test_unconnected_ports_give_zeros_or_nothing(
        self, step: Step, start_manual: StartManual
    ):
        single, multi = Port(recv=h()), Port(recv=h(), multi=True)
        node = Probe(lambda: (single.recv(), multi.recv()), single=single, multi=multi)
        run = start_manual(Compose(node=node))
        step(run)
        ((value, peers),) = cast(list[tuple[torch.Tensor, object]], node.results)
        assert torch.equal(value, torch.zeros(2, 3))
        assert peers == {}

    def test_many_peers_are_keyed_by_channel_in_connection_order(
        self, step: Step, start_manual: StartManual
    ):
        merge = PrevPort(recv=h(), multi=True)
        senders = [NextPort(send=h()) for _ in range(3)]
        channels = [s.connect(merge) for s in senders[::-1]]  # connect 2, 1, 0
        nodes = {
            f"s{k}": Probe(lambda s=s, k=k: s.send(torch.full((2, 3), float(k))), p=s)
            for k, s in enumerate(senders)
        }
        sink = Probe(merge.recv, prev=merge)
        run = start_manual(Compose(**nodes, sink=sink))
        step(run, *nodes.values(), sink)
        (received,) = sink.results
        received = cast(dict[Channel, torch.Tensor], received)
        assert list(received) == channels == list(merge.channels)
        for channel, k in zip(channels, (2, 1, 0), strict=True):
            assert torch.equal(received[channel], torch.full((2, 3), float(k)))

    def test_gives_new_tensors_without_gradients(
        self, step: Step, start_manual: StartManual
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
        run = start_manual(Compose(source=source, sink=sink, nb=neighbour))
        step(run, source, sink, neighbour)
        ((first, again),) = cast(list[tuple[torch.Tensor, torch.Tensor]], sink.results)
        assert not first.requires_grad and not again.requires_grad
        assert first.device == torch.device("cpu")
        assert torch.equal(again, torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, neighbour.results[0]), torch.ones(2, 3))
        assert torch.equal(run.read(out)[0], torch.ones(2, 3))

    def test_with_time_gives_the_time_of_the_write(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)
        sink = Probe(inp.recv_with_time, prev=inp)
        run = start_manual(Compose(source=source, sink=sink))
        step(run, source, sink)
        ((value, written_at),) = cast(
            list[tuple[torch.Tensor, Timestamp]], sink.results
        )
        assert torch.equal(value, torch.ones(2, 3))
        assert written_at == virtual.now()

    def test_only_the_owning_node_receives(self, step: Step, start_manual: StartManual):
        mine, theirs = Port(recv=h()), Port(recv=h())
        owner = Probe(mine=mine)
        other = Probe(theirs.recv, theirs=Port(recv=h()))  # reads another's port
        run = start_manual(Compose(owner=Probe(mine=theirs), node=owner, other=other))
        step(run)
        assert isinstance(other.results[0], RuntimeError)
        with pytest.raises(RuntimeError, match="only the node that owns"):
            mine.recv()  # from the host

    def test_rejects_a_port_that_receives_nothing(
        self, step: Step, start_manual: StartManual
    ):
        out = Port(send=h())
        node = Probe(out.recv, out=out)
        run = start_manual(Compose(node=node))
        step(run)
        assert isinstance(node.results[0], TypeError)


class TestHasReceived:
    def test_is_false_for_the_initial_value_and_true_once_sent(
        self, step: Step, start_manual: StartManual
    ):
        out = NextPort(send=h(), initial=torch.ones(2, 3))
        inp = PrevPort(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.zeros(2, 3)), next=out)
        sink = Probe(inp.has_received, prev=inp)
        run = start_manual(Compose(source=source, sink=sink))
        step(run, sink)
        step(run, source, sink)  # sends the zeros: a value equal to the initial one
        assert sink.results == [False, True]

    def test_many_peers_are_keyed_by_channel_as_recv_is(
        self, step: Step, start_manual: StartManual
    ):
        merge = PrevPort(recv=h(), multi=True)
        quiet, talker = NextPort(send=h()), NextPort(send=h())
        to_quiet = quiet.connect(merge)
        to_talker = talker.connect(merge)
        nodes = {
            "quiet": Probe(next=quiet),
            "talker": Probe(lambda: talker.send(torch.ones(2, 3)), next=talker),
        }
        sink = Probe(merge.has_received, prev=merge)
        unconnected = Port(recv=h(), multi=True)
        lonely = Probe(unconnected.has_received, prev=unconnected)
        run = start_manual(Compose(**nodes, sink=sink, lonely=lonely))
        step(run, *nodes.values(), sink, lonely)
        assert sink.results == [{to_quiet: False, to_talker: True}]
        assert lonely.results == [{}]

    def test_send_to_counts_only_for_that_peer(
        self, step: Step, start_manual: StartManual
    ):
        out = NextPort(send=h())
        a, b = PrevPort(recv=h()), PrevPort(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.has_received, prev=a), Probe(b.has_received, prev=b)
        run = start_manual(Compose(source=source, a=sink_a, b=sink_b))
        step(run, source, sink_a, sink_b)
        assert (sink_a.results, sink_b.results) == ([True], [False])

    def test_counts_what_the_host_writes(self, step: Step, start_manual: StartManual):
        given, written = Port(recv=h()), Port(recv=h())
        node = Probe(
            lambda: (given.has_received(), written.has_received()),
            given=given,
            written=written,
        )
        run = start_manual(Compose(node=node), inputs={given: torch.ones(2, 3)})
        step(run)
        run.write({written: torch.ones(2, 3)})
        step(run)
        assert node.results == [(True, False), (True, True)]

    @pytest.mark.parametrize("with_inputs", [False, True])
    def test_a_restored_run_keeps_what_was_sent(
        self, step: Step, start_manual: StartManual, with_inputs: bool
    ):
        def build() -> tuple[Compose, Probe, Probe, Port[Never, torch.Tensor]]:
            out = NextPort(send=h(), initial=torch.ones(2, 3))
            a, b = Port(recv=h()), Port(recv=h())
            to_a = out.connect(a)
            out.connect(b)
            host = Port(recv=h())
            source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
            sink = Probe(
                lambda: (a.has_received(), b.has_received(), host.has_received()),
                a=a,
                b=b,
                host=host,
            )
            return Compose(source=source, sink=sink), source, sink, host

        net, source, _, _ = build()
        run = start_manual(net)
        step(run, source)  # sends to a only; nothing is written to host
        tree = run.state_dict()
        run.shutdown()

        restored, _, sink, host = build()
        inputs = {host: torch.ones(2, 3)} if with_inputs else None
        run = start_manual(restored, timer=VirtualTimer(), state=tree, inputs=inputs)
        step(run, sink)
        assert sink.results == [(True, False, with_inputs)]

    def test_rejects_a_port_that_receives_nothing(
        self, step: Step, start_manual: StartManual
    ):
        out = Port(send=h())
        node = Probe(out.has_received, out=out)
        run = start_manual(Compose(node=node))
        step(run)
        assert isinstance(node.results[0], TypeError)

    def test_only_the_owning_node_asks(self, start_manual: StartManual):
        inp = Port(recv=h())
        start_manual(Compose(node=Probe(inp=inp)))
        with pytest.raises(RuntimeError, match="only the node that owns"):
            inp.has_received()  # from the host


class TestSend:
    def test_copies_the_values(self, step: Step, start_manual: StartManual):
        out = Port(send=h())
        value = torch.zeros(2, 3)

        def send_then_change() -> None:
            out.send(value)
            value.add_(1.0)

        run = start_manual(Compose(node=Probe(send_then_change, out=out)))
        step(run)
        assert torch.equal(run.read(out)[0], torch.zeros(2, 3))

    def test_to_one_channel_changes_only_that_peer(
        self, step: Step, start_manual: StartManual
    ):
        out, a, b = NextPort(send=h()), PrevPort(recv=h()), PrevPort(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.recv, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Compose(source=source, a=sink_a, b=sink_b))
        step(run, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), torch.zeros(2, 3))
        # The output the host reads changes only with send to every peer.
        assert torch.equal(run.read(out)[0], torch.zeros(2, 3))

    def test_to_a_channel_of_another_port(self, step: Step, start_manual: StartManual):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        foreign = Port(send=h()).connect(Port(recv=h()))
        out.connect(inp)
        node = Probe(lambda: out.send_to(foreign, torch.ones(2, 3)), next=out)
        run = start_manual(Compose(node=node, sink=Probe(prev=inp)))
        step(run)
        assert isinstance(node.results[0], ValueError)

    def test_rejects_a_port_that_sends_nothing(
        self, step: Step, start_manual: StartManual
    ):
        inp = Port(recv=h())
        # The type checker rejects it too: the port's message type is Never.
        node = Probe(lambda: inp.send(torch.ones(2, 3)), inp=inp)  # pyright: ignore[reportArgumentType]
        run = start_manual(Compose(node=node))
        step(run)
        assert isinstance(node.results[0], TypeError)

    def test_only_the_owning_node_sends(self, start_manual: StartManual):
        out = Port(send=h())
        start_manual(Compose(node=Probe(out=out)))
        with pytest.raises(RuntimeError, match="only the node that owns"):
            out.send(torch.ones(2, 3))


class TestInitial:
    def test_every_peer_and_the_host_read_it_until_the_first_send(
        self, step: Step, start_manual: StartManual
    ):
        twos = torch.full((2, 3), 2.0)
        out = NextPort(send=h(), initial=twos)
        a, b = PrevPort(recv=h()), PrevPort(recv=h())
        out.connect(a)
        out.connect(b)
        sink_a, sink_b = Probe(a.recv_with_time, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Compose(source=Probe(next=out), a=sink_a, b=sink_b))
        step(run, sink_a, sink_b)  # the source does not send
        ((value, written_at),) = cast(
            list[tuple[torch.Tensor, Timestamp]], sink_a.results
        )
        assert torch.equal(value, twos)
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), twos)
        # Stamped as the zeros would be: with the time the run started (0).
        assert written_at.ns == 0
        assert torch.equal(run.read(out)[0], twos)

    def test_send_to_replaces_it_for_that_peer_only(
        self, step: Step, start_manual: StartManual
    ):
        twos = torch.full((2, 3), 2.0)
        out = NextPort(send=h(), initial=twos)
        a, b = PrevPort(recv=h()), PrevPort(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.recv, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Compose(source=source, a=sink_a, b=sink_b))
        step(run, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), twos)
        assert torch.equal(run.read(out)[0], twos)

    def test_of_a_dict_message(self, step: Step, start_manual: StartManual):
        out = Port(send=fb(), initial={"error": torch.ones(2, 3)})
        inp = Port(recv=fb())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Compose(source=Probe(out=out), sink=sink))
        step(run, sink)
        received = cast(Feedback, sink.results[0])
        assert torch.equal(received["error"], torch.ones(2, 3))

    def test_is_copied_when_the_port_is_made(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        value = torch.ones(2, 3, requires_grad=True)
        out = Port(send=h(), initial=value)
        with torch.no_grad():
            value.add_(1.0)  # after the declaration: must not reach the slots
        run = start_manual(Compose(node=Probe(out=out)))
        received = run.read(out)[0]
        assert torch.equal(received, torch.ones(2, 3))
        assert not received.requires_grad

    def test_a_restored_run_takes_the_value_from_the_state(
        self, step: Step, start_manual: StartManual
    ):
        def build(initial: float) -> tuple[Compose, Probe, Probe]:
            out = NextPort(send=h(), initial=torch.full((2, 3), initial))
            inp = PrevPort(recv=h())
            out.connect(inp)
            source = Probe(lambda: out.send(torch.full((2, 3), 5.0)), next=out)
            sink = Probe(inp.recv, prev=inp)
            return Compose(source=source, sink=sink), source, sink

        net, source, _ = build(2.0)
        run = start_manual(net)
        step(run, source)  # the source sends 5
        tree = run.state_dict()
        run.shutdown()

        restored, _, sink = build(7.0)
        timer = VirtualTimer()
        run = start_manual(restored, timer=timer, state=tree)
        step(run, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]), torch.full((2, 3), 5.0))

    def test_rejects_a_port_that_sends_nothing(self):
        with pytest.raises(TypeError, match="sends no messages"):
            Port(recv=h(), initial=torch.zeros(2, 3))  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize(
        ("value", "error"),
        [
            (torch.zeros(3, 2), ValueError),
            (torch.zeros(2, 3, dtype=torch.float64), TypeError),
            ([[0.0] * 3] * 2, TypeError),
        ],
    )
    def test_rejects_a_value_that_does_not_match_the_send_spec(
        self, value: object, error: type[Exception]
    ):
        with pytest.raises(error, match="initial"):
            NextPort(send=h(), initial=value)  # pyright: ignore[reportArgumentType]

    def test_rejects_a_dict_with_other_keys(self):
        with pytest.raises(ValueError, match="keys do not match"):
            Port(send=fb(), initial={"loss": torch.zeros(2, 3)})  # pyright: ignore[reportArgumentType]


class TestAccelerator:
    def test_receives_on_the_device_and_sends_to_the_host(
        self,
        step: Step,
        start_manual: StartManual,
        accelerator: torch.device,
    ):
        out, inp, back = NextPort(send=h()), PrevPort(recv=h()), Port(send=h())
        out.connect(inp)

        class OnAccelerator(Probe):
            def __init__(self, update: Callable[[], object], **ports: Any) -> None:
                super().__init__(update, **ports)
                Node.__init__(self, device=accelerator.type)

        def update() -> torch.Tensor:
            x = inp.recv()
            back.send(2 * x)
            return x

        # The source stays on the CPU: MPS takes one node at most.
        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)
        sink = OnAccelerator(update, prev=inp, back=back)
        run = start_manual(Compose(source=source, sink=sink))
        step(run, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        # The index-less device is fixed, so the message compares equal to it.
        assert received.device == sink.device == accelerator
        assert torch.equal(received.cpu(), torch.ones(2, 3))
        sent, _ = run.read(back)
        assert sent.device == torch.device("cpu")
        assert torch.equal(sent, torch.full((2, 3), 2.0))

    def test_takes_an_initial_value_on_the_device(
        self,
        step: Step,
        start_manual: StartManual,
        accelerator: torch.device,
    ):
        out = Port(send=h(), initial=torch.ones(2, 3, device=accelerator))
        inp = Port(recv=h())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Compose(source=Probe(out=out), sink=sink))
        step(run, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]), torch.ones(2, 3))
