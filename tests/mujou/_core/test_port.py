import queue
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, TypedDict, assert_type, cast, override

import pytest
import torch

from mujou._core import _native
from mujou._core.network import Group, Network
from mujou._core.node import Node
from mujou._core.port import Channel, Multi, NextPort, Port, PrevPort, Single
from mujou._core.spec import DictSpec, TensorSpec
from mujou.testing import VirtualTimer
from mujou.time import Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual, Step


class Feedback(TypedDict):
    error: torch.Tensor


class Other(TypedDict):
    error: torch.Tensor


def h() -> TensorSpec:
    return TensorSpec((2, 3))


def fb(typed_dict: type[Feedback] = Feedback) -> DictSpec[Feedback]:
    return DictSpec(typed_dict, error=TensorSpec((2, 3)))


class Tag(TypedDict):
    id: torch.Tensor
    mask: torch.Tensor


def tag() -> DictSpec[Tag]:
    """What the carry tests carry: a sample ID and a mask."""
    return DictSpec(
        Tag, id=TensorSpec((), torch.int64), mask=TensorSpec((2,), torch.bool)
    )


def tagged(k: int) -> Tag:
    return {"id": torch.tensor(k), "mask": torch.tensor([k % 2 == 0, True])}


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


class TestCarryTypes:
    def test_the_carry_types_follow_the_specs_statically(self):
        # Checked by pyright (assert_type); at runtime only the specs are seen.
        made = NextPort(send=h(), send_carry=tag())
        received = PrevPort(recv=h(), recv_carry=tag())
        passed = NextPort(send=h(), send_carry=received)
        both = Port(send=h(), recv=h(), send_carry=tag(), recv_carry=fb())
        assert_type(made, NextPort[torch.Tensor, Never, Multi, Tag, Never])
        assert_type(received, PrevPort[Never, torch.Tensor, Single, Never, Tag])
        assert_type(passed, NextPort[torch.Tensor, Never, Multi, Tag, Never])
        assert_type(both, Port[torch.Tensor, torch.Tensor, Single, Tag, Feedback])

        def carry_types() -> None:
            # Only for the type checker: these raise before a run.
            assert_type(received.carried(), Tag)
            assert_type(received.carried("id"), torch.Tensor)
            made.send(torch.zeros(2, 3), carry=tagged(0))
            passed.send(torch.zeros(2, 3))
            passed.send(torch.zeros(2, 3), carry=tagged(0))  # pyright: ignore[reportArgumentType]

        assert made.send_carry_spec == tag() and made.recv_carry_spec is None
        assert passed.send_carry_spec is received.recv_carry_spec
        assert passed.carry_source is received and made.carry_source is None

    def test_a_port_of_many_peers_receives_no_carry_statically(self):
        with pytest.raises(
            TypeError, match="takes many peers, so it takes no recv_carry"
        ):
            Port(recv=h(), multi=True, recv_carry=tag())  # pyright: ignore[reportCallIssue, reportArgumentType]
        with pytest.raises(TypeError, match="NextPort takes many peers"):
            NextPort(recv=h(), recv_carry=tag())  # pyright: ignore[reportCallIssue, reportArgumentType]


class TestCarrySpecs:
    @pytest.mark.parametrize("spec", [(2, 3), torch.zeros(3), TensorSpec])
    def test_rejects_what_is_not_a_spec(self, spec: object):
        with pytest.raises(
            TypeError, match="send_carry must be a TensorSpec, a DictSpec, a Port"
        ):
            Port(send=h(), send_carry=spec)  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownVariableType]
        with pytest.raises(TypeError, match="PrevPort recv_carry must be"):
            PrevPort(recv=h(), recv_carry=spec)  # pyright: ignore[reportArgumentType, reportCallIssue, reportUnknownVariableType]

    def test_rejects_a_carry_without_its_direction(self):
        with pytest.raises(
            TypeError, match="sends no messages, so it takes no send_carry"
        ):
            Port(recv=h(), send_carry=tag())
        with pytest.raises(
            TypeError, match="receives no messages, so it takes no recv_carry"
        ):
            Port(send=h(), recv_carry=tag())

    def test_passes_on_only_from_a_port_that_receives_a_carry(self):
        received = PrevPort(recv=h())
        with pytest.raises(TypeError, match="has no recv_carry, so there is nothing"):
            NextPort(send=h(), send_carry=received)  # pyright: ignore[reportArgumentType]

    def test_repr_shows_the_carry(self):
        received = PrevPort(recv=h(), recv_carry=tag())
        assert repr(received) == (
            f"PrevPort(send=None, recv={h()!r}, multi=False, recv_carry={tag()!r})"
        )
        made = Port(send=h(), send_carry=tag())
        assert repr(made).endswith(f"multi=False, send_carry={tag()!r})")
        passed = NextPort(send=h(), send_carry=received)
        assert repr(passed).endswith(f"multi=True, send_carry={received!r})")


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
        self, a: Port[Any, Any, Any, Any, Any], b: Port[Any, Any, Any, Any, Any]
    ):
        with pytest.raises(TypeError, match="do not match"):
            a.connect(b)
        assert a.channels == () and b.channels == ()

    def test_matches_the_carried_data_in_each_direction(self):
        received = PrevPort(recv=h(), send=h(), recv_carry=tag())
        NextPort(send=h(), recv=h(), send_carry=tag()).connect(received)
        passed = NextPort(send=h(), send_carry=received)
        passed.connect(PrevPort(recv=h(), recv_carry=tag()))
        PrevPort(recv=h(), recv_carry=tag()).connect(passed)

    @pytest.mark.parametrize(
        ("send_carry", "recv_carry", "message"),
        [
            (
                tag(),
                fb(),
                r"carries DictSpec\(Tag.* with what it sends, but .* expects "
                r"DictSpec\(Feedback",
            ),
            (
                tag(),
                None,
                r"carries DictSpec\(Tag.* with what it sends, but .* expects nothing$",
            ),
            (
                None,
                tag(),
                r"carries nothing with what it sends, but .* expects DictSpec\(Tag",
            ),
        ],
        ids=["other", "unexpected", "missing"],
    )
    def test_rejects_carried_data_that_does_not_match(
        self,
        send_carry: DictSpec[Any] | None,
        recv_carry: DictSpec[Any] | None,
        message: str,
    ):
        sender = Port(send=h(), send_carry=send_carry)  # pyright: ignore[reportArgumentType]
        receiver = Port(recv=h(), recv_carry=recv_carry)  # pyright: ignore[reportArgumentType]
        with pytest.raises(TypeError, match="^port specs do not match: .*" + message):
            sender.connect(receiver)
        assert not sender.channels and not receiver.channels

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
            super().__init__()
            self.prev = PrevPort(recv=h())
            self.next = NextPort(send=h())
            self.alias = self.next  # a second name does not rename the port

    def test_a_bare_port_is_shown_by_its_specs(self):
        assert repr(Port(send=h())) == f"Port(send={h()!r}, recv=None, multi=False)"

    def test_a_port_of_a_node_is_named_by_its_first_owner_and_attribute(self):
        shift = self.Shift()

        class Exposing(Network):
            def __init__(self) -> None:
                super().__init__()
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
    returns (or the exception it raised). A subclass sets ``on_device`` to
    compute on another device (a keyword would collide with ``**ports``)."""

    on_device = "cpu"

    def __init__(
        self,
        update: Callable[[], object] = lambda: None,
        **ports: Port[Any, Any, Any, Any, Any],
    ) -> None:
        super().__init__(device=self.on_device)
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
        run = start_manual(Group(node=node))
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
        run = start_manual(Group(**nodes, sink=sink))
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
        run = start_manual(Group(source=source, sink=sink, nb=neighbour))
        step(run, source, sink, neighbour)
        ((first, again),) = cast(list[tuple[torch.Tensor, torch.Tensor]], sink.results)
        assert not first.requires_grad and not again.requires_grad
        assert first.device == torch.device("cpu")
        assert torch.equal(again, torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, neighbour.results[0]), torch.ones(2, 3))
        assert torch.equal(run.read(out), torch.ones(2, 3))

    def test_with_time_gives_the_time_of_the_write(
        self, step: Step, virtual: VirtualTimer, start_manual: StartManual
    ):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)
        sink = Probe(inp.recv_with_time, prev=inp)
        run = start_manual(Group(source=source, sink=sink))
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
        run = start_manual(Group(owner=Probe(mine=theirs), node=owner, other=other))
        step(run)
        assert isinstance(other.results[0], RuntimeError)
        with pytest.raises(RuntimeError, match="only the node that owns"):
            mine.recv()  # from the host

    def test_rejects_a_port_that_receives_nothing(
        self, step: Step, start_manual: StartManual
    ):
        out = Port(send=h())
        node = Probe(out.recv, out=out)
        run = start_manual(Group(node=node))
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
        run = start_manual(Group(source=source, sink=sink))
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
        run = start_manual(Group(**nodes, sink=sink, lonely=lonely))
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
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        step(run, source, sink_a, sink_b)
        assert (sink_a.results, sink_b.results) == ([True], [False])

    def test_counts_what_the_host_writes(self, step: Step, start_manual: StartManual):
        given, written = Port(recv=h()), Port(recv=h())
        node = Probe(
            lambda: (given.has_received(), written.has_received()),
            given=given,
            written=written,
        )
        run = start_manual(Group(node=node), inputs={given: torch.ones(2, 3)})
        step(run)
        run.write({written: torch.ones(2, 3)})
        step(run)
        assert node.results == [(True, False), (True, True)]

    @pytest.mark.parametrize("with_inputs", [False, True])
    def test_a_restored_run_keeps_what_was_sent(
        self, step: Step, start_manual: StartManual, with_inputs: bool
    ):
        def build() -> tuple[Group, Probe, Probe, Port[Never, torch.Tensor]]:
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
            return Group(source=source, sink=sink), source, sink, host

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
        run = start_manual(Group(node=node))
        step(run)
        assert isinstance(node.results[0], TypeError)

    def test_only_the_owning_node_asks(self, start_manual: StartManual):
        inp = Port(recv=h())
        start_manual(Group(node=Probe(inp=inp)))
        with pytest.raises(RuntimeError, match="only the node that owns"):
            inp.has_received()  # from the host


class TestSend:
    def test_copies_the_values(self, step: Step, start_manual: StartManual):
        out = Port(send=h())
        value = torch.zeros(2, 3)

        def send_then_change() -> None:
            out.send(value)
            value.add_(1.0)

        run = start_manual(Group(node=Probe(send_then_change, out=out)))
        step(run)
        assert torch.equal(run.read(out), torch.zeros(2, 3))

    def test_to_one_channel_changes_only_that_peer(
        self, step: Step, start_manual: StartManual
    ):
        out, a, b = NextPort(send=h()), PrevPort(recv=h()), PrevPort(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = Probe(a.recv, prev=a), Probe(b.recv, prev=b)
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        step(run, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), torch.zeros(2, 3))
        # The output the host reads changes only with send to every peer.
        assert torch.equal(run.read(out), torch.zeros(2, 3))

    def test_to_a_channel_of_another_port(self, step: Step, start_manual: StartManual):
        out, inp = NextPort(send=h()), PrevPort(recv=h())
        foreign = Port(send=h()).connect(Port(recv=h()))
        out.connect(inp)
        node = Probe(lambda: out.send_to(foreign, torch.ones(2, 3)), next=out)
        run = start_manual(Group(node=node, sink=Probe(prev=inp)))
        step(run)
        assert isinstance(node.results[0], ValueError)

    def test_rejects_a_port_that_sends_nothing(
        self, step: Step, start_manual: StartManual
    ):
        inp = Port(recv=h())
        # The type checker rejects it too: the port's message type is Never.
        node = Probe(lambda: inp.send(torch.ones(2, 3)), inp=inp)  # pyright: ignore[reportArgumentType]
        run = start_manual(Group(node=node))
        step(run)
        assert isinstance(node.results[0], TypeError)

    def test_only_the_owning_node_sends(self, start_manual: StartManual):
        out = Port(send=h())
        start_manual(Group(node=Probe(out=out)))
        with pytest.raises(RuntimeError, match="only the node that owns"):
            out.send(torch.ones(2, 3))


class Signalling(Probe):
    """A ``Probe`` that also signals each update it finished."""

    def __init__(
        self,
        update: Callable[[], object] = lambda: None,
        **ports: Port[Any, Any, Any, Any, Any],
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
                assert "only the node that owns" in str(result)
        assert torch.equal(run.read(out), torch.zeros(2, 3))
        assert len(owner.results) == 2


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
        run = start_manual(Group(source=Probe(next=out), a=sink_a, b=sink_b))
        step(run, sink_a, sink_b)  # the source does not send
        ((value, written_at),) = cast(
            list[tuple[torch.Tensor, Timestamp]], sink_a.results
        )
        assert torch.equal(value, twos)
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), twos)
        # Stamped as the zeros would be: with the time the run started (0).
        assert written_at.ns == 0
        assert torch.equal(run.read(out), twos)

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
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        step(run, source, sink_a, sink_b)
        assert torch.equal(cast(torch.Tensor, sink_a.results[0]), torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, sink_b.results[0]), twos)
        assert torch.equal(run.read(out), twos)

    def test_of_a_dict_message(self, step: Step, start_manual: StartManual):
        out = Port(send=fb(), initial={"error": torch.ones(2, 3)})
        inp = Port(recv=fb())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Group(source=Probe(out=out), sink=sink))
        step(run, sink)
        received = cast(Feedback, sink.results[0])
        assert torch.equal(received["error"], torch.ones(2, 3))

    def test_is_copied_when_the_port_is_made(self, start_manual: StartManual):
        value = torch.ones(2, 3, requires_grad=True)
        out = Port(send=h(), initial=value)
        with torch.no_grad():
            value.add_(1.0)  # after the declaration: must not reach the slots
        run = start_manual(Group(node=Probe(out=out)))
        received = run.read(out)
        assert torch.equal(received, torch.ones(2, 3))
        assert not received.requires_grad

    def test_a_restored_run_takes_the_value_from_the_state(
        self, step: Step, start_manual: StartManual
    ):
        def build(initial: float) -> tuple[Group, Probe, Probe]:
            out = NextPort(send=h(), initial=torch.full((2, 3), initial))
            inp = PrevPort(recv=h())
            out.connect(inp)
            source = Probe(lambda: out.send(torch.full((2, 3), 5.0)), next=out)
            sink = Probe(inp.recv, prev=inp)
            return Group(source=source, sink=sink), source, sink

        net, source, _ = build(2.0)
        run = start_manual(net)
        step(run, source)  # the source sends 5
        tree = run.state_dict()
        run.shutdown()

        restored, _, sink = build(7.0)
        run = start_manual(restored, timer=VirtualTimer(), state=tree)
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


def scalar() -> TensorSpec:
    return TensorSpec(())


class Path:
    """Source -> Relay -> Sink: the source sends ``k`` carrying ``tagged(k)`` at
    its k-th update, the relay sends what it received plus 100 and passes the
    carry on, and the sink records ``(value, carried id)``.

    The ports are made here so that a test can reach them, and a test replaces
    what a node does by setting its ``update``. ``sink_ports`` are more ports of
    the sink.
    """

    def __init__(
        self, sink: type[Probe] = Probe, **sink_ports: Port[Any, Any, Any, Any, Any]
    ) -> None:
        self.out = NextPort(send=scalar(), send_carry=tag())
        self.prev = PrevPort(recv=scalar(), recv_carry=tag())
        self.next = NextPort(send=scalar(), send_carry=self.prev)
        self.end = PrevPort(recv=scalar(), recv_carry=tag())
        self.out.connect(self.prev)
        self.next.connect(self.end)
        self.k = 0
        self.source = Probe(self.send, out=self.out)
        self.relay = Probe(self.pass_on, prev=self.prev, next=self.next)
        self.sink = sink(self.receive, end=self.end, **sink_ports)

    def send(self) -> None:
        self.k += 1
        self.out.send(torch.tensor(float(self.k)), carry=tagged(self.k))

    def pass_on(self) -> None:
        self.next.send(self.prev.recv() + 100)

    def receive(self) -> tuple[float, int]:
        return self.end.recv().item(), int(self.end.carried("id").item())

    def group(self) -> Group:
        return Group(source=self.source, relay=self.relay, sink=self.sink)


class TestCarried:
    def test_is_the_carry_of_the_value_received_in_any_order(
        self, step: Step, start_manual: StartManual
    ):
        path = Path()
        run = start_manual(path.group())
        nodes = [path.source, path.relay, path.sink]
        order = torch.Generator().manual_seed(0)
        for _ in range(60):
            perm = torch.randperm(3, generator=order).tolist()
            step(run, *(nodes[i] for i in perm))
        received = cast(list[tuple[float, int]], path.sink.results)
        # Before the relay first sends, the initial value and the zeros it carries.
        pairs = {(value, k) for value, k in received if (value, k) != (0.0, 0)}
        assert len(pairs) > 20
        for value, k in pairs:
            assert value == k + 100, f"{value} came with the carry of sample {k}"

    def test_is_the_initial_carry_until_the_first_recv(
        self, step: Step, start_manual: StartManual
    ):
        path = Path()
        results: list[tuple[Tag, Tag]] = []

        def update() -> None:
            before = path.end.carried()
            path.end.recv()
            results.append((before, path.end.carried()))

        path.sink.update = update
        run = start_manual(path.group())
        step(run, path.source, path.relay, path.sink)
        ((before, after),) = results
        assert torch.equal(before["id"], torch.tensor(0))
        assert torch.equal(before["mask"], torch.zeros(2, dtype=torch.bool))
        assert torch.equal(after["id"], torch.tensor(1))
        assert torch.equal(after["mask"], tagged(1)["mask"])

    def test_an_initial_value_carries_zeros(
        self, step: Step, start_manual: StartManual
    ):
        out = NextPort(send=scalar(), send_carry=tag(), initial=torch.tensor(5.0))
        end = PrevPort(recv=scalar(), recv_carry=tag())
        out.connect(end)
        sink = Probe(lambda: (end.recv(), end.carried()), end=end)
        run = start_manual(Group(source=Probe(out=out), sink=sink))
        step(run, sink)
        value, carried = cast(tuple[torch.Tensor, Tag], sink.results[0])
        assert value.item() == 5.0
        assert carried["id"].item() == 0 and not carried["mask"].any()

    def test_copies_one_field(self, step: Step, start_manual: StartManual):
        path = Path()
        run = start_manual(path.group())
        step(run, path.source, path.relay, path.sink)
        path.sink.update = lambda: (path.end.carried("mask"), path.end.carried("mask"))
        step(run, path.sink)
        a, b = cast(tuple[torch.Tensor, torch.Tensor], path.sink.results[-1])
        assert torch.equal(a, tagged(1)["mask"])
        assert a.data_ptr() != b.data_ptr()
        assert a.device == path.sink.device

    def test_passes_on_what_was_last_received(
        self, step: Step, start_manual: StartManual
    ):
        path = Path()
        path.relay.update = lambda: path.next.send(torch.tensor(-1.0))
        run = start_manual(path.group())
        # Nothing received yet: the relay carries the zeros of its source.
        step(run, path.source, path.relay, path.sink)
        assert path.sink.results[-1] == (-1.0, 0)
        path.relay.update = path.prev.recv
        step(run, path.relay)  # receives sample 1
        step(run, path.source)  # sample 2 waits unreceived
        path.relay.update = lambda: path.next.send(torch.tensor(-1.0))
        step(run, path.relay, path.sink)
        assert path.sink.results[-1] == (-1.0, 1)

    def test_shares_the_carried_buffers_along_the_path(
        self, step: Step, start_manual: StartManual
    ):
        path = Path()
        run = start_manual(path.group())
        step(run, path.source, path.relay)
        slots = run._session._slots  # pyright: ignore[reportPrivateUsage]
        source, relayed = slots["source.out/ch0"][0], slots["relay.next/ch0"][0]

        def data_ptrs(slot: _native.Slot | _native.InputSlot) -> list[int]:
            return [torch.from_dlpack(v).data_ptr() for v in slot.read().carry.buffers]

        assert data_ptrs(relayed) == data_ptrs(source)

    def test_rejects_a_missing_or_unexpected_carry(
        self, step: Step, start_manual: StartManual
    ):
        plain = Port(send=scalar())
        path = Path(plain=plain)
        x = torch.tensor(1.0)
        path.source.update = lambda: path.out.send(x)
        path.relay.update = lambda: path.next.send(x, carry=tagged(1))  # pyright: ignore[reportArgumentType]
        path.sink.update = lambda: plain.send(x, carry=tagged(1))  # pyright: ignore[reportArgumentType]
        run = start_manual(path.group())
        step(run)
        errors = [n.results[0] for n in (path.source, path.relay, path.sink)]
        messages = [
            r"'source\.out' carries DictSpec\(Tag.*\); pass it as carry=",
            r"'relay\.next' passes on the data its send_carry port received, so it "
            "takes no carry",
            r"'sink\.plain' carries no data",
        ]
        for error, message in zip(errors, messages, strict=True):
            assert isinstance(error, TypeError)
            assert re.search(message, str(error)), error

    def test_rejects_what_has_no_carry_or_key(
        self, step: Step, start_manual: StartManual
    ):
        plain = PrevPort(recv=scalar())
        tensor = PrevPort(recv=scalar(), recv_carry=scalar())
        path = Path(plain=plain, tensor=tensor)
        attempts: list[Callable[[], object]] = [
            plain.carried,
            lambda: tensor.carried("id"),
            lambda: path.end.carried("label"),
        ]
        path.sink.update = lambda: [_raised(a) for a in attempts]
        run = start_manual(path.group())
        step(run, path.sink)
        (errors,) = cast(list[list[Exception]], path.sink.results)
        assert isinstance(errors[0], TypeError) and "no recv_carry" in str(errors[0])
        assert isinstance(errors[1], TypeError) and "carries a tensor" in str(errors[1])
        assert isinstance(errors[2], KeyError) and "'label'" in str(errors[2])
        with pytest.raises(RuntimeError, match="only the node that owns the port"):
            path.end.carried()


def _raised(attempt: Callable[[], object]) -> Exception | None:
    try:
        attempt()
    except Exception as e:  # noqa: BLE001 - returned for the test to inspect
        return e
    return None


@pytest.mark.cuda
class TestCuda:
    def test_receives_on_the_device_and_sends_to_the_host(
        self, step: Step, start_manual: StartManual
    ):
        out, inp, back = NextPort(send=h()), PrevPort(recv=h()), Port(send=h())
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
        step(run, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        # The index-less device is fixed, so the message compares equal to it.
        assert received.device == sink.device == torch.device("cuda", 0)
        assert torch.equal(received.cpu(), torch.ones(2, 3))
        sent = run.read(back)
        assert sent.device == torch.device("cpu")
        assert torch.equal(sent, torch.full((2, 3), 2.0))

    def test_copies_the_carried_data_to_the_device(
        self, step: Step, start_manual: StartManual
    ):
        class OnCuda(Probe):
            on_device = "cuda"

        path = Path(sink=OnCuda)
        path.sink.update = lambda: (path.end.recv(), path.end.carried("id"))
        run = start_manual(path.group())
        step(run, path.source, path.relay, path.sink)
        _, carried = cast(tuple[torch.Tensor, torch.Tensor], path.sink.results[0])
        assert carried.device == torch.device("cuda", 0)
        assert carried.item() == 1

    def test_takes_an_initial_value_on_the_device(
        self, step: Step, start_manual: StartManual
    ):
        out = Port(send=h(), initial=torch.ones(2, 3, device="cuda:0"))
        inp = Port(recv=h())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Group(source=Probe(out=out), sink=sink))
        step(run, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]), torch.ones(2, 3))
