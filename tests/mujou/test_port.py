from typing import Any, Never, TypedDict, assert_type

import pytest
import torch

from mujou._port import Channel, Multi, NextPort, Port, PrevPort, Single
from mujou._spec import DictSpec, TensorSpec


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
