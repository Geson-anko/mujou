# pyright: strict
# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Checked by pyright in strict mode: the names of ``mujou.typing`` annotate ports
and specs as type checkers infer them."""

from typing import Never, TypedDict, assert_type

import torch

import mujou.typing
from mujou import Channel, DictSpec, NextPort, Port, PrevPort, TensorSpec
from mujou.typing import MessageSpec, Multi, Single


class Sample(TypedDict):
    x: torch.Tensor


def sender[T](spec: MessageSpec[T]) -> NextPort[T, Never, Multi]:
    return NextPort(send=spec)


# Only type-checked: a port receives in a hook of a running node.
def received(
    one: PrevPort[Never, torch.Tensor, Single],
    many: PrevPort[Never, torch.Tensor, Multi],
) -> None:
    assert_type(one.recv(), torch.Tensor)
    assert_type(many.recv(), dict[Channel, torch.Tensor])


class TestPublicNames:
    def test_exports(self):
        assert sorted(mujou.typing.__all__) == ["MessageSpec", "Multi", "Single"]


class TestAnnotations:
    def test_the_peer_count_is_single_or_multi(self):
        one: PrevPort[Never, torch.Tensor, Single] = PrevPort(recv=TensorSpec((3,)))
        many: PrevPort[Never, torch.Tensor, Multi] = PrevPort(
            recv=TensorSpec((3,)), multi=True
        )
        assert_type(Port(send=TensorSpec((3,))), Port[torch.Tensor, Never, Single])
        assert one.recv_spec == many.recv_spec

    def test_a_message_spec_takes_any_spec(self):
        spec = DictSpec(Sample, x=TensorSpec((2,)))
        tensors: MessageSpec[torch.Tensor] = TensorSpec((2,))
        samples: MessageSpec[Sample] = spec
        assert_type(sender(tensors), NextPort[torch.Tensor, Never, Multi])
        assert_type(sender(samples), NextPort[Sample, Never, Multi])
        assert isinstance(sender(spec).send_spec, DictSpec)
