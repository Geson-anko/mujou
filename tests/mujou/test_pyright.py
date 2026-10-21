# pyright: strict
# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Checked by pyright in strict mode, as users' code is: what a type checker infers
for ports, specs, nodes and runs must be precise, and what the types forbid must be
reported (an ignore that pyright does not need is reported too)."""

from typing import Never, TypedDict, assert_type, override

import pytest
import torch

import mujou
import mujou.testing
from mujou import (
    Channel,
    DictSpec,
    MessageSpec,
    MultiPort,
    Network,
    Node,
    Port,
    TensorSpec,
)
from mujou.time import Timestamp, VirtualTimer


class Feedback(TypedDict):
    error: torch.Tensor
    target: torch.Tensor


FEEDBACK = DictSpec(Feedback, error=TensorSpec((2,)), target=TensorSpec((2,)))
VECTOR = TensorSpec((2,))


def sender[T](spec: MessageSpec[T]) -> MultiPort[T]:
    return MultiPort(send=spec)


# Only type-checked: a port receives and sends in a hook of a running node.
def used(
    one: Port[torch.Tensor, Feedback],
    many: MultiPort[torch.Tensor, Feedback],
    deaf: Port[torch.Tensor],
    mute: Port[Never, torch.Tensor],
    channel: Channel,
) -> None:
    assert_type(one.recv(), Feedback)
    assert_type(one.recv_with_time(), tuple[Feedback, Timestamp | None])
    assert_type(many.recv(), dict[Channel, Feedback])
    assert_type(many.recv_with_time(), dict[Channel, tuple[Feedback, Timestamp | None]])
    one.send(torch.zeros(2))
    many.send_to(channel, torch.zeros(2))
    deaf.send(torch.zeros(2))
    mute.send(torch.zeros(2))  # pyright: ignore[reportArgumentType]
    one.send(FEEDBACK.zeros())  # pyright: ignore[reportArgumentType]
    one.send_to(channel, torch.zeros(2))  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


# Only type-checked: the host reads what a port sends, from either kind of port.
def read(run: mujou.Run, one: Port[Feedback], many: MultiPort[torch.Tensor]) -> None:
    assert_type(run.read(one), Feedback)
    assert_type(run.read_with_time(many), tuple[torch.Tensor, Timestamp | None])


class TestPorts:
    def test_the_specs_give_the_message_types(self):
        assert_type(Port(send=VECTOR, recv=FEEDBACK), Port[torch.Tensor, Feedback])
        assert_type(Port(recv=VECTOR), Port[Never, torch.Tensor])
        assert_type(Port(send=VECTOR), Port[torch.Tensor, Never])
        assert_type(MultiPort(recv=FEEDBACK), MultiPort[Never, Feedback])
        assert_type(Port(send=VECTOR, initial=torch.ones(2)), Port[torch.Tensor])

    def test_connect_takes_the_reversed_types(self):
        a = Port(send=VECTOR, recv=FEEDBACK)
        a.connect(Port(send=FEEDBACK, recv=VECTOR))
        Port(send=VECTOR).connect(MultiPort(recv=VECTOR))
        with pytest.raises(TypeError, match="do not match"):
            Port(send=VECTOR).connect(Port(send=VECTOR))  # pyright: ignore[reportArgumentType]

    def test_the_specs_are_keywords(self):
        with pytest.raises(TypeError):
            Port(VECTOR)  # pyright: ignore[reportCallIssue]

    def test_a_message_spec_takes_any_spec(self):
        tensors: MessageSpec[torch.Tensor] = VECTOR
        feedback: MessageSpec[Feedback] = FEEDBACK
        assert_type(sender(tensors), MultiPort[torch.Tensor, Never])
        assert_type(sender(feedback), MultiPort[Feedback, Never])
        assert isinstance(sender(FEEDBACK).send_spec, DictSpec)


class Layer(Node):
    def __init__(self, n: int) -> None:
        super().__init__(seed=0)
        self.prev = Port(recv=TensorSpec((n,)))
        self.out = Port(send=TensorSpec((n,)))


class DictSink(Node):
    def __init__(self) -> None:
        super().__init__()
        self.prev = Port(recv=FEEDBACK)


class Fan(Node):
    """Merges its peers with annotated ports."""

    def __init__(self) -> None:
        super().__init__()
        self.prev: MultiPort[Never, torch.Tensor] = MultiPort(recv=VECTOR)
        self.next: Port[torch.Tensor] = Port(send=VECTOR)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(sum(self.prev.recv().values(), torch.zeros(2)))


class Wrapped(Network):
    def __init__(self) -> None:
        super().__init__()
        self.layer = Layer(2)
        self.prev = self.layer.prev


class TestRuns:
    def test_a_subclass_and_a_prebuilt_write(self) -> None:
        net = Wrapped()
        values = {net.prev: torch.ones(2)}  # a dict built before the call
        with mujou.run(net, inputs=values) as run:
            run.write(values)
            _, t = run.read_with_time(net.layer.out)
            assert t is None or t.ns >= 0

    def test_a_typed_dict_message_is_written(self) -> None:
        node = DictSink()
        msg: Feedback = FEEDBACK.zeros()
        values = {node.prev: msg}  # a dict built before the call
        with mujou.run(node, inputs={node.prev: msg}) as run:
            run.write({node.prev: msg})
            run.write(values)
        other = DictSink()
        others = {other.prev: msg}
        with mujou.testing.run(other, inputs=others) as manual:
            manual.write(others)

    def test_a_manual_run_gives_back_the_type_of_its_timer(self) -> None:
        with mujou.testing.run(Layer(2)) as run:
            assert_type(run.timer, VirtualTimer)
            run.timer.advance(0.001)  # no cast needed

    def test_annotated_ports_are_read(self) -> None:
        a, b, fan = Layer(2), Layer(2), Fan()
        a.out.connect(fan.prev)
        b.out.connect(fan.prev)
        with mujou.testing.run(mujou.Group(a=a, b=b, fan=fan)) as run:
            run.step(0.0)
            value, _ = run.read_with_time(fan.next)
            assert_type(value, torch.Tensor)

    def test_a_run_takes_one_root(self) -> None:
        # Without the ignores pyright reports the calls, and an unneeded ignore
        # is reported too: the signature itself asks for one root.
        a, b = Layer(2), Layer(2)
        with pytest.raises(TypeError, match="positional argument"):
            mujou.run(a, b)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError, match="positional argument"):
            mujou.testing.run(a, b)  # pyright: ignore[reportCallIssue]
