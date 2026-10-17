# pyright: strict
# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Checked by pyright in strict mode, as users' code and mujou's own nodes are:
subclassing ``mujou.Node`` and writing to a run must type-check cleanly."""

from typing import TypedDict, assert_type

import pytest
import torch

import mujou
import mujou.testing
from mujou import DictSpec, Network, Node, Port, PrevPort, TensorSpec
from mujou.testing import VirtualTimer
from mujou.time import Timer


class Layer(Node):
    def __init__(self, n: int) -> None:
        super().__init__(seed=0)
        self.prev = PrevPort(recv=TensorSpec((n,)))
        self.out = Port(send=TensorSpec((n,)))


class Feedback(TypedDict):
    error: torch.Tensor
    target: torch.Tensor


FEEDBACK = DictSpec(Feedback, error=TensorSpec((2,)), target=TensorSpec((2,)))


class DictSink(Node):
    def __init__(self) -> None:
        super().__init__()
        self.prev = PrevPort(recv=FEEDBACK)


class Single(Network):
    def __init__(self) -> None:
        self.layer = Layer(2)
        self.prev = self.layer.prev


class TestStrictTyping:
    def test_a_subclass_and_a_prebuilt_write(self) -> None:
        net = Single()
        values = {net.prev: torch.ones(2)}  # a dict built before the call
        with mujou.run(net, inputs=values) as run:
            run.write(values)
            _, t = run.read(net.layer.out)
            assert t.ns >= 0

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
        with mujou.testing.run(Layer(2), timer=Timer()) as run:
            assert_type(run.timer, Timer)

    def test_a_run_takes_at_least_one_component(self) -> None:
        # Without the ignore pyright reports the call, and an unneeded ignore is
        # reported too: the signature itself asks for a component.
        with pytest.raises(TypeError, match="component"):
            mujou.run()  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError, match="component"):
            mujou.testing.run()  # pyright: ignore[reportCallIssue]
