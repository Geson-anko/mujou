import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

import mujou.pc as pc
from mujou import Group, Network, TensorSpec
from mujou.time import VirtualTimer
from tests.mujou.nodes import F64, assert_same_tree
from tests.mujou.pc._reference import MASK, N_OUT, TAU, B, H, make_modules, sample

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual

WIDTHS = [4, 5, 5]
X, Y = TensorSpec((B, WIDTHS[0]), F64), TensorSpec((B, N_OUT), F64)
LEARN_INTERVAL = 3 * H


def chain(*modules: nn.Module) -> pc.Chain:
    return pc.Chain(
        *modules, x=X, y=Y, loss=pc.mse, tau=TAU, learn_interval=LEARN_INTERVAL
    )


def by_hand(layers: list[nn.Module], head: nn.Module) -> tuple[Group, pc.Input]:
    """The nodes of ``chain(*layers, head)``, made and connected one by one, and
    the input."""
    nodes: dict[str, Any] = {"input": pc.Input(x=X, y=Y, period=TAU.h / 4)}
    last = nodes["input"]
    for i, module in enumerate(layers):
        layer = pc.Layer(
            module, act_spec=last.act_spec, tau=TAU, learn_interval=LEARN_INTERVAL
        )
        last.next.connect(layer.prev)
        nodes[f"layer{i}"] = last = layer
    nodes["output"] = pc.Output(
        head,
        act_spec=last.act_spec,
        loss=pc.mse,
        tau=TAU,
        learn_interval=LEARN_INTERVAL,
    )
    last.next.connect(nodes["output"].prev)
    return Group(**nodes), nodes["input"]


def run_states(
    start_manual: StartManual, net: Network, inp: pc.Input
) -> dict[str, Any]:
    """The state tree of ``net`` after 20 steps in the order of the tree, with a
    new labeled sample written to ``inp`` every 7 steps, without the nodes'
    generators (seeded by chance)."""
    run = start_manual(net, timer=VirtualTimer())
    for s in range(20):
        if s % 7 == 0:
            x, y = sample(WIDTHS, s)
            run.write({inp.prev: {"x": x, "y": y, "mask": MASK}})
        run.step(H)
    tree = run.state_dict()
    for node in tree["nodes"].values():
        node["state"].pop("generator")
    return tree


class TestChain:
    def test_runs_as_the_nodes_connected_by_hand(self, start_manual: StartManual):
        layers, head = make_modules("linear", "tanh", WIDTHS)
        initial = copy.deepcopy(head.state_dict())
        net = chain(*copy.deepcopy(layers), copy.deepcopy(head))
        expected = run_states(start_manual, *by_hand(layers, head))
        assert_same_tree(run_states(start_manual, net, net.input), expected)
        learned = expected["nodes"]["output"]["state"]["module"]["weight"]
        assert not torch.equal(learned, initial["weight"])  # it did learn

    def test_the_children_are_named_after_their_place(self):
        layers, head = make_modules("linear", "tanh", WIDTHS)
        net = chain(*layers, head)
        assert list(vars(net)) == [
            "input",
            "layer0",
            "layer1",
            "output",
            "prev",
            "next",
        ]
        assert [layer.module for layer in net.layers] == layers
        assert net.layers[1] is getattr(net, "layer1")
        assert net.output.module is head

    def test_each_node_takes_the_act_of_the_one_before(self):
        layers, head = make_modules("linear", "tanh", WIDTHS)
        net = chain(*layers, head)
        nodes = [net.input, *net.layers]
        for sender, receiver in zip(nodes, [*nodes[1:], net.output], strict=True):
            assert receiver.prev.recv_spec == sender.act_spec
            assert sender.next.channels == receiver.prev.channels

    def test_exposes_the_ports_of_the_host(self):
        layers, head = make_modules("linear", "tanh", WIDTHS)
        net = chain(*layers, head)
        assert net.prev is net.input.prev
        assert net.next is net.output.next

    def test_every_node_runs_at_a_quarter_of_tau_h(self):
        layers, head = make_modules("linear", "tanh", WIDTHS)
        net = chain(*layers, head)
        periods = [
            node.period for node in vars(net).values() if isinstance(node, pc.Layer)
        ]
        periods += [net.input.period, net.output.period]
        assert periods == pytest.approx([TAU.h / 4] * 4)

    def test_one_module_is_the_output(self):
        head = nn.Linear(WIDTHS[0], N_OUT).double()
        net = chain(head)
        assert list(vars(net)) == ["input", "output", "prev", "next"]
        assert net.output.prev.recv_spec == net.input.act_spec
