import copy
from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

import mujou.pc as pc
from mujou import Group, Network, TensorSpec
from mujou.time import VirtualTimer
from mujou.types import Size
from tests.mujou.nodes import F64, assert_same_tree
from tests.mujou.pc._reference import (
    ADAMW,
    MASK,
    N_OUT,
    SGD,
    TAU,
    B,
    H,
    make_modules,
    sample,
)

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual

WIDTHS = [4, 5, 5]
X, Y = TensorSpec((B, WIDTHS[0]), F64), TensorSpec((B, N_OUT), F64)
LEARN_INTERVAL = 3 * H


def chain(*modules: nn.Module, optimizer: pc.Optimizer = SGD) -> pc.Chain:
    return pc.Chain(
        *modules,
        x=X,
        y=Y,
        batch=B,
        loss=pc.MSELoss(B),
        tau=TAU,
        optimizer=optimizer,
        learn_interval=LEARN_INTERVAL,
    )


def by_hand(layers: list[nn.Module], head: nn.Module) -> tuple[Group, pc.Input]:
    """The nodes of ``chain(*layers, head)``, made and connected one by one, and
    the input."""
    nodes: dict[str, Any] = {"input": pc.Input(x=X, y=Y, period=TAU.h / 4, batch=B)}
    last = nodes["input"]
    for i, module in enumerate(layers):
        layer = pc.Layer(
            module,
            act_spec=last.act_spec,
            tau=TAU,
            optimizer=SGD,
            learn_interval=LEARN_INTERVAL,
        )
        last.next.connect(layer.prev)
        nodes[f"layer{i}"] = last = layer
    nodes["output"] = pc.Output(
        head,
        act_spec=last.act_spec,
        loss=pc.MSELoss(B),
        tau=TAU,
        optimizer=SGD,
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

    def test_each_node_binds_the_optimizer_to_its_own_parameters(
        self, start_manual: StartManual
    ):
        # One AdamW for the whole chain: each node binds it to its own module,
        # so each keeps moments of its own (one bound optimizer in two nodes
        # would keep the run from starting).
        layers, head = make_modules("linear", "tanh", WIDTHS)
        net = chain(*layers, head, optimizer=ADAMW)
        tree = run_states(start_manual, net, net.input)
        modules = {"layer0": layers[0], "layer1": layers[1], "output": head}
        for path, module in modules.items():
            moments = tree["nodes"][path]["state"]["optimizer"]["m"]
            assert [m.shape for m in moments] == [p.shape for p in module.parameters()]
        assert "optimizer" not in tree["nodes"]["input"]["state"]

    def test_one_module_is_the_output(self):
        head = nn.Linear(WIDTHS[0], N_OUT).double()
        net = chain(head)
        assert list(vars(net)) == ["input", "output", "prev", "next"]
        assert net.output.prev.recv_spec == net.input.act_spec


def learned(
    start_manual: StartManual,
    batch: Size,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    loss: type[pc.MSELoss | pc.CrossEntropyLoss] = pc.MSELoss,
) -> dict[str, dict[str, Any]]:
    """The state of each node of a chain over ``batch`` that learns with
    ``loss`` after 20 steps with the sample ``(x, y, mask)``, written once."""
    layers, head = make_modules("linear", "tanh", WIDTHS)
    net = pc.Chain(
        *layers,
        head,
        x=TensorSpec(tuple(x.shape), F64),
        y=TensorSpec(tuple(y.shape), y.dtype),
        batch=batch,
        loss=loss(batch),
        tau=TAU,
        optimizer=SGD,
        learn_interval=LEARN_INTERVAL,
    )
    run = start_manual(net, timer=VirtualTimer())
    run.write({net.prev: {"x": x, "y": y, "mask": mask}})
    for _ in range(20):
        run.step(H)
    return {path: node["state"] for path, node in run.state_dict()["nodes"].items()}


def assert_learned_alike(
    states: dict[str, dict[str, Any]], expected: dict[str, dict[str, Any]]
) -> None:
    """The weights are equal, and the activities of each sample."""
    for path in ["layer0", "layer1", "output"]:
        for name, weight in expected[path]["module"].items():
            torch.testing.assert_close(
                states[path]["module"][name], weight, rtol=1e-12, atol=1e-14
            )
    for path in ["layer0", "layer1"]:
        for name in ["h", "lam"]:
            got, want = states[path][name], expected[path][name]
            torch.testing.assert_close(
                got, want.reshape(got.shape), rtol=1e-12, atol=1e-14
            )


class TestBatch:
    def _sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(6, WIDTHS[0], dtype=F64, generator=generator)
        y = torch.randn(6, N_OUT, dtype=F64, generator=generator)
        return x, y, torch.tensor([True, False, True, True, True, False])

    def test_two_dimensions_learn_as_the_flattened_batch(
        self, start_manual: StartManual
    ):
        x, y, mask = self._sample()
        expected = learned(start_manual, 6, x, y, mask)
        states = learned(
            start_manual,
            (2, 3),
            x.reshape(2, 3, -1),
            y.reshape(2, 3, -1),
            mask.reshape(2, 3),
        )
        assert_learned_alike(states, expected)
        initial = make_modules("linear", "tanh", WIDTHS)[1].state_dict()["weight"]
        assert not torch.equal(expected["output"]["module"]["weight"], initial)

    def test_no_batch_learns_as_a_batch_of_one(self, start_manual: StartManual):
        x, y, mask = self._sample()
        expected = learned(start_manual, 1, x[:1], y[:1], mask[:1])
        states = learned(start_manual, (), x[0], y[0], mask[0])
        assert_learned_alike(states, expected)

    def test_one_class_without_a_batch_learns_as_a_batch_of_one(
        self, start_manual: StartManual
    ):
        # The target is a scalar: the class of the one sample.
        x, _, _ = self._sample()
        y = torch.tensor([1, 0, 1, 1, 0, 1])
        expected = learned(
            start_manual, 1, x[:1], y[:1], torch.tensor([True]), pc.CrossEntropyLoss
        )
        states = learned(
            start_manual, (), x[0], y[0], torch.tensor(True), pc.CrossEntropyLoss
        )
        assert_learned_alike(states, expected)
