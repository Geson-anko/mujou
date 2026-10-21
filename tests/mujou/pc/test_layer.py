import copy
import functools
import math
import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from torch import nn

import mujou.pc as pc
from mujou import DictSpec, Group, TensorSpec
from mujou.testing import ManualRun
from mujou.time import VirtualTimer
from tests.mujou.nodes import F64, assert_same_tree, close
from tests.mujou.pc._reference import (
    MASK,
    N_OUT,
    NO_LEARNING,
    TAU,
    B,
    H,
    Reference,
    credits,
    forward,
    make_modules,
    masked_mse,
    sample,
)

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual


class HandChain:
    """``Input -> Layer -> ... -> Output (mse)`` over ``B`` samples, float64,
    wired by hand."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        head: nn.Module,
        widths: Sequence[int],
        *,
        tau: pc.TimeConstants = TAU,
        learn_interval: float | None = None,
        max_dt: float | None = None,
    ) -> None:
        x, y = TensorSpec((B, widths[0]), F64), TensorSpec((B, N_OUT), F64)
        self.inp = pc.Input(x=x, y=y, period=H)
        learning: dict[str, Any] = {
            "tau": tau,
            "learn_interval": learn_interval,
            "max_dt": max_dt,
        }
        self.layers: list[pc.Layer] = []
        act = self.inp.act_spec
        for module in layers:
            self.layers.append(pc.Layer(module, act_spec=act, **learning))
            act = self.layers[-1].act_spec
        self.out = pc.Output(head, act_spec=act, loss=pc.mse, **learning)
        senders = [self.inp, *self.layers]
        for sender, receiver in zip(senders, [*self.layers, self.out], strict=True):
            sender.next.connect(receiver.prev)
        named = {f"l{i}": layer for i, layer in enumerate(self.layers)}
        self.net = Group(inp=self.inp, out=self.out, **named)

    def from_next(self) -> list[Any]:
        """The nodes from the output to the input: each reads what the others
        sent in the previous step (a Jacobi update)."""
        return [self.out, *reversed(self.layers), self.inp]

    def from_prev(self) -> list[Any]:
        return [self.inp, *self.layers, self.out]

    def write(
        self, run: ManualRun, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> int:
        return run.write({self.inp.prev: {"x": x, "y": y, "mask": mask}}).ns


def activity(h: TensorSpec) -> DictSpec[pc.Activity]:
    return pc.activity_spec(h, TensorSpec((B, N_OUT)))


def run_steps(run: ManualRun, order: Sequence[Any], n: int) -> None:
    for _ in range(n):
        run.step(H, *order)


class TestConstruction:
    def _layer(self, **kwargs: Any) -> pc.Layer:
        args: dict[str, Any] = {
            "act_spec": activity(TensorSpec((B, 4))),
            "tau": TAU,
            "learn_interval": None,
        }
        module = kwargs.pop("module", nn.Linear(4, 5))
        return pc.Layer(module, **(args | kwargs))

    def test_the_output_shape_comes_from_the_module(self):
        layer = self._layer(module=nn.Sequential(nn.Linear(4, 6), nn.Tanh()))
        assert layer.next.send_spec == activity(TensorSpec((B, 6)))
        assert layer.prev.recv_spec == activity(TensorSpec((B, 4)))
        assert layer.h.shape == layer.lam.shape == (B, 6)

    def test_the_dtype_comes_from_the_module(self):
        layer = self._layer(
            module=nn.Linear(4, 5).double(), act_spec=activity(TensorSpec((B, 4), F64))
        )
        assert layer.next.send_spec == activity(TensorSpec((B, 5), F64))

    def test_the_period_defaults_to_a_quarter_of_tau_h(self):
        assert self._layer().period == pytest.approx(TAU.h / 4)
        assert self._layer(period=2e-3).period == 2e-3

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"module": nn.Flatten(0)}, "same batch"),
            ({"period": 3e-3}, "max_dt"),  # longer than tau.h / 2
            ({"period": 2e-3, "max_dt": 1e-3}, "max_dt"),
        ],
    )
    def test_value_errors(self, kwargs: dict[str, Any], match: str):
        with pytest.raises(ValueError, match=match):
            self._layer(**kwargs)


class TestJacobi:
    @pytest.mark.parametrize("learn_interval", [None, 4 * H])
    def test_stepping_from_the_output_is_the_jacobi_update(
        self, start_manual: StartManual, learn_interval: float | None
    ):
        # Linear + tanh, then a residual block: the reference moves all nodes at
        # once; stepping from the output to the input reads only values of the
        # previous step everywhere, so the two agree to rounding. Learning in
        # every step, the layers move h by the jumps of their own theta; at
        # intervals, these jumps also go up the chain with the counts of jumps.
        widths = [4, 5, 5]
        layers = [
            make_modules("linear", "tanh", widths[:2])[0][0],
            make_modules("res", "tanh", widths[1:], seed=1)[0][0],
        ]
        head = make_modules("linear", "tanh", widths, seed=2)[1]
        reference = Reference(layers, head, widths, learn_interval)
        chain = HandChain(layers, head, widths, learn_interval=learn_interval)
        run = start_manual(chain.net)
        samples = [sample(widths, seed) for seed in (0, 1)]
        x, y = samples[0]
        time = chain.write(run, x, y, MASK)
        for k in range(50):
            if k == 20:  # a new input, and a new target with it
                x, y = samples[1]
                time = chain.write(run, x, y, MASK)
            run.step(H, *chain.from_next())
            reference.step(x, y, MASK, time, H, run.timer.now().ns)
            tree = run.state_dict()
            for i, layer in enumerate(reference.layers):
                state = tree["nodes"][f"l{i}"]["state"]
                close(state["h"], reference.h[i + 1], 1e-12)
                close(state["lam"], reference.lam[i], 1e-12)
                for key, value in layer.state_dict().items():
                    close(state["module"][key], value, 1e-12)
                sent = tree["nodes"][f"l{i}"]["ports"]["next"]["out"]["value"]
                close(sent["h"], reference.h[i + 1], 1e-12)
                assert int(sent["jumps"]) == reference.jumps[i + 1]
                assert torch.equal(sent["y"], reference.carried[i + 1][0])
                assert torch.equal(sent["mask"], reference.carried[i + 1][1])
                close(
                    tree["nodes"][f"l{i}"]["ports"]["prev"]["out"]["value"],
                    reference.g_in[i],
                    1e-12,
                )
            assert (
                int(tree["nodes"]["inp"]["ports"]["next"]["out"]["value"]["jumps"])
                == reference.jumps[0]
            )
            for key, value in reference.head.state_dict().items():
                close(tree["nodes"]["out"]["state"]["module"][key], value, 1e-12)
            close(
                tree["nodes"]["out"]["ports"]["prev"]["out"]["value"],
                reference.g_in[-1],
                1e-12,
            )
            prediction = run.read(chain.out.next)
            close(prediction["y_hat"], reference.y_hat, 1e-12)
            assert torch.equal(prediction["y"], reference.prediction_carried[0])
            assert torch.equal(prediction["mask"], reference.prediction_carried[1])
        if learn_interval is not None:  # the chain did jump
            assert reference.jumps[-1] > reference.jumps[0] + 1


# Steps to the fixed point from h = lam = 0, to 1e-10 at TAU and these sizes.
RELAX_STEPS = 3000
# The same for the chain of eight residual blocks, to 1e-9.
DEEP_STEPS = 6000
CASES = [(block, act) for block in ("linear", "res") for act in ("linear", "tanh")]


@pytest.mark.slow
class TestFixedPoint:
    @pytest.mark.parametrize(("block", "act"), CASES)
    def test_lam_is_the_credit_of_backpropagation(
        self, start_manual: StartManual, block: str, act: str
    ):
        widths = [4, 4, 4] if block == "res" else [4, 5, 3]
        layers, head = make_modules(block, act, widths)
        chain = HandChain(layers, head, widths, tau=NO_LEARNING)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        run_steps(run, chain.from_next(), RELAX_STEPS)
        nodes = run.state_dict()["nodes"]
        expected = credits(layers, head, x, y, MASK)
        with torch.no_grad():
            forward_hs, _ = forward(layers, head, x)
        for i, (credit, h) in enumerate(zip(expected, forward_hs, strict=True)):
            close(nodes[f"l{i}"]["state"]["h"], h, 1e-10)
            close(nodes[f"l{i}"]["state"]["lam"], credit, 1e-8)
        if block == "linear":
            # skill pc-alm 4: lam_{L-1} = W_L^T e_L,
            # lam_i = W_{i+1}^T (sigma'(a_{i+1}) * lam_{i+1}).
            hs, y_hat = forward(layers, head, x)
            assert isinstance(head, nn.Linear)
            lam_last = torch.where(MASK[:, None], y - y_hat, 0.0) @ head.weight
            linear = cast(nn.Linear, cast(nn.Sequential, layers[1])[0])
            a = linear(hs[0])
            slope = torch.ones_like(a) if act == "linear" else 1 - torch.tanh(a) ** 2
            lam_first = (slope * lam_last) @ linear.weight
            close(nodes["l1"]["state"]["lam"], lam_last.detach(), 1e-8)
            close(nodes["l0"]["state"]["lam"], lam_first.detach(), 1e-8)

    @pytest.mark.parametrize(("block", "act"), CASES)
    def test_one_update_at_the_fixed_point_is_sgd(
        self, start_manual: StartManual, block: str, act: str
    ):
        # The first update comes learn_interval after the input was written (at
        # time 0), at the fixed point, weighted by the time since the start.
        widths = [4, 4, 4] if block == "res" else [4, 5, 3]
        layers, head = make_modules(block, act, widths)
        sgd = [copy.deepcopy(m) for m in [*layers, head]]
        chain = HandChain(layers, head, widths, learn_interval=RELAX_STEPS * H)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        run_steps(run, chain.from_next(), RELAX_STEPS)
        # Backpropagation of the mean loss, one step of the same rate.
        rate = RELAX_STEPS * H / TAU.theta
        _, y_hat = forward(sgd[:-1], sgd[-1], x)
        params = [p for m in sgd for p in m.parameters()]
        grads = torch.autograd.grad(masked_mse(y_hat, y, MASK) / B, params)
        with torch.no_grad():
            for p, g in zip(params, grads, strict=True):
                p -= rate * g
        nodes = run.state_dict()["nodes"]
        for name, module in zip(
            [*(f"l{i}" for i in range(len(layers))), "out"], sgd, strict=True
        ):
            for key, value in module.state_dict().items():
                close(nodes[name]["state"]["module"][key], value, 1e-10)


@pytest.mark.slow
class TestDeepChain:
    def test_eight_residual_nodes_reach_the_fixed_point(
        self, start_manual: StartManual
    ):
        # Residual blocks have Jacobians near the identity. Holding the
        # deviation h - mu of each node fixed while mu follows the nodes below
        # (the previous design) let the credit of the loss loop back through the
        # product of the Jacobians, and this chain diverged (lam ~ 1e40 in 4000
        # steps). With h as the state, it reaches the fixed point. The target is
        # the forward pass plus noise, for credits of order 1.
        widths = [8] * 9
        layers, head = make_modules("res", "tanh", widths)
        x, noise = sample(widths, 0)
        with torch.no_grad():
            forward_hs, y_hat = forward(layers, head, x)
        y = y_hat + noise
        chain = HandChain(layers, head, widths, tau=NO_LEARNING)
        run = start_manual(chain.net)
        chain.write(run, x, y, MASK)
        run_steps(run, chain.from_next(), DEEP_STEPS)
        nodes = run.state_dict()["nodes"]
        expected = credits(layers, head, x, y, MASK)
        for i, (credit, h) in enumerate(zip(expected, forward_hs, strict=True)):
            close(nodes[f"l{i}"]["state"]["h"], h, 1e-9)
            close(nodes[f"l{i}"]["state"]["lam"], credit, 1e-8)

    @pytest.mark.parametrize(
        "learn_interval", [None, 16 * H], ids=["every-update", "16-periods"]
    )
    def test_learning_keeps_lam_at_the_credit(
        self, start_manual: StartManual, learn_interval: float | None
    ):
        # A theta so slow it barely moves: only the jumps theta makes move h,
        # not the relaxation of the nodes below, so lam goes to the credit and
        # stays there. (Moving h with every change of mu, the previous design,
        # made lam grow to 1e5 in 500 steps here with learn_interval=None,
        # however slow theta was, before it settled elsewhere.) 16 periods is
        # about the round trip 2 * depth * period of this chain (18 periods); an
        # interval much shorter advances the jumps in almost every update, the
        # nodes above move h with every change of mu again, and lam grows
        # (4 periods: about 4 times the credit; 1 or 2 periods: it diverges).
        widths = [8] * 9
        layers, head = make_modules("res", "tanh", widths)
        x, noise = sample(widths, 0)
        with torch.no_grad():
            _, y_hat = forward(layers, head, x)
        y = y_hat + noise
        bound = 2 * max(c.abs().max() for c in credits(layers, head, x, y, MASK))
        slow = pc.TimeConstants(h=TAU.h, lam=TAU.lam, theta=1e9)
        chain = HandChain(layers, head, widths, tau=slow, learn_interval=learn_interval)
        run = start_manual(chain.net)
        chain.write(run, x, y, MASK)
        for _ in range(DEEP_STEPS // 100):
            run_steps(run, chain.from_next(), 100)
            nodes = run.state_dict()["nodes"]
            assert (
                max(nodes[f"l{i}"]["state"]["lam"].abs().max() for i in range(8))
                < bound
            )
        nodes = run.state_dict()["nodes"]
        moved = [layer.module for layer in chain.layers]  # the nodes' own modules
        expected = credits(moved, chain.out.module, x, y, MASK)
        for i, credit in enumerate(expected):
            close(nodes[f"l{i}"]["state"]["lam"], credit, 1e-8)


@pytest.mark.slow
class TestNodeBoundaries:
    def test_two_constraints_in_one_node_learn_as_in_two(
        self, start_manual: StartManual
    ):
        # Linear + tanh twice, as one node or as two: at the fixed point, the
        # update of theta is that of backpropagation either way.
        widths = [4, 5, 3]
        layers, head = make_modules("linear", "tanh", widths)
        x, y = sample(widths, 0)
        whole = nn.Sequential(*copy.deepcopy(layers))
        updated: list[dict[str, torch.Tensor]] = []
        for modules, chain_widths in [
            ([whole], [widths[0], widths[-1]]),
            ([copy.deepcopy(m) for m in layers], widths),
        ]:
            chain = HandChain(
                modules,
                copy.deepcopy(head),
                chain_widths,
                learn_interval=RELAX_STEPS * H,
            )
            run = start_manual(chain.net, timer=VirtualTimer())
            chain.write(run, x, y, MASK)
            run_steps(run, chain.from_next(), RELAX_STEPS)
            nodes = run.state_dict()["nodes"]
            names = [f"l{i}" for i in range(len(modules))]
            # The parameters by their names in nn.Sequential(*layers).
            params = {
                f"{i}.{key}" if len(names) > 1 else key: value
                for i, name in enumerate(names)
                for key, value in nodes[name]["state"]["module"].items()
            }
            updated.append(
                params
                | {f"head.{k}": v for k, v in nodes["out"]["state"]["module"].items()}
            )
        assert updated[0].keys() == updated[1].keys()
        for key, value in updated[0].items():
            close(value, updated[1][key], 1e-10)


@pytest.mark.slow
class TestPcLimit:
    def test_without_the_dual_the_fixed_point_is_that_of_pc(
        self, start_manual: StartManual
    ):
        # alpha = 0: lam stays exactly zero, and the activities settle where the
        # energy of predictive coding is stationary, away from the forward pass.
        widths = [4, 5, 3]
        layers, head = make_modules("linear", "tanh", widths)
        tau = pc.TimeConstants(h=TAU.h, lam=math.inf, theta=math.inf)
        chain = HandChain(layers, head, widths, tau=tau)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        for _ in range(RELAX_STEPS):
            run.step(H, *chain.from_next())
            for i in range(len(layers)):
                lam = run.state_dict()["nodes"][f"l{i}"]["state"]["lam"]
                assert torch.equal(lam, torch.zeros_like(lam))
        tree = run.state_dict()
        hs = [
            tree["nodes"][f"l{i}"]["ports"]["next"]["out"]["value"]["h"]
            for i in range(len(layers))
        ]
        g = [
            tree["nodes"]["l1"]["ports"]["prev"]["out"]["value"],
            tree["nodes"]["out"]["ports"]["prev"]["out"]["value"],
        ]
        for i in range(len(layers)):
            state = tree["nodes"][f"l{i}"]["state"]
            r = state["h"] - state["mu"]
            close(r, -g[i], 1e-10)  # r = -g
            assert r.abs().max() > 1e-3
        # The gradient of the energy of PC with respect to each h is zero.
        free = [h.clone().requires_grad_() for h in hs]
        inputs = [x, *free]
        energy = sum(
            0.5 * ((h - module(h_in)) ** 2).sum()
            for module, h_in, h in zip(layers, inputs[:-1], free, strict=True)
        ) + masked_mse(head(free[-1]), y, MASK)
        assert isinstance(energy, torch.Tensor)
        for grad in torch.autograd.grad(energy, free):
            close(grad, torch.zeros_like(grad), 1e-10)
        forward_hs, _ = forward(layers, head, x)
        assert (hs[-1] - forward_hs[-1]).abs().max() > 1e-3


class TestScalar:
    def test_lam_is_w2_times_the_error(self, start_manual: StartManual):
        # y_hat = w2 h, h = w1 x: at the fixed point lam = w2 (y - y_hat).
        w1, w2, x, y = 0.5, 2.0, 1.5, 1.0
        first, head = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            first.weight.fill_(w1)
            head.weight.fill_(w2)
        one = TensorSpec((1, 1), F64)
        inp = pc.Input(x=one, y=one, period=H)
        layer = pc.Layer(
            first.double(), act_spec=inp.act_spec, tau=NO_LEARNING, learn_interval=None
        )
        out = pc.Output(
            head.double(),
            act_spec=layer.act_spec,
            loss=pc.mse,
            tau=NO_LEARNING,
            learn_interval=None,
        )
        inp.next.connect(layer.prev)
        layer.next.connect(out.prev)
        run = start_manual(Group(inp=inp, layer=layer, out=out))
        run.write(
            {
                inp.prev: {
                    "x": torch.full((1, 1), x, dtype=F64),
                    "y": torch.full((1, 1), y, dtype=F64),
                    "mask": torch.ones(1, dtype=torch.bool),
                }
            }
        )
        run_steps(run, [out, layer, inp], 1500)
        lam = run.state_dict()["nodes"]["layer"]["state"]["lam"]
        close(lam, torch.full((1, 1), w2 * (y - w2 * w1 * x), dtype=F64), 1e-10)
        # The host reads the h the layer sent: the forward value.
        close(run.read(layer.next)["h"], torch.full((1, 1), w1 * x, dtype=F64), 1e-10)


@pytest.mark.slow
class TestSchedules:
    def test_every_order_reaches_the_same_fixed_point(self, start_manual: StartManual):
        widths = [4, 4, 4]
        x, y = sample(widths, 0)
        expected = credits(*make_modules("res", "tanh", widths), x, y, MASK)
        shuffle = random.Random(0)
        orders: list[Callable[[HandChain], list[Any]]] = [
            HandChain.from_next,
            HandChain.from_prev,
            lambda chain: shuffle.sample(chain.from_prev(), k=len(chain.from_prev())),
        ]
        for order in orders:
            layers, head = make_modules("res", "tanh", widths)
            chain = HandChain(layers, head, widths, tau=NO_LEARNING)
            run = start_manual(chain.net, timer=VirtualTimer())
            chain.write(run, x, y, MASK)
            for _ in range(RELAX_STEPS):
                run.step(H, *order(chain))
            nodes = run.state_dict()["nodes"]
            for i, credit in enumerate(expected):
                close(nodes[f"l{i}"]["state"]["lam"], credit, 1e-8)

    def test_every_order_learns_as_sgd(self, start_manual: StartManual):
        # Learning at intervals, twice, at the fixed point each time: whatever
        # the order, each update is one step of SGD, the second from the
        # parameters of the first. The jumps that the first update makes go up
        # the chain with the counts of jumps, and the chain relaxes to the next fixed
        # point.
        widths = [4, 5, 3]
        x, y = sample(widths, 0)
        layers, head = make_modules("linear", "tanh", widths)
        sgd = [copy.deepcopy(m) for m in [*layers, head]]
        tau = pc.TimeConstants(h=TAU.h, lam=TAU.lam, theta=RELAX_STEPS * H / 0.1)
        rate = RELAX_STEPS * H / tau.theta
        params = [p for m in sgd for p in m.parameters()]
        for _ in range(2):
            _, y_hat = forward(sgd[:-1], sgd[-1], x)
            grads = torch.autograd.grad(masked_mse(y_hat, y, MASK) / B, params)
            with torch.no_grad():
                for p, g in zip(params, grads, strict=True):
                    p -= rate * g
        shuffle = random.Random(0)
        orders: list[Callable[[HandChain], list[Any]]] = [
            HandChain.from_next,
            HandChain.from_prev,
            lambda chain: shuffle.sample(chain.from_prev(), k=len(chain.from_prev())),
        ]
        for order in orders:
            layers, head = make_modules("linear", "tanh", widths)
            chain = HandChain(
                layers, head, widths, tau=tau, learn_interval=RELAX_STEPS * H
            )
            run = start_manual(chain.net, timer=VirtualTimer())
            chain.write(run, x, y, MASK)
            for _ in range(2 * RELAX_STEPS):
                run.step(H, *order(chain))
            nodes = run.state_dict()["nodes"]
            for name, module in zip(["l0", "l1", "out"], sgd, strict=True):
                for key, value in module.state_dict().items():
                    close(nodes[name]["state"]["module"][key], value, 1e-10)


class TestWithoutTeacher:
    def test_h_is_the_forward_pass(self, start_manual: StartManual):
        # Without a target, nothing drives h from mu or lam from zero. Stepped
        # from the input, each node reads what the previous one sent in the same
        # step: in the first step, every node moves h with mu = f(h_in) to the
        # forward pass, and stays there.
        widths = [4, 5, 5]
        layers = [
            make_modules("linear", "tanh", widths[:2])[0][0],
            make_modules("res", "tanh", widths[1:], seed=1)[0][0],
        ]
        head = make_modules("linear", "tanh", widths, seed=2)[1]
        before = [copy.deepcopy(m.state_dict()) for m in [*layers, head]]
        chain = HandChain(layers, head, widths)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, torch.zeros(B, dtype=torch.bool))
        with torch.no_grad():
            hs, _ = forward(layers, head, x)
        for _ in range(10):
            run.step(H, *chain.from_prev())
            tree = run.state_dict()
            for i, h in enumerate(hs):
                assert torch.equal(
                    tree["nodes"][f"l{i}"]["ports"]["next"]["out"]["value"]["h"], h
                )
                assert torch.equal(tree["nodes"][f"l{i}"]["state"]["h"], h)
                lam = tree["nodes"][f"l{i}"]["state"]["lam"]
                assert torch.equal(lam, torch.zeros_like(lam))
        tree = run.state_dict()
        names = [*(f"l{i}" for i in range(len(layers))), "out"]
        for name, state in zip(names, before, strict=True):
            for key, value in state.items():
                assert torch.equal(tree["nodes"][name]["state"]["module"][key], value)

    def test_a_new_input_takes_one_step_per_sender(self, start_manual: StartManual):
        # At the forward pass of a first input (h - mu = 0), a second one is
        # stepped from the output: each node reads the previous step's values,
        # so the new forward pass reaches l1 in three steps (Input, l0, l1).
        widths = [4, 5, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = HandChain(layers, head, widths)
        run = start_manual(chain.net)
        no_target = torch.zeros(B, dtype=torch.bool)
        x, y = sample(widths, 0)
        chain.write(run, x, y, no_target)
        run.step(H, *chain.from_prev())
        x, y = sample(widths, 1)
        chain.write(run, x, y, no_target)
        with torch.no_grad():
            hs, _ = forward(layers, head, x)
        for _ in range(2):
            run.step(H, *chain.from_next())
        assert not torch.equal(
            run.state_dict()["nodes"]["l1"]["ports"]["next"]["out"]["value"]["h"],
            hs[-1],
        )
        run.step(H, *chain.from_next())
        assert torch.equal(
            run.state_dict()["nodes"]["l1"]["ports"]["next"]["out"]["value"]["h"],
            hs[-1],
        )


# Input -> Layer -> Output that learns K steps after each new input, and a host
# that writes a new input every M steps.
K, M = 3, 5


class Learning:
    """``HandChain`` of ``widths``, stepped from the output with a host that writes a
    new labeled input (at a time other than zero) every ``M`` steps.

    With ``state``, the run starts from it, as if it had taken ``steps`` steps.
    """

    def __init__(
        self,
        start_manual: StartManual,
        *,
        widths: Sequence[int] = (4, 5),
        max_dt: float | None = None,
        seed: int = 0,
        state: dict[str, Any] | None = None,
        steps: int = 0,
    ) -> None:
        self.widths = list(widths)
        layers, head = make_modules("linear", "tanh", self.widths, seed=seed)
        self.chain = HandChain(
            layers, head, self.widths, learn_interval=K * H, max_dt=max_dt
        )
        # A timer of its own, as the restored run has.
        timer = VirtualTimer() if state is None else None
        self.run: ManualRun = start_manual(self.chain.net, timer=timer, state=state)
        self.steps = steps

    def step(self, masks: Callable[[int], torch.Tensor] = lambda _: MASK) -> None:
        self.run.timer.advance(H)
        if self.steps % M == 0:
            x, y = sample(self.widths, self.steps)
            self.chain.write(self.run, x, y, masks(self.steps // M))
        self.steps += 1
        self.run.step(0.0, *self.chain.from_next())


def h_before_mu(state: dict[str, Any]) -> torch.Tensor:
    """The ``h`` of a layer in a step, from its state before the step, once it
    took the jump of its ``theta`` of the step before (if any)."""
    return state["h"] + state["jump"] if state["jumped"] else state["h"]


def layer_update(
    tree: dict[str, Any], module: nn.Module, elapsed: float, *, reanchor: bool = False
) -> dict[str, torch.Tensor]:
    """The parameters of ``l0`` after one update from the state ``tree`` taken
    before the step: ``theta - (elapsed / tau_theta) g_theta / B``. With
    ``reanchor``, ``h`` moves with ``mu`` in the step (a new input, or a jump of
    ``h_in``), so ``h - mu`` is ``h - mu_last``.

    It computes ``g_theta`` with ``functional.vjp`` and moves the parameters
    with ``functional.learn_``, which their own tests check against finite
    differences and the written-out step: the tests that use it check when the
    update comes and how much time weighs it.
    """
    module = copy.deepcopy(module)
    module.load_state_dict(tree["nodes"]["l0"]["state"]["module"])
    params = list(module.parameters())
    state = tree["nodes"]["l0"]["state"]
    sent = tree["nodes"]["inp"]["ports"]["next"]["out"]["value"]
    labeled = sent["mask"][:, None]
    h = h_before_mu(state)

    def delta(mu: torch.Tensor) -> torch.Tensor:
        r = h - (state["mu"] if reanchor else mu)
        return torch.where(labeled, r + state["lam"], 0.0)

    _, _, g_theta = pc.functional.vjp(module, sent["h"], delta, params)
    pc.functional.learn_(params, g_theta, elapsed, TAU, B)
    return module.state_dict()


def equal_states(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return all(torch.equal(a[key], b[key]) for key in a)


class TestLearnInterval:
    def test_theta_moves_k_steps_after_each_input_by_the_elapsed_time(
        self, start_manual: StartManual
    ):
        # The host writes at steps 1, 6, 11, ...; the layer reads the input of
        # the previous step, so it sees them at steps 2, 7, 12, ..., and learns
        # K steps after each write, at steps 4, 9, 14, ...; the time since its
        # last update spans the switch. Its first update (step 1) sees the zeros
        # carried before the first input, which changes at step 2 before the
        # layer learned from it, so the time of step 1 is dropped.
        learning = Learning(start_manual)
        module = learning.chain.layers[0].module
        learned: list[tuple[int, float]] = []
        last = 1
        for s in range(1, 21):
            before = learning.run.state_dict()
            learning.step()
            after = learning.run.state_dict()["nodes"]["l0"]["state"]["module"]
            if equal_states(after, before["nodes"]["l0"]["state"]["module"]):
                continue
            elapsed = (s - last) * H
            expected = layer_update(before, module, elapsed)
            for key, value in expected.items():
                close(after[key], value, 1e-14)
            learned.append((s, elapsed))
            last = s
        assert [s for s, _ in learned] == [4, 9, 14, 19]
        assert [round(e / H) for _, e in learned] == [3, 5, 5, 5]

    def test_a_first_input_written_at_time_zero_is_learned(
        self, start_manual: StartManual
    ):
        # Its time 0 is also that of the zeros carried before it, so the nodes
        # do not see it as new, and learn 2 K steps after time 0 as for the
        # zeros; before the input arrives, the mask (all False) keeps them from
        # learning. The layer and the output learn in the same step. (The
        # input's count of jumps still moves h with it: at step K, the credit would not
        # have reached the layer yet, delta = 0.)
        widths = [4, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = HandChain(layers, head, widths, learn_interval=2 * K * H)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        assert chain.write(run, x, y, MASK) == 0
        names = ["l0", "out"]
        changed: dict[str, list[int]] = {name: [] for name in names}
        for s in range(1, 4 * K + 3):
            before = run.state_dict()
            run.step(H, *chain.from_next())
            if s == 2 * K:
                expected = layer_update(before, layers[0], s * H)
                after = run.state_dict()["nodes"]["l0"]["state"]["module"]
                for key, value in expected.items():
                    close(after[key], value, 1e-14)
                assert not equal_states(after, before["nodes"]["l0"]["state"]["module"])
            for name in names:
                after = run.state_dict()["nodes"][name]["state"]["module"]
                if not equal_states(after, before["nodes"][name]["state"]["module"]):
                    changed[name].append(s)
        assert changed == {"l0": [2 * K, 4 * K], "out": [2 * K, 4 * K]}

    def test_without_learning_the_interval_changes_nothing(
        self, start_manual: StartManual
    ):
        # theta = inf: the nodes never learn, whatever learn_interval, so theta
        # stays and no jump of theta goes up the chain. The run is that of
        # learn_interval=None bit for bit, but for the state of the clocks.
        widths = [4, 5, 5]
        trees: list[dict[str, Any]] = []
        for learn_interval in (None, 2 * H):
            layers, head = make_modules("linear", "tanh", widths)
            initial = [copy.deepcopy(m.state_dict()) for m in [*layers, head]]
            chain = HandChain(
                layers, head, widths, tau=NO_LEARNING, learn_interval=learn_interval
            )
            run = start_manual(chain.net, timer=VirtualTimer())
            for s in range(30):
                if s % 7 == 0:
                    x, y = sample(widths, s)
                    chain.write(run, x, y, MASK)
                run.step(H, *chain.from_next())
            tree = run.state_dict()
            for node in tree["nodes"].values():
                node["state"].pop("clock", None)
                node["state"].pop("generator")  # seeded by chance
            trees.append(tree)
            for module, state in zip([*layers, head], initial, strict=True):
                assert equal_states(module.state_dict(), state)
        assert_same_tree(trees[1], trees[0])

    def test_an_input_that_changes_before_it_is_learned_drops_its_time(
        self, start_manual: StartManual
    ):
        # A new input every step, faster than learn_interval (K steps): the
        # layer never learns while they come, and each one drops the time
        # counted for the one before. Once the inputs stop (the last, written
        # at step 7, is seen at step 8), the update K steps after the write is
        # weighted by the time since step 8, included (without the drop, it
        # would be 10 steps).
        learning = Learning(start_manual)
        module = learning.chain.layers[0].module
        run = learning.run

        def step(s: int) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
            run.timer.advance(H)
            if s <= 7:
                x, y = sample(learning.widths, s)
                learning.chain.write(run, x, y, MASK)
            before = run.state_dict()
            run.step(0.0, *learning.chain.from_next())
            return before, run.state_dict()["nodes"]["l0"]["state"]["module"]

        for s in range(1, 7 + K):
            before, after = step(s)
            assert equal_states(after, before["nodes"]["l0"]["state"]["module"]), s
        before, after = step(7 + K)
        expected = layer_update(before, module, K * H)
        for key, value in expected.items():
            close(after[key], value, 1e-14)


def learned_steps(
    chain: HandChain,
    widths: Sequence[int],
    run: ManualRun,
    order: Callable[[], Sequence[Any]],
    steps: int,
) -> dict[str, list[int]]:
    """Step ``run`` ``steps`` times with ``chain``'s host writing a labeled input
    at steps 1, 13, ...; return the steps at which each node learned (the time
    since its last update of theta went back to zero: its parameters may not
    move, before the credit reaches it), once it has seen an input."""
    names = [*(f"l{i}" for i in range(len(chain.layers))), "out"]
    learned: dict[str, list[int]] = {name: [] for name in names}
    for s in range(1, steps + 1):
        run.timer.advance(H)
        if s % 12 == 1:
            x, y = sample(widths, s)
            chain.write(run, x, y, MASK)
        run.step(0.0, *order())
        nodes = run.state_dict()["nodes"]
        for name in names:
            # Before the first input, the zeros carried (time 0) skip theirs.
            clock = nodes[name]["state"]["clock"]
            if clock["seen_ns"] != 0 and clock["elapsed"] == 0.0:
                learned[name].append(s)
    return learned


class TestLearnTime:
    def test_the_nodes_on_a_path_learn_in_the_same_step(
        self, start_manual: StartManual
    ):
        # learn_interval = 8 steps after each write, longer than the input takes
        # to reach the output (4 steps from the output down): every node learns
        # at steps 9 and 21, whatever the order of the updates.
        shuffle = random.Random(0)
        for name in ("from_next", "from_prev", "random"):
            widths = [4, 5, 5, 5]
            layers, head = make_modules("linear", "tanh", widths)
            chain = HandChain(layers, head, widths, learn_interval=8 * H)
            run = start_manual(chain.net, timer=VirtualTimer())

            def order(chain: HandChain = chain, name: str = name) -> list[Any]:
                if name == "random":
                    nodes = chain.from_prev()
                    return shuffle.sample(nodes, k=len(nodes))
                return getattr(chain, name)()

            learned = learned_steps(chain, widths, run, order, 24)
            assert learned == dict.fromkeys(learned, [9, 21]), name

    def test_a_node_does_not_learn_in_the_update_that_first_sees_an_input(
        self, start_manual: StartManual
    ):
        # learn_interval = 2 steps, shorter than the path: stepped from the
        # output, l0 sees the input (written at step 1) at step 2 and learns at
        # steps 3 and 5, but l1, l2 and the output see it at steps 3, 4 and 5,
        # at or past the time to learn, and learn in the update after; then at
        # the next time on the grid of the write (step 5), if any.
        widths = [4, 5, 5, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = HandChain(layers, head, widths, learn_interval=2 * H)
        run = start_manual(chain.net, timer=VirtualTimer())
        learned = learned_steps(chain, widths, run, chain.from_next, 6)
        assert learned == {"l0": [3, 5], "l1": [4, 5], "l2": [5], "out": [6]}


def relaxed(
    before: dict[str, Any],
    after: dict[str, Any],
    module: nn.Module,
    dt: float,
    *,
    reanchor: bool,
    node: str = "l0",
    h_in: str = "inp",
    g: str = "out",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``h`` and ``lam`` of the layer ``node`` after one Euler step of ``dt``
    from the state ``before``, written out, stepped from the output: the node
    reads ``h_in`` as the previous node ``h_in`` sent it before the step, and
    ``g`` as the next node ``g`` sent it in the step. ``h`` first takes the
    jump of ``theta`` of the step before (if any), then, with ``reanchor``,
    moves with ``mu``."""
    state = before["nodes"][node]["state"]
    module = copy.deepcopy(module)
    module.load_state_dict(state["module"])
    with torch.no_grad():
        mu = module(before["nodes"][f"{h_in}"]["ports"]["next"]["out"]["value"]["h"])
    h, lam = h_before_mu(state), state["lam"]
    if reanchor:
        h = mu + (h - state["mu"])
    g_sent = after["nodes"][f"{g}"]["ports"]["prev"]["out"]["value"]
    h = h - (dt / TAU.h) * (h - mu + lam + g_sent)  # tau_h dh/dt = -(r + lam + g)
    lam = lam + (dt / TAU.lam) * (h - mu)  # tau_lam dlam/dt = r
    return h, lam


class TestMaxDt:
    @pytest.mark.parametrize(("max_dt", "counted"), [(None, 2 * H), (3 * H, 3 * H)])
    def test_a_long_dt_counts_max_dt(
        self, start_manual: StartManual, max_dt: float | None, counted: float
    ):
        # The layer learns at step 4 (K steps after the write at step 1); dt =
        # 5 H at step 5, which is past the next time to learn (step 7): h takes
        # the jump of theta and then h and lam take one step of max_dt
        # (tau.h / 2 = 2 H by default), and the update of theta is weighted by
        # max_dt only.
        learning = Learning(start_manual, max_dt=max_dt)
        module = learning.chain.layers[0].module
        for _ in range(4):
            learning.step()
        before = learning.run.state_dict()
        assert before["nodes"]["l0"]["state"]["jumped"]
        learning.run.timer.advance(4 * H)
        learning.step()
        after = learning.run.state_dict()
        h, lam = relaxed(before, after, module, counted, reanchor=False)
        close(after["nodes"]["l0"]["state"]["h"], h, 1e-14)
        close(after["nodes"]["l0"]["state"]["lam"], lam, 1e-14)
        expected = layer_update(before, module, counted)
        for key, value in expected.items():
            close(after["nodes"]["l0"]["state"]["module"][key], value, 1e-14)

    def test_the_default_keeps_the_step_of_lam_stable(self):
        # dt / tau.lam <= 3 as well as dt / tau.h <= 1/2: max_dt is 1.5e-3, which
        # a period must not pass.
        layer = functools.partial(
            pc.Layer, act_spec=activity(TensorSpec((B, 4))), learn_interval=None
        )
        fast_dual = pc.TimeConstants(h=4e-3, lam=0.5e-3, theta=1.0)
        layer(nn.Linear(4, 5), tau=fast_dual, period=1.5e-3)
        with pytest.raises(ValueError, match="max_dt"):
            layer(nn.Linear(4, 5), tau=fast_dual, period=1.6e-3)
        no_dual = pc.TimeConstants(h=4e-3, lam=math.inf, theta=1.0)
        layer(nn.Linear(4, 5), tau=no_dual, period=2e-3)
        with pytest.raises(ValueError, match="max_dt"):
            layer(nn.Linear(4, 5), tau=no_dual, period=2.1e-3)


class TestJumps:
    def test_h_moves_with_new_inputs_and_with_the_jumps_of_theta(
        self, start_manual: StartManual
    ):
        # The layer sees the inputs at steps 2, 7, ... and learns at 4, 9, ...:
        # h moves with mu at steps 2 and 7 (new inputs); at step 5 it takes the
        # jump f(h_in; theta_new) - f(h_in; theta) that learning made at step 4,
        # at the h_in of step 4, and only relaxes otherwise. Each of these
        # advances the jumps it sends (and the first update, from zeros).
        learning = Learning(start_manual)
        module = learning.chain.layers[0].module
        jumps: list[int] = []
        for s in range(1, 9):
            before = learning.run.state_dict()
            learning.step()
            after = learning.run.state_dict()
            jumps.append(
                int(after["nodes"]["l0"]["ports"]["next"]["out"]["value"]["jumps"])
            )
            if s == 1:
                continue  # the first update moves h from zeros
            h, lam = relaxed(before, after, module, H, reanchor=s in (2, 7))
            close(after["nodes"]["l0"]["state"]["h"], h, 1e-14)
            close(after["nodes"]["l0"]["state"]["lam"], lam, 1e-14)
            assert after["nodes"]["l0"]["state"]["jumped"] == (s == 4)
            if s == 4:
                h_in = before["nodes"]["inp"]["ports"]["next"]["out"]["value"]["h"]
                old, new = copy.deepcopy(module), copy.deepcopy(module)
                old.load_state_dict(before["nodes"]["l0"]["state"]["module"])
                new.load_state_dict(after["nodes"]["l0"]["state"]["module"])
                with torch.no_grad():
                    close(after["nodes"]["l0"]["state"]["mu"], new(h_in), 1e-14)
                    close(
                        after["nodes"]["l0"]["state"]["jump"],
                        new(h_in) - old(h_in),
                        1e-14,
                    )
        assert jumps == [1, 2, 2, 2, 3, 3, 4, 4]

    def test_a_jump_below_moves_the_next_h_with_r_unchanged(
        self, start_manual: StartManual
    ):
        # l1 reads the first input at step 3. Both layers learn at step 4 and
        # take their own jumps at step 5; l0 sends its jump with a new count of jumps,
        # which l1 reads at step 6 and moves with: h - mu_last before the move
        # is h - mu after it. At step 8, l1 reads a new input and the count l0
        # advanced for it at step 7: one move, one new count.
        learning = Learning(start_manual, widths=(4, 5, 5))
        module = learning.chain.layers[1].module
        jumps: list[int] = []
        for s in range(1, 10):
            before = learning.run.state_dict()
            learning.step()
            after = learning.run.state_dict()
            jumps.append(
                int(after["nodes"]["l1"]["ports"]["next"]["out"]["value"]["jumps"])
            )
            if s < 3:
                continue  # the moves from zeros and the zeros of l0
            h, lam = relaxed(
                before,
                after,
                module,
                H,
                reanchor=s in (3, 6, 8),
                node="l1",
                h_in="l0",
                g="out",
            )
            close(after["nodes"]["l1"]["state"]["h"], h, 1e-14)
            close(after["nodes"]["l1"]["state"]["lam"], lam, 1e-14)
        assert jumps == [1, 2, 3, 3, 4, 5, 5, 6, 6]

    def test_learning_in_every_update_moves_no_node_above(
        self, start_manual: StartManual
    ):
        # With learn_interval=None, theta moves by dt in every update: each layer
        # takes its own jump, but advances its jumps only for the new input.
        widths = [4, 5, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = HandChain(layers, head, widths)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        run_steps(run, chain.from_next(), 5)
        for _ in range(10):
            run.step(H, *chain.from_next())
            tree = run.state_dict()
            for i in range(2):
                assert tree["nodes"][f"l{i}"]["state"]["jumped"]
                assert (
                    int(
                        tree["nodes"][f"l{i}"]["ports"]["next"]["out"]["value"]["jumps"]
                    )
                    == 2 + i
                )


class TestUnlabeled:
    @pytest.mark.parametrize("learn_interval", [None, K * H])
    def test_no_theta_moves_while_every_mask_is_false(
        self, start_manual: StartManual, learn_interval: float | None
    ):
        # With mask False for every sample of every input, no node learns: h
        # moves, and the theta of every node stays the same to the bit.
        widths = [4, 5, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = HandChain(layers, head, widths, learn_interval=learn_interval)
        run = start_manual(chain.net, timer=VirtualTimer())
        before = run.state_dict()["nodes"]
        no_target = torch.zeros(B, dtype=torch.bool)
        for k in range(6 * M):
            if k % M == 0:
                x, y = sample(widths, k)
                chain.write(run, x, y, no_target)
            run.step(H, *chain.from_next())
        after = run.state_dict()["nodes"]
        learners = [path for path, node in after.items() if "module" in node["state"]]
        assert sorted(learners) == ["l0", "l1", "out"]
        for path in learners:
            assert_same_tree(
                after[path]["state"]["module"], before[path]["state"]["module"]
            )
        assert not torch.equal(after["l1"]["state"]["h"], before["l1"]["state"]["h"])

    def test_no_target_skips_the_update_and_drops_the_time(
        self, start_manual: StartManual
    ):
        # The first input has no target: the layer does not learn at step 4 and
        # its next update (step 9) is weighted by the 5 steps since then only.
        learning = Learning(start_manual)
        module = learning.chain.layers[0].module
        no_target = torch.zeros(B, dtype=torch.bool)

        def masks(k: int) -> torch.Tensor:
            return no_target if k == 0 else MASK

        for _ in range(3):
            learning.step(masks)
        before = learning.run.state_dict()
        learning.step(masks)  # step 4
        after = learning.run.state_dict()["nodes"]["l0"]["state"]["module"]
        assert equal_states(after, before["nodes"]["l0"]["state"]["module"])
        for _ in range(4):
            learning.step(masks)
        before = learning.run.state_dict()
        learning.step(masks)  # step 9
        after = learning.run.state_dict()["nodes"]["l0"]["state"]["module"]
        expected = layer_update(before, module, 5 * H)
        for key, value in expected.items():
            close(after[key], value, 1e-14)

    def test_a_sample_without_a_target_sends_back_nothing(
        self, start_manual: StartManual
    ):
        # The first input has targets for all samples, the second for samples 0
        # and 2 only: sample 1 still holds the delta of the first input, but
        # sends back no credit for it and adds nothing to the update of theta.
        learning = Learning(start_manual)
        module = learning.chain.layers[0].module

        def masks(k: int) -> torch.Tensor:
            return torch.ones(B, dtype=torch.bool) if k == 0 else MASK

        for _ in range(8):
            learning.step(masks)
        before = learning.run.state_dict()
        learning.step(masks)  # step 9: learns from the second input
        after = learning.run.state_dict()
        state = before["nodes"]["l0"]["state"]
        assert (state["h"] - state["mu"] + state["lam"])[1].abs().max() > 1e-6
        g_in = after["nodes"]["l0"]["ports"]["prev"]["out"]["value"]
        assert torch.equal(g_in[1], torch.zeros_like(g_in[1]))
        expected = layer_update(before, module, 5 * H)
        for key, value in expected.items():
            close(after["nodes"]["l0"]["state"]["module"][key], value, 1e-14)


def without_generators(tree: dict[str, Any]) -> dict[str, Any]:
    """``tree`` without the state of the nodes' generators (seeded by chance,
    and unused by the pc nodes)."""
    for node in tree["nodes"].values():
        node["state"].pop("generator")
    return tree


class TestState:
    def test_a_restored_run_goes_on_as_the_same_run(self, start_manual: StartManual):
        # Taken at step 6, mid-phase, right after the input sent the input it
        # was written (the layer has read the earlier one) and the layer took
        # the jump of its theta.
        whole = Learning(start_manual)
        for _ in range(16):
            whole.step()
        first = Learning(start_manual)
        for _ in range(6):
            first.step()
        tree = first.run.state_dict()
        assert {
            "clock",
            "jump",
            "jumped",
            "jumps",
            "seen_jumps",
            "mu",
            "module",
            "h",
            "lam",
        } <= set(tree["nodes"]["l0"]["state"])
        assert {"jumps", "seen_write"} <= set(tree["nodes"]["inp"]["state"])
        first.run.shutdown()
        restored = Learning(start_manual, seed=1, state=tree, steps=6)
        for _ in range(10):
            restored.step()
        assert_same_tree(
            without_generators(restored.run.state_dict()),
            without_generators(whole.run.state_dict()),
        )
