import copy
import math
import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast, override

import pytest
import torch
from torch import nn

import mujou.pc as pc
from mujou import DictSpec, Group, TensorSpec
from mujou.testing import ManualRun, VirtualTimer

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual, Step

H = 1e-3  # the step of the `step` fixture, in seconds

B = 3
F64 = torch.float64
N_OUT = 2
TAU = pc.TimeConstants(h=4e-3, lam=8e-3, theta=0.05)
NO_LEARNING = pc.TimeConstants(h=TAU.h, lam=TAU.lam, theta=math.inf)
MASK = torch.tensor([True, False, True])


class ResBlock(nn.Module):
    """``h + fc2(act(fc1(h)))``: a residual block as the ``f`` of one node."""

    def __init__(self, n: int, act: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n, n), nn.Linear(n, n)
        self.act = act

    @override
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(self.act(self.fc1(h)))


class Act(nn.Module):
    def __init__(self, act: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.act = act

    @override
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.act(h)


def identity(h: torch.Tensor) -> torch.Tensor:
    return h


ACTS = {"linear": identity, "tanh": torch.tanh}


def make_modules(
    block: str, act: str, widths: Sequence[int], seed: int = 0
) -> tuple[list[nn.Module], nn.Module]:
    """The modules of the layers and the head, float64: ``Linear`` then the
    activation, or a residual block (of constant width)."""
    layers: list[nn.Module] = []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for n_in, n_out in zip(widths[:-1], widths[1:], strict=True):
            if block == "linear":
                layers.append(nn.Sequential(nn.Linear(n_in, n_out), Act(ACTS[act])))
            else:
                assert n_in == n_out
                layers.append(ResBlock(n_in, ACTS[act]))
        head = nn.Linear(widths[-1], N_OUT)
    return [m.double() for m in layers], head.double()


class Chain:
    """``Input -> Layer -> ... -> Output (mse)`` over ``B`` samples, float64."""

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
        carry = pc.carry_spec(TensorSpec((B, N_OUT), F64), batch=B)
        self.inp = pc.Input((B, widths[0]), carry=carry, period=H, dtype=F64)
        self.layers = [
            pc.Layer(
                module,
                input_shape=(B, n_in),
                carry=carry,
                tau=tau,
                learn_interval=learn_interval,
                max_dt=max_dt,
                dtype=F64,
            )
            for module, n_in in zip(layers, widths[:-1], strict=True)
        ]
        self.out = pc.Output(
            head,
            loss=pc.mse,
            input_shape=(B, widths[-1]),
            carry=carry,
            tau=tau,
            learn_interval=learn_interval,
            max_dt=max_dt,
            dtype=F64,
        )
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
        self, run: ManualRun[Any], x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> int:
        return run.write({self.inp.prev: {"x": x, "y": y, "mask": mask}}).ns


def activity(h: TensorSpec) -> DictSpec[pc.Activity]:
    return DictSpec(pc.Activity, h=h, epoch=TensorSpec((), torch.int64))


def sample(widths: Sequence[int], seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(B, widths[0], dtype=F64, generator=generator)
    y = torch.randn(B, N_OUT, dtype=F64, generator=generator)
    return x, y


def masked_mse(
    y_hat: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The loss of the output node, written out: sum_b mask_b 0.5 ||y_b - y_hat_b||^2."""
    return (0.5 * (y - y_hat) ** 2).sum(dim=1)[mask].sum()


def forward(
    layers: Sequence[nn.Module], head: nn.Module, x: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """``[h_1, ..., h_L]`` and ``y_hat`` of the forward pass, in the graph."""
    hs = [x]
    for module in layers:
        hs.append(module(hs[-1]))
    return hs[1:], head(hs[-1])


def credits(
    layers: Sequence[nn.Module],
    head: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> list[torch.Tensor]:
    """``-d loss / d h_i`` of backpropagation through the whole network."""
    hs, y_hat = forward(layers, head, x)
    grads = torch.autograd.grad(masked_mse(y_hat, y, mask), hs)
    return [-g for g in grads]


class Clock:
    """When a node learns, as ``specs/pc-module.md`` states it: from the time
    ``time`` (ns) the host wrote the input, every ``interval`` (ns) after it,
    never in the update that first sees the input; ``elapsed`` is the time since
    the last update, dropped when the input changes before the node learned."""

    def __init__(self, interval: int | None) -> None:
        self.interval = interval
        self.seen, self.learn_at, self.elapsed, self.learned_input = -1, 0, 0.0, False

    def tick(self, dt: float, time: int, now: int) -> tuple[bool, bool]:
        new = time != self.seen
        if new:
            self.seen = time
            if self.interval is not None:
                self.learn_at = time + self.interval
            if not self.learned_input:
                self.elapsed = 0.0
            self.learned_input = False
        self.elapsed += dt
        return new, self.interval is None or (not new and now >= self.learn_at)

    def learned(self, now: int) -> None:
        self.elapsed, self.learned_input = 0.0, True
        if self.interval is not None:
            behind = now - self.learn_at
            self.learn_at += (behind // self.interval + 1) * self.interval


class Reference:
    """PC-ALM in plain PyTorch: every node moves at once from the values of the
    previous step (a Jacobi update).

    ``h[0]`` is the ``x`` the input sent and ``h[i + 1]`` the activity of layer
    ``i`` (its state, which it sent), each with the epoch, target, mask and time
    it carries; all start at zero (mask False, time 0), as the ports do. A layer
    moves its ``h`` with its prediction, ``h <- mu + (h - mu_last)``, when the
    carried time or the epoch of ``h_in`` changes, and then advances its epoch.
    When it learns, it keeps the jump ``f(h_in; theta_new) - f(h_in; theta)``
    and adds it to ``h`` in its next step, advancing its epoch for it only when
    it learns at intervals. The input's epoch counts the writes.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        head: nn.Module,
        widths: Sequence[int],
        learn_interval: float | None = None,
    ) -> None:
        self.layers = [copy.deepcopy(m) for m in layers]
        self.head = copy.deepcopy(head)
        interval = None if learn_interval is None else round(learn_interval * 1e9)
        self.clocks = [Clock(interval) for _ in widths]  # the layers, then the head
        self.lam = [torch.zeros(B, n, dtype=F64) for n in widths[1:]]
        self.mu_last = [torch.zeros(B, n, dtype=F64) for n in widths[1:]]
        self.jump: list[torch.Tensor | None] = [None for _ in widths[1:]]
        self.seen_epoch = [0 for _ in widths[1:]]
        self.input_seen = -1
        self.epoch = [0 for _ in widths]  # what each sender sent
        self.h = [torch.zeros(B, n, dtype=F64) for n in widths]
        self.carried = [
            (torch.zeros(B, N_OUT, dtype=F64), torch.zeros(B, dtype=torch.bool), 0)
            for _ in widths
        ]
        self.g_in: list[torch.Tensor] = [torch.zeros(B, n, dtype=F64) for n in widths]

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        time: int,
        dt: float,
        now: int,
    ) -> None:
        tau = TAU
        # The output: the gradient of the masked loss.
        y_c, mask_c, time_c = self.carried[-1]
        _, learn = self.clocks[-1].tick(dt, time_c, now)
        h_in = self.h[-1].clone().requires_grad_()
        params = list(self.head.parameters())
        self.y_hat = self.head(h_in)
        g_next, *g_theta = torch.autograd.grad(
            masked_mse(self.y_hat, y_c, mask_c), [h_in, *params]
        )
        self.y_hat = self.y_hat.detach()
        self.prediction_carried = (y_c, mask_c)
        self.g_in[-1] = g_next
        if learn:
            self._learn(params, g_theta, mask_c, self.clocks[-1], now)
        # The layers, from the last: each with the g the next one made from the
        # previous values.
        h_new, epoch_new = list(self.h), list(self.epoch)
        for i in reversed(range(len(self.layers))):
            _, mask_i, time_i = self.carried[i]
            clock = self.clocks[i]
            new, learn = clock.tick(dt, time_i, now)
            if learn and not mask_i.any():
                clock.learned(now)
                learn = False
            h = self.h[i + 1]
            jump = self.jump[i]
            if jump is not None:  # theta moved in the step before
                h = h + jump
                self.jump[i] = None
            reanchor = new or self.epoch[i] != self.seen_epoch[i]
            self.seen_epoch[i] = self.epoch[i]
            module = self.layers[i]
            h_in = self.h[i].clone().requires_grad_()
            params = list(module.parameters())
            mu = module(h_in)
            # r = h - mu, which is h - mu_last when h moves with mu below.
            r = h - (self.mu_last[i] if reanchor else mu.detach())
            delta = torch.where(mask_i[:, None], r + self.lam[i], 0.0)
            g_in, *g_theta = torch.autograd.grad(mu, [h_in, *params], -delta)
            mu = mu.detach()
            if reanchor:
                h = mu + (h - self.mu_last[i])
            h = h - (dt / tau.h) * (h - mu + self.lam[i] + g_next)
            self.lam[i] = self.lam[i] + (dt / tau.lam) * (h - mu)
            self.mu_last[i] = mu
            h_new[i + 1] = h
            if reanchor or (jump is not None and clock.interval is not None):
                epoch_new[i + 1] = self.epoch[i + 1] + 1
            if learn:
                self._learn(params, g_theta, mask_i, clock, now)
                with torch.no_grad():
                    moved = module(self.h[i])
                self.jump[i] = moved - mu
                self.mu_last[i] = moved
            self.g_in[i] = g_in
            g_next = g_in
        h_new[0] = x
        if time != self.input_seen:  # a write the input has not seen
            self.input_seen = time
            epoch_new[0] = self.epoch[0] + 1
        self.h, self.epoch = h_new, epoch_new
        self.carried = [(y, mask, time), *self.carried[:-1]]

    def _learn(
        self,
        params: Sequence[torch.Tensor],
        g_theta: list[torch.Tensor],
        mask: torch.Tensor,
        clock: Clock,
        now: int,
    ) -> None:
        if mask.any():
            with torch.no_grad():
                for p, g in zip(params, g_theta, strict=True):
                    p -= (clock.elapsed / TAU.theta) * g / B
        clock.learned(now)


def close(actual: torch.Tensor, expected: torch.Tensor, atol: float) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)


def run_steps(run: ManualRun[Any], step: Step, order: Sequence[Any], n: int) -> None:
    for _ in range(n):
        step(run, *order)


class TestConstruction:
    def _layer(self, **kwargs: Any) -> pc.Layer:
        args: dict[str, Any] = {
            "input_shape": (B, 4),
            "carry": pc.carry_spec(TensorSpec((B, N_OUT)), batch=B),
            "tau": TAU,
        }
        module = kwargs.pop("module", nn.Linear(4, 5))
        return pc.Layer(module, **(args | kwargs))

    def test_the_output_shape_comes_from_the_module(self):
        layer = self._layer(module=nn.Sequential(nn.Linear(4, 6), nn.Tanh()))
        assert layer.next.send_spec == activity(TensorSpec((B, 6)))
        assert layer.prev.recv_spec == activity(TensorSpec((B, 4)))
        assert layer.h.shape == layer.lam.shape == (B, 6)

    def test_the_dtype_comes_from_the_module(self):
        layer = self._layer(module=nn.Linear(4, 5).double(), dtype=F64)
        assert layer.next.send_spec == activity(TensorSpec((B, 5), F64))

    def test_the_period_defaults_to_a_quarter_of_tau_h(self):
        assert self._layer().period == pytest.approx(TAU.h / 4)
        assert self._layer(period=2e-3).period == 2e-3

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"input_shape": (B + 1, 4)}, "input_shape"),
            ({"module": nn.Flatten(0)}, "output of module"),
            ({"module": nn.Linear(4, 5, device="meta")}, "meta"),
            ({"max_dt": 0.0}, "max_dt"),
            ({"max_dt": math.inf}, "max_dt"),
            ({"learn_interval": 0.0}, "learn_interval"),
            ({"learn_interval": math.inf}, "learn_interval"),
            ({"period": 3e-3}, "max_dt"),  # longer than tau.h / 2
            ({"period": 2e-3, "max_dt": 1e-3}, "max_dt"),
        ],
    )
    def test_value_errors(self, kwargs: dict[str, Any], match: str):
        with pytest.raises(ValueError, match=match):
            self._layer(**kwargs)

    def test_type_errors(self):
        with pytest.raises(TypeError, match="TimeConstants"):
            self._layer(tau=(4e-3, 8e-3, 1.0))
        with pytest.raises(TypeError, match="carry"):
            self._layer(carry=TensorSpec((B,)))

    @pytest.mark.parametrize("name", ["learn_interval", "max_dt"])
    def test_a_bool_is_not_a_time(self, name: str):
        with pytest.raises(TypeError, match=name):
            self._layer(**{name: True})

    def test_a_module_must_return_a_tensor(self):
        class Pair(nn.Module):
            @override
            def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return h, h

        with pytest.raises(TypeError, match="tensor"):
            self._layer(module=Pair())


class TestJacobi:
    @pytest.mark.parametrize("learn_interval", [None, 4 * H])
    def test_stepping_from_the_output_is_the_jacobi_update(
        self, start_manual: StartManual, step: Step, learn_interval: float | None
    ):
        # Linear + tanh, then a residual block: the reference moves all nodes at
        # once; stepping from the output to the input reads only values of the
        # previous step everywhere, so the two agree to rounding. Learning in
        # every step, the layers move h by the jumps of their own theta; at
        # intervals, these jumps also go up the chain with the epoch.
        widths = [4, 5, 5]
        layers = [
            make_modules("linear", "tanh", widths[:2])[0][0],
            make_modules("res", "tanh", widths[1:], seed=1)[0][0],
        ]
        head = make_modules("linear", "tanh", widths, seed=2)[1]
        reference = Reference(layers, head, widths, learn_interval)
        chain = Chain(layers, head, widths, learn_interval=learn_interval)
        run = start_manual(chain.net)
        samples = [sample(widths, seed) for seed in (0, 1)]
        x, y = samples[0]
        time = chain.write(run, x, y, MASK)
        for k in range(50):
            if k == 20:  # a new input, and a new target with it
                x, y = samples[1]
                time = chain.write(run, x, y, MASK)
            step(run, *chain.from_next())
            reference.step(x, y, MASK, time, H, run.timer.now().ns)
            tree = run.state_dict()
            for i, layer in enumerate(reference.layers):
                state = tree["nodes"][f"l{i}"]
                close(state["h"], reference.h[i + 1], 1e-12)
                close(state["lam"], reference.lam[i], 1e-12)
                for key, value in layer.state_dict().items():
                    close(state["module"][key], value, 1e-12)
                activity, _, _, carried = tree["slots"][f"l{i}.next/out"]
                close(activity["h"], reference.h[i + 1], 1e-12)
                assert int(activity["epoch"]) == reference.epoch[i + 1]
                assert torch.equal(carried["y"], reference.carried[i + 1][0])
                assert torch.equal(carried["mask"], reference.carried[i + 1][1])
                close(tree["slots"][f"l{i}.prev/out"][0], reference.g_in[i], 1e-12)
            assert int(tree["slots"]["inp.next/out"][0]["epoch"]) == reference.epoch[0]
            for key, value in reference.head.state_dict().items():
                close(tree["nodes"]["out"]["module"][key], value, 1e-12)
            close(tree["slots"]["out.prev/out"][0], reference.g_in[-1], 1e-12)
            prediction = run.read(chain.out.next)
            close(prediction["y_hat"], reference.y_hat, 1e-12)
            assert torch.equal(prediction["y"], reference.prediction_carried[0])
            assert torch.equal(prediction["mask"], reference.prediction_carried[1])
        if learn_interval is not None:  # the chain did jump
            assert reference.epoch[-1] > reference.epoch[0] + 1


# Steps to the fixed point from h = lam = 0, to 1e-10 at TAU and these sizes.
RELAX_STEPS = 3000
# The same for the chain of eight residual blocks, to 1e-9.
DEEP_STEPS = 6000
CASES = [(block, act) for block in ("linear", "res") for act in ("linear", "tanh")]


@pytest.mark.slow
class TestFixedPoint:
    @pytest.mark.parametrize(("block", "act"), CASES)
    def test_lam_is_the_credit_of_backpropagation(
        self, start_manual: StartManual, step: Step, block: str, act: str
    ):
        widths = [4, 4, 4] if block == "res" else [4, 5, 3]
        layers, head = make_modules(block, act, widths)
        chain = Chain(layers, head, widths, tau=NO_LEARNING)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        run_steps(run, step, chain.from_next(), RELAX_STEPS)
        nodes = run.state_dict()["nodes"]
        expected = credits(layers, head, x, y, MASK)
        with torch.no_grad():
            forward_hs, _ = forward(layers, head, x)
        for i, (credit, h) in enumerate(zip(expected, forward_hs, strict=True)):
            close(nodes[f"l{i}"]["h"], h, 1e-10)
            close(nodes[f"l{i}"]["lam"], credit, 1e-8)
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
            close(nodes["l1"]["lam"], lam_last.detach(), 1e-8)
            close(nodes["l0"]["lam"], lam_first.detach(), 1e-8)

    @pytest.mark.parametrize(("block", "act"), CASES)
    def test_one_update_at_the_fixed_point_is_sgd(
        self, start_manual: StartManual, step: Step, block: str, act: str
    ):
        # The first update comes learn_interval after the input was written (at
        # time 0), at the fixed point, weighted by the time since the start.
        widths = [4, 4, 4] if block == "res" else [4, 5, 3]
        layers, head = make_modules(block, act, widths)
        sgd = [copy.deepcopy(m) for m in [*layers, head]]
        chain = Chain(layers, head, widths, learn_interval=RELAX_STEPS * H)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        run_steps(run, step, chain.from_next(), RELAX_STEPS)
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
                close(nodes[name]["module"][key], value, 1e-10)


@pytest.mark.slow
class TestDeepChain:
    def test_eight_residual_nodes_reach_the_fixed_point(
        self, start_manual: StartManual, step: Step
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
        chain = Chain(layers, head, widths, tau=NO_LEARNING)
        run = start_manual(chain.net)
        chain.write(run, x, y, MASK)
        run_steps(run, step, chain.from_next(), DEEP_STEPS)
        nodes = run.state_dict()["nodes"]
        expected = credits(layers, head, x, y, MASK)
        for i, (credit, h) in enumerate(zip(expected, forward_hs, strict=True)):
            close(nodes[f"l{i}"]["h"], h, 1e-9)
            close(nodes[f"l{i}"]["lam"], credit, 1e-8)

    @pytest.mark.parametrize(
        "learn_interval", [None, 16 * H], ids=["every-update", "16-periods"]
    )
    def test_learning_keeps_lam_at_the_credit(
        self, start_manual: StartManual, step: Step, learn_interval: float | None
    ):
        # A theta so slow it barely moves: only the jumps theta makes move h,
        # not the relaxation of the nodes below, so lam goes to the credit and
        # stays there. (Moving h with every change of mu, the previous design,
        # made lam grow to 1e5 in 500 steps here with learn_interval=None,
        # however slow theta was, before it settled elsewhere.) 16 periods is
        # about the round trip 2 * depth * period of this chain (18 periods); an
        # interval much shorter advances the epoch in almost every update, the
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
        chain = Chain(layers, head, widths, tau=slow, learn_interval=learn_interval)
        run = start_manual(chain.net)
        chain.write(run, x, y, MASK)
        for _ in range(DEEP_STEPS // 100):
            run_steps(run, step, chain.from_next(), 100)
            nodes = run.state_dict()["nodes"]
            assert max(nodes[f"l{i}"]["lam"].abs().max() for i in range(8)) < bound
        nodes = run.state_dict()["nodes"]
        moved = [layer.module for layer in chain.layers]  # the nodes' own modules
        expected = credits(moved, chain.out.module, x, y, MASK)
        for i, credit in enumerate(expected):
            close(nodes[f"l{i}"]["lam"], credit, 1e-8)


@pytest.mark.slow
class TestNodeBoundaries:
    def test_two_constraints_in_one_node_learn_as_in_two(
        self, start_manual: StartManual, step: Step
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
            chain = Chain(
                modules,
                copy.deepcopy(head),
                chain_widths,
                learn_interval=RELAX_STEPS * H,
            )
            run = start_manual(chain.net, timer=VirtualTimer())
            chain.write(run, x, y, MASK)
            run_steps(run, step, chain.from_next(), RELAX_STEPS)
            nodes = run.state_dict()["nodes"]
            names = [f"l{i}" for i in range(len(modules))]
            # The parameters by their names in nn.Sequential(*layers).
            params = {
                f"{i}.{key}" if len(names) > 1 else key: value
                for i, name in enumerate(names)
                for key, value in nodes[name]["module"].items()
            }
            updated.append(
                params | {f"head.{k}": v for k, v in nodes["out"]["module"].items()}
            )
        assert updated[0].keys() == updated[1].keys()
        for key, value in updated[0].items():
            close(value, updated[1][key], 1e-10)


@pytest.mark.slow
class TestPcLimit:
    def test_without_the_dual_the_fixed_point_is_that_of_pc(
        self, start_manual: StartManual, step: Step
    ):
        # alpha = 0: lam stays exactly zero, and the activities settle where the
        # energy of predictive coding is stationary, away from the forward pass.
        widths = [4, 5, 3]
        layers, head = make_modules("linear", "tanh", widths)
        tau = pc.TimeConstants(h=TAU.h, lam=math.inf, theta=math.inf)
        chain = Chain(layers, head, widths, tau=tau)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        for _ in range(RELAX_STEPS):
            step(run, *chain.from_next())
            for i in range(len(layers)):
                lam = run.state_dict()["nodes"][f"l{i}"]["lam"]
                assert torch.equal(lam, torch.zeros_like(lam))
        tree = run.state_dict()
        hs = [tree["slots"][f"l{i}.next/out"][0]["h"] for i in range(len(layers))]
        g = [tree["slots"]["l1.prev/out"][0], tree["slots"]["out.prev/out"][0]]
        for i in range(len(layers)):
            state = tree["nodes"][f"l{i}"]
            r = state["h"] - state["_mu"]
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
    def test_lam_is_w2_times_the_error(self, start_manual: StartManual, step: Step):
        # y_hat = w2 h, h = w1 x: at the fixed point lam = w2 (y - y_hat).
        w1, w2, x, y = 0.5, 2.0, 1.5, 1.0
        first, head = nn.Linear(1, 1, bias=False), nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            first.weight.fill_(w1)
            head.weight.fill_(w2)
        carry = pc.carry_spec(TensorSpec((1, 1), F64), batch=1)
        inp = pc.Input((1, 1), carry=carry, period=H, dtype=F64)
        layer = pc.Layer(
            first.double(), input_shape=(1, 1), carry=carry, tau=NO_LEARNING, dtype=F64
        )
        out = pc.Output(
            head.double(),
            loss=pc.mse,
            input_shape=(1, 1),
            carry=carry,
            tau=NO_LEARNING,
            dtype=F64,
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
        run_steps(run, step, [out, layer, inp], 1500)
        lam = run.state_dict()["nodes"]["layer"]["lam"]
        close(lam, torch.full((1, 1), w2 * (y - w2 * w1 * x), dtype=F64), 1e-10)


@pytest.mark.slow
class TestSchedules:
    def test_every_order_reaches_the_same_fixed_point(
        self, start_manual: StartManual, step: Step
    ):
        widths = [4, 4, 4]
        x, y = sample(widths, 0)
        expected = credits(*make_modules("res", "tanh", widths), x, y, MASK)
        shuffle = random.Random(0)
        orders: list[Callable[[Chain], list[Any]]] = [
            Chain.from_next,
            Chain.from_prev,
            lambda chain: shuffle.sample(chain.from_prev(), k=len(chain.from_prev())),
        ]
        for order in orders:
            layers, head = make_modules("res", "tanh", widths)
            chain = Chain(layers, head, widths, tau=NO_LEARNING)
            run = start_manual(chain.net, timer=VirtualTimer())
            chain.write(run, x, y, MASK)
            for _ in range(RELAX_STEPS):
                step(run, *order(chain))
            nodes = run.state_dict()["nodes"]
            for i, credit in enumerate(expected):
                close(nodes[f"l{i}"]["lam"], credit, 1e-8)

    def test_every_order_learns_as_sgd(self, start_manual: StartManual, step: Step):
        # Learning at intervals, twice, at the fixed point each time: whatever
        # the order, each update is one step of SGD, the second from the
        # parameters of the first. The jumps that the first update makes go up
        # the chain with the epochs, and the chain relaxes to the next fixed
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
        orders: list[Callable[[Chain], list[Any]]] = [
            Chain.from_next,
            Chain.from_prev,
            lambda chain: shuffle.sample(chain.from_prev(), k=len(chain.from_prev())),
        ]
        for order in orders:
            layers, head = make_modules("linear", "tanh", widths)
            chain = Chain(layers, head, widths, tau=tau, learn_interval=RELAX_STEPS * H)
            run = start_manual(chain.net, timer=VirtualTimer())
            chain.write(run, x, y, MASK)
            for _ in range(2 * RELAX_STEPS):
                step(run, *order(chain))
            nodes = run.state_dict()["nodes"]
            for name, module in zip(["l0", "l1", "out"], sgd, strict=True):
                for key, value in module.state_dict().items():
                    close(nodes[name]["module"][key], value, 1e-10)


class TestWithoutTeacher:
    def test_h_is_the_forward_pass(self, start_manual: StartManual, step: Step):
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
        chain = Chain(layers, head, widths)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, torch.zeros(B, dtype=torch.bool))
        with torch.no_grad():
            hs, _ = forward(layers, head, x)
        for _ in range(10):
            step(run, *chain.from_prev())
            tree = run.state_dict()
            for i, h in enumerate(hs):
                assert torch.equal(tree["slots"][f"l{i}.next/out"][0]["h"], h)
                assert torch.equal(tree["nodes"][f"l{i}"]["h"], h)
                lam = tree["nodes"][f"l{i}"]["lam"]
                assert torch.equal(lam, torch.zeros_like(lam))
        tree = run.state_dict()
        names = [*(f"l{i}" for i in range(len(layers))), "out"]
        for name, state in zip(names, before, strict=True):
            for key, value in state.items():
                assert torch.equal(tree["nodes"][name]["module"][key], value)

    def test_a_new_input_takes_one_step_per_sender(
        self, start_manual: StartManual, step: Step
    ):
        # At the forward pass of a first input (h - mu = 0), a second one is
        # stepped from the output: each node reads the previous step's values,
        # so the new forward pass reaches l1 in three steps (Input, l0, l1).
        widths = [4, 5, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = Chain(layers, head, widths)
        run = start_manual(chain.net)
        no_target = torch.zeros(B, dtype=torch.bool)
        x, y = sample(widths, 0)
        chain.write(run, x, y, no_target)
        step(run, *chain.from_prev())
        x, y = sample(widths, 1)
        chain.write(run, x, y, no_target)
        with torch.no_grad():
            hs, _ = forward(layers, head, x)
        for _ in range(2):
            step(run, *chain.from_next())
        assert not torch.equal(run.state_dict()["slots"]["l1.next/out"][0]["h"], hs[-1])
        step(run, *chain.from_next())
        assert torch.equal(run.state_dict()["slots"]["l1.next/out"][0]["h"], hs[-1])


# Input -> Layer -> Output that learns K steps after each new input, and a host
# that writes a new input every M steps.
K, M = 3, 5


class Learning:
    """``Chain`` of ``widths``, stepped from the output with a host that writes a
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
        self.chain = Chain(
            layers, head, self.widths, learn_interval=K * H, max_dt=max_dt
        )
        self.run: ManualRun[Any] = start_manual(
            self.chain.net, timer=VirtualTimer(), state=state
        )
        self.steps = steps

    def step(self, masks: Callable[[int], torch.Tensor] = lambda _: MASK) -> None:
        self.run.timer.advance(H)
        if self.steps % M == 0:
            x, y = sample(self.widths, self.steps)
            self.chain.write(self.run, x, y, masks(self.steps // M))
        self.steps += 1
        self.run.step(*self.chain.from_next())


def h_before_mu(state: dict[str, Any]) -> torch.Tensor:
    """The ``h`` of a layer in a step, from its state before the step, once it
    took the jump of its ``theta`` of the step before (if any)."""
    return state["h"] + state["_jump"] if state["_jumped"] else state["h"]


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
    module.load_state_dict(tree["nodes"]["l0"]["module"])
    params = list(module.parameters())
    state = tree["nodes"]["l0"]
    activity, _, _, carried = tree["slots"]["inp.next/out"]
    labeled = carried["mask"][:, None]
    h = h_before_mu(state)

    def delta(mu: torch.Tensor) -> torch.Tensor:
        r = h - (state["_mu"] if reanchor else mu)
        return torch.where(labeled, r + state["lam"], 0.0)

    _, _, g_theta = pc.functional.vjp(module, activity["h"], delta, params)
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
            after = learning.run.state_dict()["nodes"]["l0"]["module"]
            if equal_states(after, before["nodes"]["l0"]["module"]):
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
        self, start_manual: StartManual, step: Step
    ):
        # Its time 0 is also that of the zeros carried before it, so the nodes
        # do not see it as new, and learn 2 K steps after time 0 as for the
        # zeros; before the input arrives, the mask (all False) keeps them from
        # learning. The layer and the output learn in the same step. (The
        # input's epoch still moves h with it: at step K, the credit would not
        # have reached the layer yet, delta = 0.)
        widths = [4, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = Chain(layers, head, widths, learn_interval=2 * K * H)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        assert chain.write(run, x, y, MASK) == 0
        names = ["l0", "out"]
        changed: dict[str, list[int]] = {name: [] for name in names}
        for s in range(1, 4 * K + 3):
            before = run.state_dict()
            step(run, *chain.from_next())
            if s == 2 * K:
                expected = layer_update(before, layers[0], s * H)
                after = run.state_dict()["nodes"]["l0"]["module"]
                for key, value in expected.items():
                    close(after[key], value, 1e-14)
                assert not equal_states(after, before["nodes"]["l0"]["module"])
            for name in names:
                after = run.state_dict()["nodes"][name]["module"]
                if not equal_states(after, before["nodes"][name]["module"]):
                    changed[name].append(s)
        assert changed == {"l0": [2 * K, 4 * K], "out": [2 * K, 4 * K]}

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
            run.step(*learning.chain.from_next())
            return before, run.state_dict()["nodes"]["l0"]["module"]

        for s in range(1, 7 + K):
            before, after = step(s)
            assert equal_states(after, before["nodes"]["l0"]["module"]), s
        before, after = step(7 + K)
        expected = layer_update(before, module, K * H)
        for key, value in expected.items():
            close(after[key], value, 1e-14)


def learned_steps(
    chain: Chain,
    widths: Sequence[int],
    run: ManualRun[Any],
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
        run.step(*order())
        nodes = run.state_dict()["nodes"]
        for name in names:
            # Before the first input, the zeros carried (time 0) skip theirs.
            if nodes[name]["_seen_time"] != 0 and nodes[name]["_elapsed"] == 0.0:
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
            chain = Chain(layers, head, widths, learn_interval=8 * H)
            run = start_manual(chain.net, timer=VirtualTimer())

            def order(chain: Chain = chain, name: str = name) -> list[Any]:
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
        chain = Chain(layers, head, widths, learn_interval=2 * H)
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
    state = before["nodes"][node]
    module = copy.deepcopy(module)
    module.load_state_dict(state["module"])
    with torch.no_grad():
        mu = module(before["slots"][f"{h_in}.next/out"][0]["h"])
    h, lam = h_before_mu(state), state["lam"]
    if reanchor:
        h = mu + (h - state["_mu"])
    g_sent = after["slots"][f"{g}.prev/out"][0]
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
        assert before["nodes"]["l0"]["_jumped"]
        learning.run.timer.advance(4 * H)
        learning.step()
        after = learning.run.state_dict()
        h, lam = relaxed(before, after, module, counted, reanchor=False)
        close(after["nodes"]["l0"]["h"], h, 1e-14)
        close(after["nodes"]["l0"]["lam"], lam, 1e-14)
        expected = layer_update(before, module, counted)
        for key, value in expected.items():
            close(after["nodes"]["l0"]["module"][key], value, 1e-14)

    def test_the_default_keeps_the_step_of_lam_stable(self):
        # dt / tau.lam <= 3 as well as dt / tau.h <= 1/2.
        carry = pc.carry_spec(TensorSpec((B, N_OUT)), batch=B)
        fast_dual = pc.TimeConstants(h=4e-3, lam=0.5e-3, theta=1.0)
        layer = pc.Layer(
            nn.Linear(4, 5), input_shape=(B, 4), carry=carry, tau=fast_dual
        )
        assert layer._max_dt == pytest.approx(1.5e-3)  # pyright: ignore[reportPrivateUsage]
        no_dual = pc.TimeConstants(h=4e-3, lam=math.inf, theta=1.0)
        layer = pc.Layer(nn.Linear(4, 5), input_shape=(B, 4), carry=carry, tau=no_dual)
        assert layer._max_dt == pytest.approx(2e-3)  # pyright: ignore[reportPrivateUsage]


class TestJumps:
    def test_h_moves_with_new_inputs_and_with_the_jumps_of_theta(
        self, start_manual: StartManual
    ):
        # The layer sees the inputs at steps 2, 7, ... and learns at 4, 9, ...:
        # h moves with mu at steps 2 and 7 (new inputs); at step 5 it takes the
        # jump f(h_in; theta_new) - f(h_in; theta) that learning made at step 4,
        # at the h_in of step 4, and only relaxes otherwise. Each of these
        # advances the epoch it sends (and the first update, from zeros).
        learning = Learning(start_manual)
        module = learning.chain.layers[0].module
        epochs: list[int] = []
        for s in range(1, 9):
            before = learning.run.state_dict()
            learning.step()
            after = learning.run.state_dict()
            epochs.append(int(after["slots"]["l0.next/out"][0]["epoch"]))
            if s == 1:
                continue  # the first update moves h from zeros
            h, lam = relaxed(before, after, module, H, reanchor=s in (2, 7))
            close(after["nodes"]["l0"]["h"], h, 1e-14)
            close(after["nodes"]["l0"]["lam"], lam, 1e-14)
            assert after["nodes"]["l0"]["_jumped"] == (s == 4)
            if s == 4:
                h_in = before["slots"]["inp.next/out"][0]["h"]
                old, new = copy.deepcopy(module), copy.deepcopy(module)
                old.load_state_dict(before["nodes"]["l0"]["module"])
                new.load_state_dict(after["nodes"]["l0"]["module"])
                with torch.no_grad():
                    close(after["nodes"]["l0"]["_mu"], new(h_in), 1e-14)
                    close(after["nodes"]["l0"]["_jump"], new(h_in) - old(h_in), 1e-14)
        assert epochs == [1, 2, 2, 2, 3, 3, 4, 4]

    def test_a_jump_below_moves_the_next_h_with_r_unchanged(
        self, start_manual: StartManual
    ):
        # l1 reads the first input at step 3. Both layers learn at step 4 and
        # take their own jumps at step 5; l0 sends its jump with a new epoch,
        # which l1 reads at step 6 and moves with: h - mu_last before the move
        # is h - mu after it. At step 8, l1 reads a new input and the epoch l0
        # advanced for it at step 7: one move, one new epoch.
        learning = Learning(start_manual, widths=(4, 5, 5))
        module = learning.chain.layers[1].module
        epochs: list[int] = []
        for s in range(1, 10):
            before = learning.run.state_dict()
            learning.step()
            after = learning.run.state_dict()
            epochs.append(int(after["slots"]["l1.next/out"][0]["epoch"]))
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
            close(after["nodes"]["l1"]["h"], h, 1e-14)
            close(after["nodes"]["l1"]["lam"], lam, 1e-14)
        assert epochs == [1, 2, 3, 3, 4, 5, 5, 6, 6]

    def test_learning_in_every_update_moves_no_node_above(
        self, start_manual: StartManual, step: Step
    ):
        # With learn_interval=None, theta moves by dt in every update: each layer
        # takes its own jump, but advances its epoch only for the new input.
        widths = [4, 5, 5]
        layers, head = make_modules("linear", "tanh", widths)
        chain = Chain(layers, head, widths)
        run = start_manual(chain.net)
        x, y = sample(widths, 0)
        chain.write(run, x, y, MASK)
        run_steps(run, step, chain.from_next(), 5)
        for _ in range(10):
            step(run, *chain.from_next())
            tree = run.state_dict()
            for i in range(2):
                assert tree["nodes"][f"l{i}"]["_jumped"]
                assert int(tree["slots"][f"l{i}.next/out"][0]["epoch"]) == 2 + i


class TestUnlabeled:
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
        after = learning.run.state_dict()["nodes"]["l0"]["module"]
        assert equal_states(after, before["nodes"]["l0"]["module"])
        for _ in range(4):
            learning.step(masks)
        before = learning.run.state_dict()
        learning.step(masks)  # step 9
        after = learning.run.state_dict()["nodes"]["l0"]["module"]
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
        state = before["nodes"]["l0"]
        assert (state["h"] - state["_mu"] + state["lam"])[1].abs().max() > 1e-6
        g_in = after["slots"]["l0.prev/out"][0]
        assert torch.equal(g_in[1], torch.zeros_like(g_in[1]))
        expected = layer_update(before, module, 5 * H)
        for key, value in expected.items():
            close(after["nodes"]["l0"]["module"][key], value, 1e-14)


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
            "_seen_time",
            "_learn_at",
            "_elapsed",
            "_learned_input",
            "_jump",
            "_jumped",
            "_epoch",
            "_seen_epoch",
            "_mu",
            "module",
            "h",
            "lam",
        } <= set(tree["nodes"]["l0"])
        assert {"_epoch", "_seen_write"} <= set(tree["nodes"]["inp"])
        first.run.shutdown()
        restored = Learning(start_manual, seed=1, state=tree, steps=6)
        for _ in range(10):
            restored.step()
        assert_same_tree(restored.run.state_dict(), whole.run.state_dict())


def assert_same_tree(a: object, b: object) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            assert_same_tree(a[key], b[key])
    elif isinstance(a, tuple | list):
        assert isinstance(b, type(a)) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_same_tree(x, y)
    else:
        assert a == b
