"""The weight optimizers of ``mujou.pc``, which move the parameters by the
time that passed: ``pc.optim.SGD`` (with and without momentum),
``pc.optim.RMSProp`` and ``pc.optim.AdamW``. The expected values are the
formulas written out here; ``torch.optim`` runs only in a subprocess, since
building one re-enables the GIL of the free-threaded build."""

import io
import math
import os
import subprocess
import sys
import sysconfig
from typing import Any

import pytest
import torch

import mujou.pc as pc
from tests.mujou.nodes import F64, assert_same_tree

# AdamW whose eps is of the order of sqrt(v_hat) (gradients of 1e-3), so that
# a wrong place for eps shows.
TAU, TAU_M, TAU_V, EPS, W = 0.3, 0.02, 0.1, 1e-3, 0.1
ADAMW = pc.optim.AdamW(tau=TAU, tau_m=TAU_M, tau_v=TAU_V, eps=EPS, weight_decay=W)
SGD_M = pc.optim.SGD(tau=TAU, tau_m=TAU_M)
RMSPROP = pc.optim.RMSProp(tau=TAU, tau_v=TAU_V, eps=EPS)
DT = 4e-3
SCALE = 1e-3  # of the gradients


def randn(*shape: int, seed: int, dtype: torch.dtype = F64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=generator)


def params(dtype: torch.dtype = F64) -> list[torch.Tensor]:
    """A weight and a bias: two shapes, so that a mixed-up order fails."""
    return [randn(4, 3, seed=0, dtype=dtype), randn(3, seed=1, dtype=dtype)]


def grads(k: int, dtype: torch.dtype = F64) -> list[torch.Tensor]:
    """The mean gradients of step ``k`` for ``params()``."""
    return [
        SCALE * randn(4, 3, seed=100 + 2 * k, dtype=dtype),
        SCALE * randn(3, seed=101 + 2 * k, dtype=dtype),
    ]


def clones(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t.clone() for t in tensors]


def adamw_with_steps(
    p0: list[torch.Tensor], steps: list[list[torch.Tensor]], dt: float
) -> list[torch.Tensor]:
    """The discrete AdamW of torch (decoupled weight decay), written out, with
    ``lr = dt / tau``, ``beta = exp(-dt / tau_x)``, the step count ``t`` and
    the bias corrections ``1 - beta^t``; eps is added to ``sqrt(v_hat)``."""
    lr = dt / TAU
    beta1, beta2 = math.exp(-dt / TAU_M), math.exp(-dt / TAU_V)
    ps = clones(p0)
    m = [torch.zeros_like(p) for p in p0]
    v = [torch.zeros_like(p) for p in p0]
    for t, gs in enumerate(steps, start=1):
        correction1, correction2 = 1 - beta1**t, 1 - beta2**t
        for i, g in enumerate(gs):
            ps[i] = ps[i] * (1 - lr * W)
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat, v_hat = m[i] / correction1, v[i] / correction2
            ps[i] = ps[i] - lr * m_hat / (v_hat.sqrt() + EPS)
    return ps


def momentum_sgd_with_steps(
    p0: list[torch.Tensor], steps: list[list[torch.Tensor]], dt: float
) -> list[torch.Tensor]:
    """SGD with a momentum, written out with ``lr = dt / tau``, ``beta =
    exp(-dt / tau_m)``, the dampening ``beta``, the step count ``t`` and the
    bias correction ``1 - beta^t`` of the momentum."""
    lr = dt / TAU
    beta = math.exp(-dt / TAU_M)
    ps = clones(p0)
    m = [torch.zeros_like(p) for p in p0]
    for t, gs in enumerate(steps, start=1):
        for i, g in enumerate(gs):
            m[i] = beta * m[i] + (1 - beta) * g
            ps[i] = ps[i] - lr * m[i] / (1 - beta**t)
    return ps


def rmsprop_with_steps(
    p0: list[torch.Tensor], steps: list[list[torch.Tensor]], dt: float
) -> list[torch.Tensor]:
    """RMSProp written out with ``lr = dt / tau``, ``alpha = exp(-dt /
    tau_v)``, the step count ``t`` and the bias correction ``1 - alpha^t`` of
    the mean square; eps is added to ``sqrt(v_hat)``."""
    lr = dt / TAU
    alpha = math.exp(-dt / TAU_V)
    ps = clones(p0)
    v = [torch.zeros_like(p) for p in p0]
    for t, gs in enumerate(steps, start=1):
        for i, g in enumerate(gs):
            v[i] = alpha * v[i] + (1 - alpha) * g * g
            v_hat = v[i] / (1 - alpha**t)
            ps[i] = ps[i] - lr * g / (v_hat.sqrt() + EPS)
    return ps


def assert_close(actual: torch.Tensor, expected: torch.Tensor, rtol: float) -> None:
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)


class TestSGD:
    def test_one_step_moves_by_elapsed_over_tau_times_the_gradient(self):
        # tau dtheta/dt = -g: one step of elapsed is SGD at the rate elapsed / tau.
        ps = params()
        p0, gs = clones(ps), grads(0)
        pc.optim.SGD(tau=0.5).bind(ps).step_(gs, 3e-3)
        for p, p_0, g in zip(ps, p0, gs, strict=True):
            torch.testing.assert_close(
                p, p_0 - (3e-3 / 0.5) * g, rtol=1e-15, atol=1e-15
            )

    def test_the_mean_of_a_batch_of_a_power_of_two_is_the_old_update_to_the_bit(
        self,
    ):
        # The nodes used to take g_sum with the rate (elapsed / tau) / B; they
        # now hand g_sum / B to the optimizer. With B = 4 both are exact
        # scalings of the same product.
        tau, elapsed, batch = 0.05, 3e-3, 4
        ps = params()
        p0 = clones(ps)
        g_sum = [batch * g for g in grads(0)]
        pc.optim.SGD(tau=tau).bind(ps).step_([g / batch for g in g_sum], elapsed)
        for p, p_0, g in zip(ps, p0, g_sum, strict=True):
            assert torch.equal(p, p_0.sub(g, alpha=elapsed / tau / batch))

    def test_zero_elapsed_moves_nothing_and_there_is_no_state(self):
        ps = params()
        p0 = clones(ps)
        bound = pc.optim.SGD(tau=0.5).bind(ps)
        bound.step_(grads(0), 0.0)
        for p, p_0 in zip(ps, p0, strict=True):
            assert torch.equal(p, p_0)
        assert bound.state_dict() == {}

    def test_momentum_steps_of_a_constant_dt_are_momentum_sgd_with_bias_correction(
        self,
    ):
        ps = params()
        steps = [grads(k) for k in range(10)]
        expected = momentum_sgd_with_steps(params(), steps, DT)
        bound = SGD_M.bind(ps)
        for gs in steps:
            bound.step_(gs, DT)
        for p, e in zip(ps, expected, strict=True):
            assert_close(p, e, rtol=1e-12)

    def test_momentum_with_a_constant_gradient_steps_as_sgd_without_it(self):
        # With g constant, m = c_m g and m_hat = g after every step, whatever
        # the steps: the momentum leaves the learning rate as it is.
        g = grads(0)
        ps, qs = params(), params()
        with_momentum, without = SGD_M.bind(ps), pc.optim.SGD(tau=TAU).bind(qs)
        for dt in [1e-3, 5e-3, 2e-4, 3e-2, 7e-3]:
            with_momentum.step_(g, dt)
            without.step_(g, dt)
        for p, q in zip(ps, qs, strict=True):
            assert_close(p, q, rtol=1e-12)

    def test_a_momentum_much_shorter_than_the_steps_is_sgd_without_it(self):
        # tau_m -> 0: a_m = 1, so m = g and c_m = 1 after every step.
        ps, qs = params(), params()
        short = pc.optim.SGD(tau=TAU, tau_m=1e-9).bind(ps)
        without = pc.optim.SGD(tau=TAU).bind(qs)
        for k in range(5):
            short.step_(grads(k), DT)
            without.step_(grads(k), DT)
        for p, q in zip(ps, qs, strict=True):
            assert_close(p, q, rtol=1e-15)

    def test_two_steps_of_dt_are_one_step_of_two_dt_for_the_momentum(self):
        # As for AdamW: m and c_m follow the flow exactly in each step.
        g = grads(1)
        split, whole = SGD_M.bind(params()), SGD_M.bind(params())
        for bound in (split, whole):
            bound.step_(grads(0), DT)  # a warm-up: m away from zero
        split.step_(g, DT)
        split.step_(g, DT)
        whole.step_(g, 2 * DT)
        a, b = split.state_dict(), whole.state_dict()
        for x, y in zip(a["m"], b["m"], strict=True):
            assert_close(x, y, rtol=1e-12)
        assert a["c_m"] == pytest.approx(b["c_m"], rel=1e-12, abs=0.0)

    def test_zero_elapsed_first_moves_nothing_with_momentum(self):
        ps = params()
        p0 = clones(ps)
        bound = SGD_M.bind(ps)
        bound.step_(grads(0), 0.0)
        for p, p_0 in zip(ps, p0, strict=True):
            assert torch.equal(p, p_0)
        assert bound.state_dict()["c_m"] == 0.0


class TestRMSProp:
    def test_steps_of_a_constant_dt_are_the_discrete_rmsprop_with_bias_correction(
        self,
    ):
        ps = params()
        steps = [grads(k) for k in range(10)]
        expected = rmsprop_with_steps(params(), steps, DT)
        bound = RMSPROP.bind(ps)
        for gs in steps:
            bound.step_(gs, DT)
        for p, e in zip(ps, expected, strict=True):
            assert_close(p, e, rtol=1e-12)

    @pytest.mark.parametrize("tau_v", [TAU_V, 30.0])
    def test_the_first_step_is_the_sign_of_the_gradient(self, tau_v: float):
        # v_hat = g^2 after the first step, whatever tau_v:
        # theta - eta g / (|g| + eps), eta = elapsed / tau.
        rmsprop = pc.optim.RMSProp(tau=TAU, tau_v=tau_v, eps=EPS)
        ps = params()
        p0, gs = clones(ps), grads(0)
        rmsprop.bind(ps).step_(gs, DT)
        eta = DT / TAU
        for p, p_0, g in zip(ps, p0, gs, strict=True):
            assert_close(p, p_0 - eta * g / (g.abs() + EPS), 1e-12)

    def test_two_steps_of_dt_are_one_step_of_two_dt_for_the_mean_square(self):
        # As for AdamW: v and c_v follow the flow exactly in each step.
        g = grads(1)
        split, whole = RMSPROP.bind(params()), RMSPROP.bind(params())
        for bound in (split, whole):
            bound.step_(grads(0), DT)  # a warm-up: v away from zero
        split.step_(g, DT)
        split.step_(g, DT)
        whole.step_(g, 2 * DT)
        a, b = split.state_dict(), whole.state_dict()
        for x, y in zip(a["v"], b["v"], strict=True):
            assert_close(x, y, rtol=1e-12)
        assert a["c_v"] == pytest.approx(b["c_v"], rel=1e-12, abs=0.0)

    def test_zero_elapsed_first_moves_nothing_and_keeps_the_mass_at_zero(self):
        ps = params()
        p0 = clones(ps)
        bound = RMSPROP.bind(ps)
        bound.step_(grads(0), 0.0)
        for p, p_0 in zip(ps, p0, strict=True):
            assert torch.equal(p, p_0)
        assert bound.state_dict()["c_v"] == 0.0


# The optimizers with a state: the keys of their moments and of their masses.
STATEFUL = [
    (SGD_M, ["m"], ["c_m"]),
    (RMSPROP, ["v"], ["c_v"]),
    (ADAMW, ["m", "v"], ["c_m", "c_v"]),
]
STATEFUL_IDS = ["sgd-momentum", "rmsprop", "adamw"]


class TestAdamW:
    def test_steps_of_a_constant_dt_are_the_discrete_adamw(self):
        ps = params()
        steps = [grads(k) for k in range(10)]
        expected = adamw_with_steps(params(), steps, DT)
        bound = ADAMW.bind(ps)
        for gs in steps:
            bound.step_(gs, DT)
        for p, e in zip(ps, expected, strict=True):
            assert_close(p, e, rtol=1e-12)

    @pytest.mark.parametrize(("tau_m", "tau_v"), [(TAU_M, TAU_V), (1.0, 30.0)])
    def test_the_first_step_is_the_sign_of_the_gradient(
        self, tau_m: float, tau_v: float
    ):
        # m_hat = g and v_hat = g^2 after the first step, whatever tau_m and
        # tau_v: theta (1 - eta w) - eta g / (|g| + eps), eta = elapsed / tau.
        adamw = pc.optim.AdamW(
            tau=TAU, tau_m=tau_m, tau_v=tau_v, eps=EPS, weight_decay=W
        )
        ps = params()
        p0, gs = clones(ps), grads(0)
        adamw.bind(ps).step_(gs, DT)
        eta = DT / TAU
        for p, p_0, g in zip(ps, p0, gs, strict=True):
            assert_close(p, p_0 * (1 - eta * W) - eta * g / (g.abs() + EPS), 1e-12)

    def test_two_steps_of_dt_are_one_step_of_two_dt_for_the_moments(self):
        # With g constant, m, v and their masses follow tau dm/dt = g - m
        # exactly in each step, so splitting the time changes nothing. (theta
        # takes explicit Euler steps, so it is not compared.)
        g = grads(1)
        split, whole = ADAMW.bind(params()), ADAMW.bind(params())
        for bound in (split, whole):
            bound.step_(grads(0), DT)  # a warm-up: m and v away from zero
        split.step_(g, DT)
        split.step_(g, DT)
        whole.step_(g, 2 * DT)
        a, b = split.state_dict(), whole.state_dict()
        for key in ("m", "v"):
            for x, y in zip(a[key], b[key], strict=True):
                assert_close(x, y, rtol=1e-12)
        for key in ("c_m", "c_v"):
            assert a[key] == pytest.approx(b[key], rel=1e-12, abs=0.0)

    def test_the_mass_is_one_minus_the_decay_of_the_total_time(self):
        # tau_m dc/dt = 1 - c from c = 0: c = 1 - exp(-t / tau_m), whatever the
        # steps the time t was taken in; a long step makes it one. The state is
        # the moments and their masses, without a step count.
        bound = ADAMW.bind(params())
        total = 0.0
        for k, dt in enumerate([1e-3, 5e-3, 2e-4, 3e-2, 7e-3]):
            bound.step_(grads(k), dt)
            total += dt
            state = bound.state_dict()
            assert state["c_m"] == pytest.approx(
                -math.expm1(-total / TAU_M), rel=1e-12, abs=0.0
            )
            assert state["c_v"] == pytest.approx(
                -math.expm1(-total / TAU_V), rel=1e-12, abs=0.0
            )
        bound.step_(grads(5), 1e6 * TAU_V)
        state = bound.state_dict()
        assert set(state) == {"m", "v", "c_m", "c_v"}
        assert len(state["m"]) == len(state["v"]) == 2
        assert state["c_m"] == 1.0
        assert state["c_v"] == 1.0

    def test_weight_decay_alone_shrinks_by_one_minus_eta_w_per_step(self):
        # With g = 0, m = v = 0 and the Adam term is 0 / eps: theta decays by
        # 1 - eta w per step, with w paired with eta = elapsed / tau.
        ps = params()
        p0 = clones(ps)
        bound = ADAMW.bind(ps)
        for _ in range(5):
            bound.step_([torch.zeros_like(p) for p in ps], DT)
        eta = DT / TAU
        for p, p_0 in zip(ps, p0, strict=True):
            assert_close(p, p_0 * (1 - eta * W) ** 5, rtol=1e-14)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        ("optimizer", "moments", "masses"), STATEFUL, ids=STATEFUL_IDS
    )
    def test_the_moments_take_the_dtype_of_the_parameters(
        self,
        dtype: torch.dtype,
        optimizer: pc.optim.Optimizer,
        moments: list[str],
        masses: list[str],
    ):
        bound = optimizer.bind(params(dtype))
        bound.step_(grads(0, dtype), DT)
        state = bound.state_dict()
        assert set(state) == {*moments, *masses}
        for key in moments:
            for moment in state[key]:
                assert moment.dtype == dtype

    @pytest.mark.cuda
    def test_on_cuda_the_moments_are_on_cuda_and_the_step_is_that_of_the_cpu(self):
        cpu, cuda = params(), [p.cuda() for p in params()]
        on_cpu, on_cuda = ADAMW.bind(cpu), ADAMW.bind(cuda)
        for k in range(3):
            on_cpu.step_(grads(k), DT)
            on_cuda.step_([g.cuda() for g in grads(k)], DT)
        state = on_cuda.state_dict()
        for moment in [*state["m"], *state["v"]]:
            assert moment.device.type == "cuda"
        for p, q in zip(cuda, cpu, strict=True):
            assert_close(p.cpu(), q, rtol=1e-12)

    def test_zero_elapsed_first_moves_nothing_and_keeps_the_mass_at_zero(self):
        # Before the first step c = 0, and m_hat would be 0 / 0.
        ps = params()
        p0 = clones(ps)
        bound = ADAMW.bind(ps)
        bound.step_(grads(0), 0.0)
        for p, p_0 in zip(ps, p0, strict=True):
            assert torch.equal(p, p_0)
        state = bound.state_dict()
        assert state["c_m"] == 0.0
        assert state["c_v"] == 0.0


OPTIMIZERS = [pc.optim.SGD(tau=TAU), SGD_M, RMSPROP, ADAMW]
IDS = ["sgd", "sgd-momentum", "rmsprop", "adamw"]


class TestBound:
    @pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=IDS)
    def test_binds_no_parameters(self, optimizer: pc.optim.Optimizer):
        optimizer.bind([]).step_([], DT)

    @pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=IDS)
    def test_a_gradient_per_parameter(self, optimizer: pc.optim.Optimizer):
        bound = optimizer.bind(params())
        with pytest.raises(ValueError):
            bound.step_(grads(0)[:1], DT)


def saved(state: dict[str, Any]) -> dict[str, Any]:
    """``state`` through ``torch.save`` and ``torch.load(weights_only=True)``."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


class TestState:
    @pytest.mark.parametrize("optimizer", OPTIMIZERS, ids=IDS)
    def test_a_restored_optimizer_goes_on_as_the_same_one(
        self, optimizer: pc.optim.Optimizer
    ):
        # The parameters belong to the node: the restored one binds a copy of
        # them, and takes the state saved after 3 steps.
        ps = params()
        first = optimizer.bind(ps)
        for k in range(3):
            first.step_(grads(k), DT * (k + 1))
        qs = clones(ps)
        restored = optimizer.bind(qs)
        restored.load_state_dict(saved(first.state_dict()))
        for k in range(3, 8):
            first.step_(grads(k), DT)
            restored.step_(grads(k), DT)
        for p, q in zip(ps, qs, strict=True):
            assert torch.equal(q, p)
        assert_same_tree(restored.state_dict(), first.state_dict())

    @pytest.mark.parametrize(
        ("optimizer", "moments", "masses"), STATEFUL, ids=STATEFUL_IDS
    )
    def test_the_masses_load_as_floats(
        self, optimizer: pc.optim.Optimizer, moments: list[str], masses: list[str]
    ):
        bound = optimizer.bind(params())
        bound.step_(grads(0), DT)
        state = saved(bound.state_dict())
        for key in masses:
            assert type(state[key]) is float

    @pytest.mark.parametrize(
        ("optimizer", "moments", "masses"), STATEFUL, ids=STATEFUL_IDS
    )
    def test_loading_copies_the_moments(
        self, optimizer: pc.optim.Optimizer, moments: list[str], masses: list[str]
    ):
        source = optimizer.bind(params())
        source.step_(grads(0), DT)
        tree = source.state_dict()
        bound = optimizer.bind(params())
        bound.load_state_dict(tree)
        state = bound.state_dict()
        mine = [t for key in moments for t in state[key]]
        before = [t.clone() for t in mine]
        for key in moments:
            for t in tree[key]:
                t.add_(1.0)
        for t, t_0 in zip(mine, before, strict=True):
            assert torch.equal(t, t_0)


class TestChecks:
    @pytest.mark.parametrize("tau", [0.0, -1.0, math.nan, math.inf])
    def test_sgd_tau_must_be_finite_and_positive(self, tau: float):
        with pytest.raises(ValueError, match="SGD.tau"):
            pc.optim.SGD(tau=tau)

    @pytest.mark.parametrize("tau_m", [0.0, -1.0, math.nan, math.inf])
    def test_sgd_tau_m_must_be_finite_and_positive_when_given(self, tau_m: float):
        with pytest.raises(ValueError, match="SGD.tau_m"):
            pc.optim.SGD(tau=TAU, tau_m=tau_m)

    @pytest.mark.parametrize(
        ("field", "value"), [("tau", math.inf), ("tau_v", 0.0), ("eps", -1.0)]
    )
    def test_rmsprop_time_constants_and_eps_must_be_finite_and_positive(
        self, field: str, value: float
    ):
        values = {"tau": TAU, "tau_v": TAU_V, field: value}
        with pytest.raises(ValueError) as exc:
            pc.optim.RMSProp(**values)
        assert f"RMSProp.{field} must" in str(exc.value)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("tau", 0.0), ("tau_m", -1.0), ("tau_v", math.nan), ("eps", math.inf)],
    )
    def test_adamw_time_constants_and_eps_must_be_finite_and_positive(
        self, field: str, value: float
    ):
        values = {"tau": TAU, "tau_m": TAU_M, "tau_v": TAU_V, field: value}
        with pytest.raises(ValueError) as exc:
            pc.optim.AdamW(**values)
        assert f"AdamW.{field} must" in str(exc.value)

    @pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
    def test_adamw_weight_decay_must_be_finite_and_not_negative(self, value: float):
        with pytest.raises(ValueError, match="AdamW.weight_decay must"):
            pc.optim.AdamW(tau=TAU, tau_m=TAU_M, tau_v=TAU_V, weight_decay=value)
        pc.optim.AdamW(tau=TAU, tau_m=TAU_M, tau_v=TAU_V, weight_decay=0.0)

    def test_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            pc.optim.SGD(0.5)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            pc.optim.SGD(0.5, 0.1)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            pc.optim.RMSProp(TAU, TAU_V)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            pc.optim.AdamW(TAU, TAU_M, TAU_V)  # pyright: ignore[reportCallIssue]


# Runs in a subprocess: building a torch.optim optimizer imports
# torch._dynamo, which re-enables the GIL of the free-threaded build. The GIL is
# read before that, so the same fresh interpreter also shows that binding and
# stepping the optimizers of mujou keep it disabled.
TORCH_OPTIM = """
import math
import sys
import torch
import mujou.pc as pc

dt, tau, tau_m, tau_v, eps, w = 4e-3, 0.3, 0.02, 0.1, 1e-3, 0.1
generator = torch.Generator().manual_seed(0)
p0 = [
    torch.randn(4, 3, dtype=torch.float64, generator=generator),
    torch.randn(3, dtype=torch.float64, generator=generator),
]
grads = [
    [1e-3 * torch.randn(p.shape, dtype=torch.float64, generator=generator)
     for p in p0]
    for _ in range(20)
]
ours = [p.clone() for p in p0]
bound = pc.optim.AdamW(tau=tau, tau_m=tau_m, tau_v=tau_v, eps=eps, weight_decay=w).bind(ours)
for gs in grads:
    bound.step_(gs, dt)
pc.optim.SGD(tau=tau).bind([p.clone() for p in p0]).step_(grads[0], dt)

# Momentum SGD and RMSProp from a state whose mass is 1: torch has no bias
# correction, and the two agree from there.
m0 = [1e-3 * torch.randn(p.shape, dtype=torch.float64, generator=generator)
      for p in p0]
v0 = [(1e-3 * torch.randn(p.shape, dtype=torch.float64, generator=generator)) ** 2
      for p in p0]
ours_sgd = [p.clone() for p in p0]
bound = pc.optim.SGD(tau=tau, tau_m=tau_m).bind(ours_sgd)
bound.load_state_dict({"m": m0, "c_m": 1.0})
for gs in grads:
    bound.step_(gs, dt)
ours_rmsprop = [p.clone() for p in p0]
bound = pc.optim.RMSProp(tau=tau, tau_v=tau_v, eps=eps).bind(ours_rmsprop)
bound.load_state_dict({"v": v0, "c_v": 1.0})
for gs in grads:
    bound.step_(gs, dt)
print(sys._is_gil_enabled())

from torch.optim import SGD, AdamW, RMSprop


def run(optimizer, params):
    for gs in grads:
        for p, g in zip(params, gs):
            p.grad = g.clone()
        optimizer.step()


def error(ours, theirs):
    return max(
        ((a - b.detach()).abs().max() / b.detach().abs().max()).item()
        for a, b in zip(ours, theirs)
    )


theirs = [p.clone().requires_grad_() for p in p0]
optimizer = AdamW(
    theirs,
    lr=dt / tau,
    betas=(math.exp(-dt / tau_m), math.exp(-dt / tau_v)),
    eps=eps,
    weight_decay=w,
    foreach=False,
)
run(optimizer, theirs)
print("ok" if error(ours, theirs) < 1e-10 else f"adamw {error(ours, theirs)}")

beta = math.exp(-dt / tau_m)
theirs = [p.clone().requires_grad_() for p in p0]
optimizer = SGD(theirs, lr=dt / tau, momentum=beta, dampening=beta, foreach=False)
for p, m in zip(theirs, m0):
    optimizer.state[p]["momentum_buffer"] = m.clone()
run(optimizer, theirs)
e = error(ours_sgd, theirs)
print("ok" if e < 1e-10 else f"sgd {e}")

theirs = [p.clone().requires_grad_() for p in p0]
optimizer = RMSprop(
    theirs, lr=dt / tau, alpha=math.exp(-dt / tau_v), eps=eps, foreach=False
)
for p, v in zip(theirs, v0):
    optimizer.state[p] = {"step": torch.tensor(0.0), "square_avg": v.clone()}
run(optimizer, theirs)
e = error(ours_rmsprop, theirs)
print("ok" if e < 1e-10 else f"rmsprop {e}")
"""


class TestTorchOptim:
    def test_a_constant_dt_is_torch_optim_and_keeps_the_gil_disabled(self):
        # lr = dt / tau, betas = exp(-dt / tau_m), exp(-dt / tau_v): the same
        # AdamW to rounding (the masses replace the step count). Momentum SGD
        # (momentum = dampening = exp(-dt / tau_m)) and RMSProp
        # (alpha = exp(-dt / tau_v)) from a mass of 1, where the bias
        # correction of mujou is 1 and torch has none: the same to rounding.
        done = subprocess.run(
            [sys.executable, "-c", TORCH_OPTIM],
            capture_output=True,
            text=True,
            check=False,
            # PYTHON_GIL=0 would keep the GIL off whatever is imported.
            env={k: v for k, v in os.environ.items() if k != "PYTHON_GIL"},
        )
        assert done.returncode == 0, done.stderr
        gil, *result = done.stdout.splitlines()
        assert result == ["ok", "ok", "ok"], done.stdout
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert gil == "False"
