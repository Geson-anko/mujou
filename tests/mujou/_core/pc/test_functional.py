import math
from collections.abc import Callable
from typing import override

import pytest
import torch
from torch import nn

from mujou._core.pc.functional import learn_, loss_vjp, relax_, vjp
from mujou._core.pc.losses import cross_entropy, mse
from mujou._core.pc.time_constants import TimeConstants

B = 3
EPS = 1e-6  # central differences: truncation ~EPS^2, rounding ~1e-16 / EPS
RTOL = 1e-7
ATOL = 1e-9


class ResBlock(nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.inner = nn.Linear(n, n)
        self.outer = nn.Linear(n, n)

    @override
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.outer(torch.tanh(self.inner(h)))


class WithUnused(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.unused = nn.Parameter(torch.ones(2))

    @override
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def _modules() -> dict[str, Callable[[], nn.Module]]:
    return {
        "linear-tanh": lambda: nn.Sequential(nn.Linear(4, 5), nn.Tanh()),
        "resblock": lambda: ResBlock(4),
    }


def _make(name: str) -> nn.Module:
    torch.manual_seed(0)
    return _modules()[name]().double()


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)).double()


def numerical_grad(f: Callable[[], float], x: torch.Tensor) -> torch.Tensor:
    """The central difference of ``f`` with respect to each element of ``x``,
    which ``f`` reads and this function perturbs in place."""
    grad = torch.zeros_like(x)
    with torch.no_grad():
        flat = x.view(-1)
        for i in range(flat.numel()):
            original = flat[i].item()
            flat[i] = original + EPS
            f_plus = f()
            flat[i] = original - EPS
            f_minus = f()
            flat[i] = original
            grad.view(-1)[i] = (f_plus - f_minus) / (2 * EPS)
    return grad


class TestVjp:
    @pytest.mark.parametrize("name", list(_modules()))
    def test_gradients_of_the_energy(self, name: str):
        module = _make(name)
        h_in = _randn(B, 4, seed=1)
        with torch.no_grad():
            mu0 = module(h_in)
        h = mu0 + _randn(*mu0.shape, seed=2)  # the activity, held fixed
        lam = _randn(*mu0.shape, seed=3)

        def energy() -> float:
            # E = 0.5 * ||h - f(h_in) + lam||^2
            with torch.no_grad():
                return (0.5 * (h - module(h_in) + lam).square().sum()).item()

        params = list(module.parameters())
        mu, g_in, g_theta = vjp(module, h_in, h - mu0 + lam, params)

        torch.testing.assert_close(mu, mu0, rtol=0, atol=0)
        assert g_in is not None
        torch.testing.assert_close(
            g_in, numerical_grad(energy, h_in), rtol=RTOL, atol=ATOL
        )
        assert len(g_theta) == len(params)
        for p, g in zip(params, g_theta, strict=True):
            torch.testing.assert_close(
                g, numerical_grad(energy, p), rtol=RTOL, atol=ATOL
            )

    def test_without_the_input(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        delta = _randn(B, 5, seed=2)
        params = list(module.parameters())
        _, g_in, g_theta = vjp(module, h_in, delta, params, wrt_input=False)
        _, _, expected = vjp(module, h_in, delta, params)
        assert g_in is None
        for g, e in zip(g_theta, expected, strict=True):
            torch.testing.assert_close(g, e, rtol=0, atol=0)

    def test_prediction_only(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        mu, g_in, g_theta = vjp(module, h_in, torch.zeros(B, 5), wrt_input=False)
        with torch.no_grad():
            torch.testing.assert_close(mu, module(h_in), rtol=0, atol=0)
        assert g_in is None
        assert g_theta == []

    def test_unused_parameters_get_zeros(self):
        torch.manual_seed(0)
        module = WithUnused().double()
        _, _, g_theta = vjp(
            module, _randn(B, 4, seed=1), _randn(B, 4, seed=2), [module.unused]
        )
        assert torch.equal(g_theta[0], torch.zeros(2, dtype=torch.float64))

    def test_input_is_not_changed_or_attached(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        before = h_in.clone()
        mu, g_in, _ = vjp(module, h_in, _randn(B, 5, seed=2))
        assert torch.equal(h_in, before)
        assert not h_in.requires_grad
        assert mu.grad_fn is None
        assert g_in is not None
        assert g_in.grad_fn is None

    @pytest.mark.parametrize("wrt_input", [True, False])
    def test_delta_must_have_the_shape_of_mu(self, wrt_input: bool):
        module = _make("linear-tanh")
        with pytest.raises(ValueError, match="delta"):
            vjp(module, _randn(B, 4, seed=1), torch.zeros(B), wrt_input=wrt_input)


class TestLossVjp:
    @pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
    def test_gradients_of_the_masked_loss(self, loss_name: str):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        mask = torch.tensor([True, False, True])
        if loss_name == "mse":
            loss, y = mse, _randn(B, 5, seed=2)
        else:
            loss, y = cross_entropy, torch.tensor([4, 0, 2])

        def ell() -> float:
            # sum over b of mask_b * loss_b
            with torch.no_grad():
                ell_b = loss(module(h_in), y)
                return sum(ell_b[b].item() for b in range(B) if mask[b])

        params = list(module.parameters())
        y_hat, g_in, g_theta = loss_vjp(module, h_in, y, mask, loss, params)

        with torch.no_grad():
            torch.testing.assert_close(y_hat, module(h_in), rtol=0, atol=0)
        assert g_in is not None
        torch.testing.assert_close(
            g_in, numerical_grad(ell, h_in), rtol=RTOL, atol=ATOL
        )
        assert torch.equal(g_in[1], torch.zeros(4, dtype=torch.float64))
        for p, g in zip(params, g_theta, strict=True):
            torch.testing.assert_close(g, numerical_grad(ell, p), rtol=RTOL, atol=ATOL)

    def test_a_masked_out_target_does_not_matter(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        mask = torch.tensor([True, False, True])
        y = _randn(B, 5, seed=2)
        y_nan = y.clone()
        y_nan[1] = math.nan
        params = list(module.parameters())
        _, g_in, g_theta = loss_vjp(module, h_in, y, mask, mse, params)
        _, g_in_nan, g_theta_nan = loss_vjp(module, h_in, y_nan, mask, mse, params)
        assert g_in is not None
        assert g_in_nan is not None
        assert torch.equal(g_in_nan, g_in)
        for g, g_nan in zip(g_theta, g_theta_nan, strict=True):
            assert torch.equal(g_nan, g)

    def test_without_the_input(self):
        module = _make("linear-tanh")
        mask = torch.tensor([True, True, False])
        params = list(module.parameters())
        args = (module, _randn(B, 4, seed=1), _randn(B, 5, seed=2), mask, mse)
        _, g_in, g_theta = loss_vjp(*args, params, wrt_input=False)
        _, _, expected = loss_vjp(*args, params)
        assert g_in is None
        for g, e in zip(g_theta, expected, strict=True):
            torch.testing.assert_close(g, e, rtol=0, atol=0)

    def test_loss_must_be_per_sample(self):
        module = _make("linear-tanh")

        def mean_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return mse(y_hat, y).mean()

        with pytest.raises(ValueError, match="each sample"):
            loss_vjp(
                module,
                _randn(B, 4, seed=1),
                _randn(B, 5, seed=2),
                torch.ones(B, dtype=torch.bool),
                mean_loss,
            )


class TestRelax:
    TAU = TimeConstants(h=4e-3, lam=8e-3, theta=1.0)

    @staticmethod
    def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _randn(B, 5, seed=1), _randn(B, 5, seed=2), _randn(B, 5, seed=3)

    @staticmethod
    def _euler(
        u: torch.Tensor,
        lam: torch.Tensor,
        g: torch.Tensor,
        s: float,
        n: int,
        tau: TimeConstants,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # n explicit Euler steps of s with g fixed; lam moves with the moved u.
        for _ in range(n):
            u = u - s / tau.h * (u + lam + g)
            lam = lam + s / tau.lam * u
        return u, lam

    def test_one_step(self):
        u, lam, g = self._state()
        u0, lam0 = u.clone(), lam.clone()
        dt = 1e-3  # <= tau.h / 2
        relax_(u, lam, g, dt, self.TAU)
        u1 = u0 - dt / self.TAU.h * (u0 + lam0 + g)
        torch.testing.assert_close(u, u1, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(
            lam, lam0 + dt / self.TAU.lam * u1, rtol=1e-15, atol=1e-15
        )

    @pytest.mark.parametrize(
        ("dt", "max_step", "n"),
        [(4e-3, None, 2), (4e-3, 1e-3, 4), (3.5e-3, 1e-3, 4), (5e-4, 1e-3, 1)],
    )
    def test_split_into_euler_steps(self, dt: float, max_step: float | None, n: int):
        u, lam, g = self._state()
        expected_u, expected_lam = self._euler(u, lam, g, dt / n, n, self.TAU)
        relax_(u, lam, g, dt, self.TAU, max_step=max_step)
        torch.testing.assert_close(u, expected_u, rtol=1e-14, atol=1e-15)
        torch.testing.assert_close(lam, expected_lam, rtol=1e-14, atol=1e-15)

    def test_split_differs_from_one_long_step(self):
        # So that the test above would catch a missing split.
        u, lam, g = self._state()
        one_u, _ = self._euler(u, lam, g, 4e-3, 1, self.TAU)
        relax_(u, lam, g, 4e-3, self.TAU)
        assert not torch.allclose(u, one_u)

    def test_infinite_lam_time_constant_keeps_the_dual(self):
        u, lam, g = self._state()
        lam0 = lam.clone()
        relax_(u, lam, g, 1e-2, TimeConstants(h=4e-3, lam=math.inf, theta=1.0))
        assert torch.equal(lam, lam0)

    def test_zero_dt_changes_nothing(self):
        u, lam, g = self._state()
        u0, lam0 = u.clone(), lam.clone()
        relax_(u, lam, g, 0.0, self.TAU)
        assert torch.equal(u, u0)
        assert torch.equal(lam, lam0)

    @pytest.mark.parametrize("max_step", [0.0, -1e-3, math.inf, math.nan])
    def test_max_step_must_be_finite_and_positive(self, max_step: float):
        u, lam, g = self._state()
        with pytest.raises(ValueError, match="max_step"):
            relax_(u, lam, g, 1e-3, self.TAU, max_step=max_step)


class TestLearn:
    def test_step_on_the_mean_gradient(self):
        tau = TimeConstants(h=4e-3, lam=8e-3, theta=0.5)
        params = [_randn(4, 5, seed=1), _randn(5, seed=2)]
        g_theta = [_randn(4, 5, seed=3), _randn(5, seed=4)]
        before = [p.clone() for p in params]
        elapsed = 3e-3
        learn_(params, g_theta, elapsed, tau, batch=B)
        for p, p0, g in zip(params, before, g_theta, strict=True):
            torch.testing.assert_close(
                p, p0 - elapsed / tau.theta * g / B, rtol=1e-15, atol=1e-15
            )

    def test_infinite_theta_time_constant_keeps_the_parameters(self):
        tau = TimeConstants(h=4e-3, lam=8e-3, theta=math.inf)
        params = [_randn(4, 5, seed=1)]
        before = params[0].clone()
        learn_(params, [_randn(4, 5, seed=2)], 1.0, tau, batch=B)
        assert torch.equal(params[0], before)

    def test_one_gradient_per_parameter(self):
        tau = TimeConstants(h=4e-3, lam=8e-3, theta=0.5)
        params = [_randn(4, seed=1), _randn(4, seed=2)]
        before = [p.clone() for p in params]
        with pytest.raises(ValueError, match="one gradient per parameter"):
            learn_(params, [_randn(4, seed=3)], 1e-3, tau, batch=B)
        for p, p0 in zip(params, before, strict=True):
            assert torch.equal(p, p0)
