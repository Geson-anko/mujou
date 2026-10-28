import math
from collections.abc import Callable
from typing import Any, override

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mujou.pc import TimeConstants
from mujou.pc.functional import cross_entropy, loss_vjp, mse, relax_, vjp
from mujou.pc.loss import CrossEntropyLoss, MSELoss
from mujou.pc.relax import BoundRelaxer, Euler
from tests.mujou.pc._reference import Inertial

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
    with torch.random.fork_rng(devices=[]):
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
        mu, g_in, g_theta = vjp(module, h_in, lambda mu: h - mu + lam, params)

        torch.testing.assert_close(mu, mu0, rtol=0, atol=0)
        torch.testing.assert_close(
            g_in, numerical_grad(energy, h_in), rtol=RTOL, atol=ATOL
        )
        assert len(g_theta) == len(params)
        for p, g in zip(params, g_theta, strict=True):
            torch.testing.assert_close(
                g, numerical_grad(energy, p), rtol=RTOL, atol=ATOL
            )

    def test_without_parameters(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        mu, _, g_theta = vjp(module, h_in, torch.zeros_like)
        with torch.no_grad():
            torch.testing.assert_close(mu, module(h_in), rtol=0, atol=0)
        assert g_theta == []

    def test_unused_parameters_get_zeros(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = WithUnused().double()
        delta = _randn(B, 4, seed=2)
        _, _, g_theta = vjp(
            module, _randn(B, 4, seed=1), lambda _: delta, [module.unused]
        )
        assert torch.equal(g_theta[0], torch.zeros(2, dtype=torch.float64))

    def test_input_is_not_changed_or_attached(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        before = h_in.clone()
        delta = _randn(B, 5, seed=2)
        mu, g_in, _ = vjp(module, h_in, lambda _: delta)
        assert torch.equal(h_in, before)
        assert not h_in.requires_grad
        assert mu.grad_fn is None
        assert g_in.grad_fn is None

    def test_delta_is_made_from_mu_without_gradients(self):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        seen: list[torch.Tensor] = []

        def delta(mu: torch.Tensor) -> torch.Tensor:
            seen.append(mu)
            return torch.zeros_like(mu)

        mu, _, _ = vjp(module, h_in, delta, list(module.parameters()))
        assert torch.equal(seen[0], mu)
        assert not seen[0].requires_grad

    def test_delta_must_have_the_shape_of_mu(self):
        module = _make("linear-tanh")
        with pytest.raises(ValueError, match="delta"):
            vjp(module, _randn(B, 4, seed=1), lambda _: torch.zeros(B))


class TestLossVjp:
    @pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
    def test_gradients_of_the_masked_loss(self, loss_name: str):
        module = _make("linear-tanh")
        h_in = _randn(B, 4, seed=1)
        mask = torch.tensor([True, False, True])
        if loss_name == "mse":
            loss, y = MSELoss(B), _randn(B, 5, seed=2)
        else:
            loss, y = CrossEntropyLoss(B), torch.tensor([4, 0, 2])

        def ell() -> float:
            # sum over b of mask_b * loss_b
            with torch.no_grad():
                ell_b = loss(module(h_in), y)
                return sum(ell_b[b].item() for b in range(B) if mask[b])

        params = list(module.parameters())
        y_hat, g_in, g_theta = loss_vjp(module, h_in, y, mask, loss, params)

        with torch.no_grad():
            torch.testing.assert_close(y_hat, module(h_in), rtol=0, atol=0)
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
        loss = MSELoss(B)
        _, g_in, g_theta = loss_vjp(module, h_in, y, mask, loss, params)
        _, g_in_nan, g_theta_nan = loss_vjp(module, h_in, y_nan, mask, loss, params)
        assert torch.equal(g_in_nan, g_in)
        for g, g_nan in zip(g_theta, g_theta_nan, strict=True):
            assert torch.equal(g_nan, g)

    @pytest.mark.parametrize("loss_name", ["mse", "cross_entropy"])
    def test_without_a_target_g_in_is_zero(self, loss_name: str):
        # Every mask False: the energy is zero, and g_in is J^T 0 (y_hat finite).
        module = _make("linear-tanh")
        y = _randn(B, 5, seed=2) if loss_name == "mse" else torch.tensor([4, 0, 2])
        loss = MSELoss(B) if loss_name == "mse" else CrossEntropyLoss(B)
        params = list(module.parameters())
        no_target = torch.zeros(B, dtype=torch.bool)
        _, g_in, g_theta = loss_vjp(
            module, _randn(B, 4, seed=1), y, no_target, loss, params
        )
        assert torch.equal(g_in, torch.zeros_like(g_in))
        for g in g_theta:
            assert torch.equal(g, torch.zeros_like(g))

    def test_loss_must_be_per_sample(self):
        module = _make("linear-tanh")

        def mean_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return mse(y_hat, y, B).mean()

        with pytest.raises(ValueError, match="each sample"):
            loss_vjp(
                module,
                _randn(B, 4, seed=1),
                _randn(B, 5, seed=2),
                torch.ones(B, dtype=torch.bool),
                mean_loss,
            )

    def test_a_batch_of_two_dimensions_is_the_flattened_batch(self):
        # The samples of (2, 3) are those of (6,), and so are their gradients.
        module = _make("linear-tanh")
        h_in, y = _randn(6, 4, seed=1), _randn(6, 5, seed=2)
        mask = torch.tensor([True, False, True, True, False, True])
        params = list(module.parameters())
        y_hat, g_in, g_theta = loss_vjp(module, h_in, y, mask, MSELoss(6), params)
        y_hat2, g_in2, g_theta2 = loss_vjp(
            module,
            h_in.reshape(2, 3, 4),
            y.reshape(2, 3, 5),
            mask.reshape(2, 3),
            MSELoss((2, 3)),
            params,
        )
        assert torch.equal(y_hat2, y_hat.reshape(2, 3, 5))
        torch.testing.assert_close(g_in2, g_in.reshape(2, 3, 4), rtol=1e-14, atol=0)
        for g, g2 in zip(g_theta, g_theta2, strict=True):
            torch.testing.assert_close(g2, g, rtol=1e-14, atol=1e-16)

    def test_without_a_batch_the_mask_is_a_scalar(self):
        module = _make("linear-tanh")
        h_in, y = _randn(4, seed=1), _randn(5, seed=2)
        params = list(module.parameters())
        _, g_in, g_theta = loss_vjp(
            module, h_in, y, torch.tensor(True), MSELoss(), params
        )
        _, g_in1, g_theta1 = loss_vjp(
            module, h_in[None], y[None], torch.tensor([True]), MSELoss(1), params
        )
        assert torch.equal(g_in, g_in1[0])
        for g, g1 in zip(g_theta, g_theta1, strict=True):
            assert torch.equal(g, g1)
        _, g_in, _ = loss_vjp(module, h_in, y, torch.tensor(False), MSELoss())
        assert torch.equal(g_in, torch.zeros(4, dtype=torch.float64))


class TestRelax:
    TAU = TimeConstants(h=4e-3, lam=8e-3, r=math.inf)

    @staticmethod
    def state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``r``, ``lam`` and ``g``."""
        return tuple(_randn(B, 5, seed=seed) for seed in range(1, 4))  # pyright: ignore[reportReturnType]

    @pytest.mark.parametrize("dt", [1e-3, 4e-3])
    def test_one_euler_step(self, dt: float):
        # r first, then lam with the moved r; a long dt is one step.
        r, lam, g = self.state()
        r0, lam0 = r.clone(), lam.clone()
        relax_(r, lam, g, dt, self.TAU)
        r1 = r0 - dt / self.TAU.h * (r0 + lam0 + g)
        torch.testing.assert_close(r, r1, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(
            lam, lam0 + dt / self.TAU.lam * r1, rtol=1e-15, atol=1e-15
        )

    def test_infinite_lam_time_constant_keeps_the_dual(self):
        # Even when r is not finite: 0 * inf would make NaN.
        r, lam, g = self.state()
        r[0, 0] = math.inf
        lam0 = lam.clone()
        relax_(r, lam, g, 1e-3, TimeConstants(h=4e-3, lam=math.inf, r=math.inf))
        assert torch.equal(lam, lam0)

    def test_zero_dt_changes_nothing(self):
        r, lam, g = self.state()
        r0, lam0 = r.clone(), lam.clone()
        relax_(r, lam, g, 0.0, self.TAU)
        assert torch.equal(r, r0)
        assert torch.equal(lam, lam0)

    def test_the_decay_of_r(self):
        # tau_h dr/dt = -(r + lam + g) - (tau_h / tau_r) r; lam as before.
        tau = TimeConstants(h=4e-3, lam=8e-3, r=2e-3)
        r, lam, g = self.state()
        r0, lam0 = r.clone(), lam.clone()
        dt = 1e-3
        relax_(r, lam, g, dt, tau)
        r1 = r0 - dt / tau.h * ((1 + tau.h / tau.r) * r0 + lam0 + g)
        torch.testing.assert_close(r, r1, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(
            lam, lam0 + dt / tau.lam * r1, rtol=1e-15, atol=1e-15
        )


class TestRelaxWithRelaxer:
    TAU = TimeConstants(h=4e-3, lam=8e-3, r=2e-3)
    INERTIAL = Inertial(tau=2e-3)

    def test_euler_is_the_step_without_a_relaxer(self):
        r, lam, g = TestRelax.state()
        plain = (r.clone(), lam.clone())
        # dt / tau.h is not a power of two, so the two must round alike.
        relax_(r, lam, g, 1.3e-3, self.TAU, Euler().bind(r))
        relax_(plain[0], plain[1], g, 1.3e-3, self.TAU)
        assert torch.equal(r, plain[0]) and torch.equal(lam, plain[1])

    def test_a_relaxer_of_your_own_gets_r_the_force_dt_and_tau_h(self):
        # r is moved in place by the relaxer alone; lam follows with the moved
        # r.
        r, lam, g = TestRelax.state()
        r0, lam0 = r.clone(), lam.clone()
        calls: list[tuple[torch.Tensor, torch.Tensor, float, float]] = []

        class Halve(BoundRelaxer):
            @override
            def step_(
                self, r: torch.Tensor, force: torch.Tensor, dt: float, tau_h: float
            ) -> None:
                calls.append((r.clone(), force.clone(), dt, tau_h))
                r.sub_(0.5 * force)

            @override
            def reset_(self) -> None:
                pass

            @override
            def state_dict(self) -> dict[str, Any]:
                return {}

            @override
            def load_state_dict(self, state: Any, /) -> None:
                pass

        relax_(r, lam, g, 1e-3, self.TAU, Halve())
        ((seen_r, force, dt, tau_h),) = calls
        assert torch.equal(seen_r, r0)
        expected = (1 + self.TAU.h / self.TAU.r) * r0 + lam0 + g
        torch.testing.assert_close(force, expected, rtol=1e-15, atol=1e-15)
        assert (dt, tau_h) == (1e-3, self.TAU.h)
        r1 = r0 - 0.5 * force
        torch.testing.assert_close(r, r1, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(
            lam, lam0 + 1e-3 / self.TAU.lam * r1, rtol=1e-15, atol=1e-15
        )

    def test_an_update_of_no_time_changes_nothing(self):
        r, lam, g = TestRelax.state()
        relaxer = self.INERTIAL.bind(r)
        v = relaxer.state_dict()["velocity"]
        v.copy_(_randn(B, 5, seed=9))
        before = (r.clone(), lam.clone(), v.clone())
        relax_(r, lam, g, 0.0, self.TAU, relaxer)
        assert torch.equal(r, before[0]) and torch.equal(lam, before[1])
        assert torch.equal(v, before[2])

    def test_at_the_fixed_point_nothing_moves(self):
        # r = 0 and lam = -g: no force, and with no velocity there is none after.
        r, lam, g = TestRelax.state()
        r.zero_()
        lam.copy_(-g)
        r0, lam0 = r.clone(), lam.clone()
        relaxer = self.INERTIAL.bind(r)
        relax_(r, lam, g, 1e-3, self.TAU, relaxer)
        assert torch.equal(r, r0) and torch.equal(lam, lam0)
        assert not relaxer.state_dict()["velocity"].any()


def _pair(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _randn(*shape, seed=1), _randn(*shape, seed=2)


class TestMse:
    def test_mean_squared_error_of_each_sample(self):
        y_hat, y = _pair(B, 2, 4)
        ell = mse(y_hat, y, B)
        assert ell.shape == (B,)
        for b in range(B):
            expected = F.mse_loss(y_hat[b], y[b])
            torch.testing.assert_close(ell[b], expected, rtol=1e-14, atol=0)

    def test_mean_is_the_mean_loss_of_torch(self):
        # With the 1 / B that the nodes take before their optimizer, theta
        # follows the gradient of nn.MSELoss().
        y_hat, y = _pair(B, 5)
        torch.testing.assert_close(
            mse(y_hat, y, B).mean(), nn.MSELoss()(y_hat, y), rtol=1e-14, atol=0
        )

    def test_without_a_batch_is_one_sample(self):
        y_hat, y = _pair(5)
        ell = mse(y_hat, y)
        assert ell.shape == ()
        torch.testing.assert_close(ell, F.mse_loss(y_hat, y), rtol=1e-14, atol=0)

    def test_a_batch_of_two_dimensions_is_the_flattened_batch(self):
        y_hat, y = _pair(2, 3, 5)
        expected = mse(y_hat.reshape(6, 5), y.reshape(6, 5), 6).reshape(2, 3)
        assert torch.equal(mse(y_hat, y, (2, 3)), expected)

    def test_an_int_is_a_batch_of_one_dimension(self):
        y_hat, y = _pair(B, 5)
        assert torch.equal(mse(y_hat, y, B), mse(y_hat, y, (B,)))

    def test_shapes_must_match(self):
        with pytest.raises(ValueError, match="same shape"):
            mse(torch.zeros(3, 1), torch.zeros(3), 3)


class TestCrossEntropy:
    def test_of_each_sample(self):
        y_hat = _randn(B, 4, seed=1)
        y = torch.tensor([3, 0, 2])
        assert torch.equal(
            cross_entropy(y_hat, y, B), F.cross_entropy(y_hat, y, reduction="none")
        )

    def test_mean_is_the_mean_loss_of_torch(self):
        # Classes right after the batch, as in torch; the rest of a sample is
        # averaged over (each position of an image here).
        y_hat = _randn(B, 4, 2, 3, seed=1)
        y = torch.randint(0, 4, (B, 2, 3), generator=torch.Generator().manual_seed(0))
        ell = cross_entropy(y_hat, y, B)
        assert ell.shape == (B,)
        torch.testing.assert_close(
            ell.mean(), nn.CrossEntropyLoss()(y_hat, y), rtol=1e-14, atol=0
        )

    def test_probabilities_as_the_target(self):
        y_hat = _randn(B, 4, seed=1)
        y = torch.softmax(_randn(B, 4, seed=2), dim=1)
        assert torch.equal(
            cross_entropy(y_hat, y, B), F.cross_entropy(y_hat, y, reduction="none")
        )

    def test_without_a_batch_is_one_sample(self):
        y_hat, y = _randn(4, seed=1), torch.tensor(2)
        ell = cross_entropy(y_hat, y)
        assert ell.shape == ()
        torch.testing.assert_close(ell, F.cross_entropy(y_hat, y), rtol=1e-14, atol=0)

    def test_a_batch_of_two_dimensions_is_the_flattened_batch(self):
        # The classes are the dimension after the batch: the last one here.
        y_hat = _randn(2, 3, 4, seed=1)
        y = torch.randint(0, 4, (2, 3), generator=torch.Generator().manual_seed(0))
        expected = cross_entropy(y_hat.reshape(6, 4), y.reshape(6), 6).reshape(2, 3)
        assert torch.equal(cross_entropy(y_hat, y, (2, 3)), expected)


class TestLossBatch:
    @pytest.mark.parametrize("loss", [mse, cross_entropy])
    def test_the_tensors_must_start_with_the_batch(
        self, loss: Callable[..., torch.Tensor]
    ):
        # (2, 8) would reshape into 4 samples that mix the two.
        y = torch.zeros(2, 8)
        with pytest.raises(ValueError, match="start with the batch"):
            loss(torch.zeros(2, 8), y, 4)
