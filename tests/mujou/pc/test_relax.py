"""The relaxers of ``mujou.pc``, which move the activity ``h`` of a node along
the force on it: ``pc.Euler`` and ``pc.HeavyBall``. The expected values are the
formulas written out here."""

import io
import math
from typing import Any, override

import pytest
import torch

import mujou
import mujou.pc as pc
from tests.mujou.nodes import F64, assert_same_tree

TAU_H = 4e-3
TAU_V = 2e-3  # the time constant of the velocity of HeavyBall
HEAVY_BALL = pc.HeavyBall(tau=TAU_V)
RELAXERS = [pc.Euler(), HEAVY_BALL]
IDS = ["euler", "heavy-ball"]


def randn(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=F64, generator=generator)


def h0() -> torch.Tensor:
    return randn(3, 5, seed=0)


def force(k: int) -> torch.Tensor:
    return randn(3, 5, seed=10 + k)


class TestEuler:
    @pytest.mark.parametrize("dt", [1e-3, 4e-3])
    def test_one_step_is_dt_over_tau_of_the_force(self, dt: float):
        h = h0()
        pc.Euler().bind(h).step_(h, force(0), dt, TAU_H)
        torch.testing.assert_close(
            h, h0() - dt / TAU_H * force(0), rtol=1e-15, atol=1e-15
        )

    def test_has_no_state(self):
        assert pc.Euler().bind(h0()).state_dict() == {}

    def test_reset_changes_nothing(self):
        h, bound = h0(), pc.Euler().bind(h0())
        bound.reset_()
        bound.step_(h, force(0), 1e-3, TAU_H)
        torch.testing.assert_close(
            h, h0() - 1e-3 / TAU_H * force(0), rtol=1e-15, atol=1e-15
        )


class TestHeavyBall:
    def test_one_step_moves_the_velocity_then_h_along_it(self):
        # v <- e^(-dt/tau) v + force, then h <- h - dt/tau_h v.
        h, bound = h0(), HEAVY_BALL.bind(h0())
        dt = 1e-3
        expected, v = h0(), torch.zeros_like(h)
        for k in range(3):
            bound.step_(h, force(k), dt, TAU_H)
            v = math.exp(-dt / TAU_V) * v + force(k)
            expected = expected - dt / TAU_H * v
            torch.testing.assert_close(h, expected, rtol=1e-14, atol=1e-14)
            torch.testing.assert_close(
                bound.state_dict()["velocity"], v, rtol=1e-15, atol=1e-15
            )

    def test_the_first_step_is_the_plain_one(self):
        # The velocity starts at zero, so the first force moves h as Euler does.
        h, plain = h0(), h0()
        HEAVY_BALL.bind(h).step_(h, force(0), 1e-3, TAU_H)
        pc.Euler().bind(plain).step_(plain, force(0), 1e-3, TAU_H)
        torch.testing.assert_close(h, plain, rtol=1e-15, atol=1e-15)

    def test_a_step_of_no_time_changes_nothing(self):
        # A force of no time must not be stored in the velocity.
        h, bound = h0(), HEAVY_BALL.bind(h0())
        bound.step_(h, force(0), 0.0, TAU_H)
        assert torch.equal(h, h0())
        assert not bound.state_dict()["velocity"].any()

    def test_reset_drops_the_velocity(self):
        h, bound = h0(), HEAVY_BALL.bind(h0())
        bound.step_(h, force(0), 1e-3, TAU_H)
        assert bound.state_dict()["velocity"].any()
        bound.reset_()
        assert not bound.state_dict()["velocity"].any()
        # After a reset the next step is the first one again.
        moved, fresh = h.clone(), h.clone()
        bound.step_(moved, force(1), 1e-3, TAU_H)
        HEAVY_BALL.bind(fresh).step_(fresh, force(1), 1e-3, TAU_H)
        torch.testing.assert_close(moved, fresh, rtol=1e-15, atol=1e-15)

    def test_a_tiny_time_constant_is_the_plain_relaxation(self):
        # e^(-dt/tau) -> 0: the velocity is the force.
        h, plain = h0(), h0()
        bound = pc.HeavyBall(tau=1e-9).bind(h)
        for k in range(3):
            bound.step_(h, force(k), 1e-3, TAU_H)
            pc.Euler().bind(plain).step_(plain, force(k), 1e-3, TAU_H)
        torch.testing.assert_close(h, plain, rtol=1e-12, atol=1e-12)

    def test_no_force_keeps_h_where_it_is(self):
        h, bound = h0(), HEAVY_BALL.bind(h0())
        bound.step_(h, torch.zeros_like(h), 1e-3, TAU_H)
        assert torch.equal(h, h0())

    def test_a_constant_force_gives_the_geometric_series_of_the_velocity(self):
        # With a force held at 1, v -> 1 / (1 - e^(-dt/tau)): h moves that much
        # faster than with the plain relaxation.
        h = torch.zeros(1, dtype=F64)
        bound = HEAVY_BALL.bind(h)
        one = torch.ones_like(h)
        for _ in range(200):
            bound.step_(h, one, 1e-3, TAU_H)
        beta = math.exp(-1e-3 / TAU_V)
        torch.testing.assert_close(
            bound.state_dict()["velocity"],
            torch.tensor([1 / (1 - beta)], dtype=F64),
            rtol=1e-12,
            atol=0,
        )

    def test_the_velocity_has_the_shape_dtype_and_device_of_h(self):
        h = torch.zeros(2, 3, dtype=torch.float32)
        velocity = HEAVY_BALL.bind(h).state_dict()["velocity"]
        assert velocity.shape == h.shape
        assert velocity.dtype == h.dtype
        assert velocity.device == h.device


class TestBound:
    @pytest.mark.parametrize("relaxer", RELAXERS, ids=IDS)
    def test_is_a_bound_relaxer_with_a_state(self, relaxer: pc.Relaxer):
        # A node holds it as an attribute, so it is in the node's state.
        bound = relaxer.bind(h0())
        assert isinstance(bound, pc.BoundRelaxer)
        assert isinstance(bound, mujou.Stateful)


class TestAbstract:
    def test_a_relaxer_needs_bind(self):
        class NoBind(pc.Relaxer):
            pass

        with pytest.raises(TypeError, match="bind"):
            NoBind()  # pyright: ignore[reportAbstractUsage]

    def test_a_bound_relaxer_needs_step_reset_and_its_state(self):
        class NoStep(pc.BoundRelaxer):
            @override
            def reset_(self) -> None:
                pass

            @override
            def state_dict(self) -> dict[str, Any]:
                return {}

            @override
            def load_state_dict(self, state: Any, /) -> None:
                pass

        class NoReset(pc.BoundRelaxer):
            @override
            def step_(
                self, h: torch.Tensor, force: torch.Tensor, dt: float, tau_h: float
            ) -> None:
                pass

            @override
            def state_dict(self) -> dict[str, Any]:
                return {}

            @override
            def load_state_dict(self, state: Any, /) -> None:
                pass

        with pytest.raises(TypeError, match="step_"):
            NoStep()  # pyright: ignore[reportAbstractUsage]
        with pytest.raises(TypeError, match="reset_"):
            NoReset()  # pyright: ignore[reportAbstractUsage]


def saved(state: dict[str, Any]) -> dict[str, Any]:
    """``state`` through ``torch.save`` and ``torch.load(weights_only=True)``."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


class TestState:
    @pytest.mark.parametrize("relaxer", RELAXERS, ids=IDS)
    def test_a_restored_relaxer_goes_on_as_the_same_one(self, relaxer: pc.Relaxer):
        h = h0()
        first = relaxer.bind(h)
        for k in range(3):
            first.step_(h, force(k), 1e-3 * (k + 1), TAU_H)
        h_restored = h.clone()
        restored = relaxer.bind(h_restored)
        restored.load_state_dict(saved(first.state_dict()))
        for k in range(3, 8):
            first.step_(h, force(k), 1e-3, TAU_H)
            restored.step_(h_restored, force(k), 1e-3, TAU_H)
        assert torch.equal(h_restored, h)
        assert_same_tree(restored.state_dict(), first.state_dict())

    def test_the_state_shares_memory_with_the_relaxer(self):
        # As the state of an optimizer does: the node reads it in place.
        h, bound = h0(), HEAVY_BALL.bind(h0())
        velocity = bound.state_dict()["velocity"]
        bound.step_(h, force(0), 1e-3, TAU_H)
        assert velocity.any()

    def test_loading_copies_the_velocity(self):
        source = HEAVY_BALL.bind(h0())
        source.step_(h0(), force(0), 1e-3, TAU_H)
        tree = source.state_dict()
        bound = HEAVY_BALL.bind(h0())
        bound.load_state_dict(tree)
        before = bound.state_dict()["velocity"].clone()
        tree["velocity"].add_(1.0)
        assert torch.equal(bound.state_dict()["velocity"], before)


class TestChecks:
    @pytest.mark.parametrize("tau", [0.0, -1.0, math.nan, math.inf])
    def test_heavy_ball_tau_must_be_finite_and_positive(self, tau: float):
        with pytest.raises(ValueError, match="HeavyBall.tau"):
            pc.HeavyBall(tau=tau)

    def test_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            pc.HeavyBall(1e-3)  # pyright: ignore[reportCallIssue]

    def test_settings_are_frozen_values(self):
        with pytest.raises(AttributeError):
            HEAVY_BALL.tau = 1.0  # pyright: ignore[reportAttributeAccessIssue]
        assert pc.HeavyBall(tau=TAU_V) == HEAVY_BALL
