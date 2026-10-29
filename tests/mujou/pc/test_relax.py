"""The relaxers of ``mujou.pc``, which move the deviation ``r`` of a node along
the force on it: ``pc.relax.Euler``. The expected values are the formulas
written out here."""

from typing import Any, override

import pytest
import torch

import mujou
import mujou.pc as pc
from tests.mujou.nodes import F64

TAU_H = 4e-3


def randn(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=F64, generator=generator)


def r0() -> torch.Tensor:
    return randn(3, 5, seed=0)


def force(k: int) -> torch.Tensor:
    return randn(3, 5, seed=10 + k)


class TestEuler:
    @pytest.mark.parametrize("dt", [1e-3, 4e-3])
    def test_one_step_is_dt_over_tau_of_the_force(self, dt: float):
        r = r0()
        pc.relax.Euler().bind(r).step_(r, force(0), dt, TAU_H)
        torch.testing.assert_close(
            r, r0() - dt / TAU_H * force(0), rtol=1e-15, atol=1e-15
        )

    def test_has_no_state(self):
        assert pc.relax.Euler().bind(r0()).state_dict() == {}

    def test_reset_changes_nothing(self):
        r, bound = r0(), pc.relax.Euler().bind(r0())
        bound.reset_(torch.ones(r.shape[0], dtype=torch.bool))
        bound.step_(r, force(0), 1e-3, TAU_H)
        torch.testing.assert_close(
            r, r0() - 1e-3 / TAU_H * force(0), rtol=1e-15, atol=1e-15
        )


class TestBound:
    def test_is_a_bound_relaxer_with_a_state(self):
        # A node holds it as an attribute, so it is in the node's state.
        bound = pc.relax.Euler().bind(r0())
        assert isinstance(bound, pc.relax.BoundRelaxer)
        assert isinstance(bound, mujou.Stateful)


class TestAbstract:
    def test_a_relaxer_needs_bind(self):
        class NoBind(pc.relax.Relaxer):
            pass

        with pytest.raises(TypeError, match="bind"):
            NoBind()  # pyright: ignore[reportAbstractUsage]

    def test_a_bound_relaxer_needs_step_reset_and_its_state(self):
        class NoStep(pc.relax.BoundRelaxer):
            @override
            def reset_(self, samples: torch.Tensor) -> None:
                pass

            @override
            def state_dict(self) -> dict[str, Any]:
                return {}

            @override
            def load_state_dict(self, state: Any, /) -> None:
                pass

        class NoReset(pc.relax.BoundRelaxer):
            @override
            def step_(
                self, r: torch.Tensor, force: torch.Tensor, dt: float, tau_h: float
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
