"""The relaxers of ``mujou.pc``, which move the deviation ``r`` of a node along
the force on it: ``pc.relax.Euler``. The expected values are the formulas
written out here."""

import pytest
import torch

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

    def test_has_no_state_and_reset_changes_nothing(self):
        r, bound = r0(), pc.relax.Euler().bind(r0())
        assert bound.state_dict() == {}
        bound.reset_(torch.ones(r.shape[0], dtype=torch.bool))
        bound.step_(r, force(0), 1e-3, TAU_H)
        torch.testing.assert_close(
            r, r0() - 1e-3 / TAU_H * force(0), rtol=1e-15, atol=1e-15
        )
