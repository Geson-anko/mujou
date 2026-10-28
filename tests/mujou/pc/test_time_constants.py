import math

import pytest

from mujou.pc import TimeConstants


class TestTimeConstants:
    @pytest.mark.parametrize("h", [0.0, -1e-3, math.inf, math.nan])
    def test_h_must_be_finite_and_positive(self, h: float):
        with pytest.raises(ValueError, match="TimeConstants.h"):
            TimeConstants(h=h, lam=1.0, r=1.0)

    @pytest.mark.parametrize("lam", [0.0, -1.0, math.nan])
    def test_lam_must_be_positive(self, lam: float):
        with pytest.raises(ValueError, match="TimeConstants.lam"):
            TimeConstants(h=1e-3, lam=lam, r=1.0)

    def test_lam_may_be_infinite(self):
        tau = TimeConstants(h=1e-3, lam=math.inf, r=1.0)
        assert tau.lam == math.inf

    @pytest.mark.parametrize("r", [0.0, -1.0, math.nan])
    def test_r_must_be_positive(self, r: float):
        with pytest.raises(ValueError, match="TimeConstants.r"):
            TimeConstants(h=1e-3, lam=1.0, r=r)

    def test_r_is_required_and_may_be_infinite(self):
        # The decay of r keeps deep chains stable, so it is chosen, as lam is.
        with pytest.raises(TypeError):
            TimeConstants(h=1e-3, lam=1.0)  # pyright: ignore[reportCallIssue]
        assert TimeConstants(h=1e-3, lam=1.0, r=math.inf).r == math.inf

    def test_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            TimeConstants(1e-3, 1.0, 1.0)  # pyright: ignore[reportCallIssue]

    def test_the_defaults_of_a_node(self):
        tau = TimeConstants(h=4e-3, lam=8e-3, r=4e-3)
        assert tau.period == pytest.approx(1e-3)
        # The decay of r pulls r by (1 + h / r) / h: here twice as hard.
        assert tau.max_dt == pytest.approx(1e-3)  # h / (2 * (1 + h / r))
        assert TimeConstants(h=4e-3, lam=8e-3, r=math.inf).max_dt == pytest.approx(
            2e-3  # h / 2
        )
        assert TimeConstants(h=4e-3, lam=0.5e-3, r=math.inf).max_dt == pytest.approx(
            1.5e-3  # 3 * lam
        )
        assert TimeConstants(h=4e-3, lam=math.inf, r=math.inf).max_dt == 2e-3
