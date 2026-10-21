import math

import pytest

from mujou.pc import TimeConstants


class TestTimeConstants:
    @pytest.mark.parametrize("h", [0.0, -1e-3, math.inf, math.nan])
    def test_h_must_be_finite_and_positive(self, h: float):
        with pytest.raises(ValueError, match="TimeConstants.h"):
            TimeConstants(h=h, lam=1.0, theta=1.0)

    @pytest.mark.parametrize("field", ["lam", "theta"])
    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
    def test_lam_and_theta_must_be_positive(self, field: str, value: float):
        values = {"h": 1e-3, "lam": 1.0, "theta": 1.0, field: value}
        with pytest.raises(ValueError, match=f"TimeConstants.{field}"):
            TimeConstants(**values)

    def test_lam_and_theta_may_be_infinite(self):
        tau = TimeConstants(h=1e-3, lam=math.inf, theta=math.inf)
        assert tau.lam == math.inf
        assert tau.theta == math.inf

    def test_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            TimeConstants(1e-3, 1.0, 1.0)  # pyright: ignore[reportCallIssue]

    def test_the_defaults_of_a_node(self):
        tau = TimeConstants(h=4e-3, lam=8e-3, theta=1.0)
        assert tau.period == pytest.approx(1e-3)
        assert tau.max_dt == pytest.approx(2e-3)  # h / 2
        assert TimeConstants(h=4e-3, lam=0.5e-3, theta=1.0).max_dt == pytest.approx(
            1.5e-3  # 3 * lam
        )
        assert TimeConstants(h=4e-3, lam=math.inf, theta=1.0).max_dt == 2e-3
