import math

import pytest

from mujou._core.pc.time_constants import TimeConstants


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

    @pytest.mark.parametrize("value", [True, "1.0", None])
    def test_a_field_must_be_a_number(self, value: object):
        with pytest.raises(TypeError, match="TimeConstants.lam"):
            TimeConstants(h=1e-3, lam=value, theta=1.0)  # pyright: ignore[reportArgumentType]

    def test_fields_are_keyword_only(self):
        with pytest.raises(TypeError):
            TimeConstants(1e-3, 1.0, 1.0)  # pyright: ignore[reportCallIssue]
