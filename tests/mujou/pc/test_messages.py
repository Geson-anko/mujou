import pytest
import torch

from mujou import DictSpec, TensorSpec
from mujou.pc import Activity, activity_spec


class TestActivitySpec:
    def test_fields(self):
        h, y = TensorSpec((4, 3)), TensorSpec((4, 10), torch.float64)
        assert dict(activity_spec(h, y).specs) == {
            "h": h,
            "jumps": TensorSpec((), torch.int64),
            "y": y,
            "mask": TensorSpec((4,), torch.bool),
            "time": TensorSpec((), torch.int64),
        }

    def test_equal_arguments_give_equal_specs(self):
        def make() -> DictSpec[Activity]:
            return activity_spec(TensorSpec((4, 3)), TensorSpec((4,), torch.int64))

        assert make() == make()

    @pytest.mark.parametrize("shape", [(3,), (5, 2), ()])
    def test_h_and_y_must_have_the_same_batch(self, shape: tuple[int, ...]):
        with pytest.raises(ValueError, match="same batch"):
            activity_spec(TensorSpec((4, 3)), TensorSpec(shape))
