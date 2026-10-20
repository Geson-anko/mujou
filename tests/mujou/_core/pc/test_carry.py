import pytest
import torch

from mujou._core.pc.carry import Carried, Prediction, Sample, carry_spec
from mujou._core.spec import DictSpec, TensorSpec


class TestCarrySpec:
    def test_fields(self):
        target = TensorSpec((4, 10), torch.float64)
        spec = carry_spec(target, batch=4)
        assert spec.typed_dict is Carried
        assert dict(spec.specs) == {
            "y": target,
            "mask": TensorSpec((4,), torch.bool),
            "time": TensorSpec((), torch.int64),
        }

    def test_equal_arguments_give_equal_specs(self):
        def make() -> DictSpec[Carried]:
            return carry_spec(TensorSpec((4,), torch.int64), batch=4)

        assert make() == make()

    @pytest.mark.parametrize("shape", [(3,), (5, 2), ()])
    def test_target_must_lead_with_the_batch(self, shape: tuple[int, ...]):
        with pytest.raises(ValueError, match="batch 4"):
            carry_spec(TensorSpec(shape), batch=4)

    def test_target_must_be_a_tensor_spec(self):
        with pytest.raises(TypeError, match="TensorSpec"):
            carry_spec((4,), batch=4)  # pyright: ignore[reportArgumentType]

    def test_batch_must_be_positive(self):
        with pytest.raises(ValueError, match=">= 1"):
            carry_spec(TensorSpec((1,)), batch=0)


class TestMessages:
    def test_sample_and_prediction_make_dict_specs(self):
        # Every key is a tensor, so both can be the messages of a port.
        x, y, mask = (
            TensorSpec((2, 3)),
            TensorSpec((2,), torch.int64),
            TensorSpec((2,), torch.bool),
        )
        DictSpec(Sample, x=x, y=y, mask=mask)
        DictSpec(
            Prediction,
            y_hat=TensorSpec((2, 5)),
            y=y,
            mask=mask,
            time=TensorSpec((), torch.int64),
        )
