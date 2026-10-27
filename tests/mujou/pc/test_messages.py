import pytest
import torch

from mujou import DictSpec, IntSpec, TensorSpec, TimestampSpec
from mujou.pc.messages import Activity, activity_spec
from mujou.types import Size


class TestActivitySpec:
    def test_fields(self):
        h, y = TensorSpec((4, 3)), TensorSpec((4, 10), torch.float64)
        assert dict(activity_spec(h, y, batch=4).specs) == {
            "h": h,
            "jumps": IntSpec(),
            "y": y,
            "mask": TensorSpec((4,), torch.bool),
            "time": TimestampSpec(),
        }

    def test_equal_arguments_give_equal_specs(self):
        def make() -> DictSpec[Activity]:
            return activity_spec(
                TensorSpec((4, 3)), TensorSpec((4,), torch.int64), batch=4
            )

        assert make() == make()

    def test_without_a_batch_the_mask_is_a_scalar(self):
        act = activity_spec(TensorSpec((3,)), TensorSpec((), torch.int64))
        assert act.specs["mask"] == TensorSpec((), torch.bool)

    def test_an_int_is_a_batch_of_one_dimension(self):
        h, y = TensorSpec((4, 3)), TensorSpec((4,), torch.int64)
        assert activity_spec(h, y, batch=4) == activity_spec(h, y, batch=(4,))

    def test_a_batch_of_several_dimensions(self):
        act = activity_spec(TensorSpec((2, 4, 3)), TensorSpec((2, 4)), batch=(2, 4))
        assert act.specs["mask"] == TensorSpec((2, 4), torch.bool)

    @pytest.mark.parametrize(
        ("h", "y", "batch"),
        [
            ((4, 3), (3,), 4),  # y
            ((3,), (4,), 4),  # h
            ((4,), (), 4),  # y too short
            ((2, 4, 3), (2, 3), (2, 4)),
        ],
    )
    def test_h_and_y_must_start_with_the_batch(
        self, h: tuple[int, ...], y: tuple[int, ...], batch: Size
    ):
        with pytest.raises(ValueError, match="start with the batch"):
            activity_spec(TensorSpec(h), TensorSpec(y), batch=batch)
