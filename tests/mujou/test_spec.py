from typing import NotRequired, TypedDict

import pytest
import torch

from mujou._spec import DLPACK_DTYPES, DictSpec, TensorSpec, fields, join, split


class Feedback(TypedDict):
    error: torch.Tensor
    target: torch.Tensor


class Reversed(TypedDict):
    target: torch.Tensor
    error: torch.Tensor


class WithInt(TypedDict):
    error: torch.Tensor
    count: int


class Nested(TypedDict):
    inner: Feedback


class WithOptional(TypedDict):
    error: torch.Tensor
    target: NotRequired[torch.Tensor]


class Partial(TypedDict, total=False):
    error: torch.Tensor


class Empty(TypedDict):
    pass


ALL_DTYPES = list(DLPACK_DTYPES)
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def feedback_spec() -> DictSpec[Feedback]:
    return DictSpec(Feedback, error=TensorSpec((2, 3)), target=TensorSpec((2,)))


class TestTensorSpec:
    def test_default_dtype_is_read_at_creation(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            spec = TensorSpec((2,))
            torch.set_default_dtype(torch.float32)
            assert spec.dtype == torch.float64
            assert TensorSpec((2,)).dtype == torch.float32
        finally:
            torch.set_default_dtype(previous)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_zeros_on_cpu_for_every_dtype(self, dtype: torch.dtype):
        x = TensorSpec((2, 3), dtype).zeros()
        assert x.shape == (2, 3)
        assert x.dtype == dtype
        assert x.device == torch.device("cpu")
        assert not x.any()

    def test_scalar_shape(self):
        assert TensorSpec(()).zeros().shape == ()

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_rand_is_deterministic_and_in_unit_interval(self, dtype: torch.dtype):
        spec = TensorSpec((1000,), dtype)
        x = spec.rand(seed=7)
        assert torch.equal(x, spec.rand(seed=7))
        assert not torch.equal(x, spec.rand(seed=8))
        assert x.dtype == dtype
        assert bool((x >= 0).all()) and bool((x < 1).all())

    def test_rand_follows_the_generator_state(self):
        spec = TensorSpec((5,), torch.float64)
        a, b = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        assert torch.equal(spec.rand(generator=a), spec.rand(generator=b))
        # Both generators advanced alike, so their next draws match too.
        assert torch.equal(spec.rand(generator=a), spec.rand(generator=b))

    @pytest.mark.parametrize("dtype", [torch.int64, torch.uint8, torch.bool])
    def test_rand_rejects_non_floating_dtypes(self, dtype: torch.dtype):
        with pytest.raises(TypeError, match="floating point"):
            TensorSpec((2,), dtype).rand(seed=0)

    def test_rand_rejects_seed_and_generator(self):
        with pytest.raises(ValueError, match="either seed or generator"):
            TensorSpec((2,)).rand(seed=0, generator=torch.Generator())

    @pytest.mark.parametrize(
        "dtype", [torch.complex64, torch.uint16, torch.float8_e4m3fn]
    )
    def test_rejects_unsupported_dtypes(self, dtype: torch.dtype):
        with pytest.raises(TypeError, match="not supported"):
            TensorSpec((2,), dtype)

    @pytest.mark.parametrize("shape", [(0,), (2, -1)])
    def test_rejects_dimensions_below_one(self, shape: tuple[int, ...]):
        with pytest.raises(ValueError, match=">= 1"):
            TensorSpec(shape)

    @pytest.mark.parametrize("shape", [(2.0,), (True,), ("3",)])
    def test_rejects_non_int_dimensions(self, shape: tuple[object, ...]):
        with pytest.raises(TypeError, match="must be int"):
            TensorSpec(shape)  # pyright: ignore[reportArgumentType]

    def test_shape_from_an_iterator(self):
        assert TensorSpec(d for d in (2, 3)).shape == (2, 3)  # pyright: ignore[reportArgumentType]

    def test_equality_includes_shape_and_dtype(self):
        spec = TensorSpec((2, 3), torch.float32)
        assert spec == TensorSpec([2, 3], torch.float32)
        assert hash(spec) == hash(TensorSpec(torch.Size([2, 3]), torch.float32))
        assert spec != TensorSpec((3, 2), torch.float32)
        assert spec != TensorSpec((2, 3), torch.float64)


class TestDictSpec:
    def test_keys_follow_the_typed_dict_order(self):
        spec = DictSpec(Feedback, target=TensorSpec((2,)), error=TensorSpec((2, 3)))
        assert list(spec.specs) == ["error", "target"]
        assert list(spec.zeros()) == ["error", "target"]

    def test_zeros(self):
        msg = feedback_spec().zeros()
        assert msg["error"].shape == (2, 3)
        assert msg["target"].shape == (2,)
        assert not msg["error"].any() and not msg["target"].any()

    def test_missing_key(self):
        with pytest.raises(ValueError, match="missing \\['target'\\]"):
            DictSpec(Feedback, error=TensorSpec((2,)))

    def test_unknown_key(self):
        with pytest.raises(ValueError, match="unknown \\['extra'\\]"):
            DictSpec(  # pyright: ignore[reportCallIssue]
                Feedback,
                error=TensorSpec((2,)),
                target=TensorSpec((2,)),
                extra=TensorSpec((2,)),
            )

    def test_rejects_non_tensor_values(self):
        with pytest.raises(TypeError, match="'count'"):
            DictSpec(WithInt, error=TensorSpec((2,)), count=TensorSpec((1,)))

    def test_rejects_nesting(self):
        with pytest.raises(TypeError, match="'inner'"):
            DictSpec(Nested, inner=TensorSpec((1,)))

    @pytest.mark.parametrize("typed_dict", [WithOptional, Partial])
    def test_rejects_optional_keys(self, typed_dict: type[WithOptional | Partial]):
        with pytest.raises(TypeError, match="required"):
            DictSpec(typed_dict, error=TensorSpec((2,)), target=TensorSpec((2,)))

    def test_rejects_a_typed_dict_without_keys(self):
        with pytest.raises(ValueError, match="none"):
            DictSpec(Empty)

    def test_rejects_values_that_are_not_tensor_specs(self):
        with pytest.raises(TypeError, match="TensorSpec for 'target'"):
            DictSpec(Feedback, error=TensorSpec((2,)), target=(2,))  # pyright: ignore[reportArgumentType]

    def test_rejects_a_plain_dict_type(self):
        with pytest.raises(TypeError, match="TypedDict"):
            DictSpec(dict, error=TensorSpec((2,)))  # pyright: ignore[reportArgumentType]

    def test_equality_includes_the_typed_dict_and_specs(self):
        spec = feedback_spec()
        assert spec == feedback_spec()
        assert hash(spec) == hash(feedback_spec())
        assert spec != DictSpec(
            Feedback, error=TensorSpec((2, 3)), target=TensorSpec((3,))
        )
        assert spec != DictSpec(
            Reversed, error=TensorSpec((2, 3)), target=TensorSpec((2,))
        )
        assert spec != TensorSpec((2, 3))


class TestFields:
    def test_tensor_is_one_unnamed_field(self):
        assert fields(TensorSpec((2, 3), torch.bfloat16)) == [("", (2, 3), (4, 16, 1))]

    def test_dict_has_one_field_per_key_in_order(self):
        spec = DictSpec(
            Feedback,
            error=TensorSpec((2, 3), torch.float64),
            target=TensorSpec((), torch.bool),
        )
        assert fields(spec) == [
            ("error", (2, 3), (2, 64, 1)),
            ("target", (), (6, 8, 1)),
        ]


class TestSplitJoin:
    def test_tensor_round_trip(self):
        spec = TensorSpec((2, 3))
        x = spec.rand(seed=0)
        (y,) = split(spec, x, where="a.next")
        assert y is x
        assert join(spec, [x]) is x

    def test_dict_round_trip_in_key_order(self):
        spec = feedback_spec()
        msg = {"target": torch.ones(2), "error": torch.ones(2, 3)}
        tensors = split(spec, msg, where="a.next")
        assert tensors[0] is msg["error"] and tensors[1] is msg["target"]
        back = join(spec, tensors)
        assert isinstance(back, dict)
        assert list(back) == ["error", "target"]
        assert back["error"] is msg["error"] and back["target"] is msg["target"]

    def test_shape_mismatch_names_the_place(self):
        with pytest.raises(ValueError, match="a.next") as exc:
            split(TensorSpec((2, 3)), torch.zeros(3, 2), where="a.next")
        assert "(2, 3)" in str(exc.value) and "(3, 2)" in str(exc.value)

    def test_dtype_mismatch_names_the_place_and_key(self):
        msg = {"error": torch.zeros(2, 3), "target": torch.zeros(2, dtype=torch.int64)}
        with pytest.raises(TypeError, match="int64") as exc:
            split(feedback_spec(), msg, where="a.next")
        assert "a.next['target']" in str(exc.value)

    def test_dict_shape_mismatch_names_the_key(self):
        msg = {"error": torch.zeros(3, 2), "target": torch.zeros(2)}
        with pytest.raises(ValueError, match="'error'"):
            split(feedback_spec(), msg, where="a.next")

    def test_rejects_sparse_tensors(self):
        sparse = torch.zeros(2, 3).to_sparse()
        with pytest.raises(TypeError, match="strided"):
            split(TensorSpec((2, 3)), sparse, where="a.next")

    def test_dict_keys_must_match(self):
        with pytest.raises(ValueError, match="missing \\['target'\\]"):
            split(feedback_spec(), {"error": torch.zeros(2, 3)}, where="a.next")

    @pytest.mark.parametrize(
        ("spec", "msg"),
        [
            (TensorSpec((2,)), [0.0, 0.0]),
            (feedback_spec(), torch.zeros(2)),
        ],
    )
    def test_wrong_message_type(
        self, spec: TensorSpec | DictSpec[Feedback], msg: object
    ):
        with pytest.raises(TypeError, match="a.next"):
            split(spec, msg, where="a.next")
