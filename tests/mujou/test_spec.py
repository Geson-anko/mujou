from typing import TypedDict, cast

import pytest
import torch

from mujou.spec import (
    DLPACK,
    BoolSpec,
    DictSpec,
    FloatSpec,
    IntSpec,
    TensorSpec,
    TimestampSpec,
)
from mujou.time import Timestamp


class Feedback(TypedDict):
    error: torch.Tensor
    target: torch.Tensor


class Reversed(TypedDict):
    target: torch.Tensor
    error: torch.Tensor


class Mixed(TypedDict):
    h: torch.Tensor
    jumps: int
    gain: float
    on: bool
    time: Timestamp


ALL_DTYPES = list(DLPACK)
FLOAT_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def feedback_spec() -> DictSpec[Feedback]:
    return DictSpec(Feedback, error=TensorSpec((2, 3)), target=TensorSpec((2,)))


def mixed_spec() -> DictSpec[Mixed]:
    return DictSpec(
        Mixed,
        h=TensorSpec((2,)),
        jumps=IntSpec(),
        gain=FloatSpec(),
        on=BoolSpec(),
        time=TimestampSpec(),
    )


class TestTensorSpec:
    def test_the_dtype_is_float32_by_default_whatever_torch_s_default(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
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
    def test_rand_follows_the_generator_and_is_in_the_unit_interval(
        self, dtype: torch.dtype
    ):
        spec = TensorSpec((1000,), dtype)
        a, b = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        x = spec.rand(generator=a)
        assert torch.equal(x, spec.rand(generator=b))
        # Both generators advanced alike, so their next draws match too.
        assert torch.equal(spec.rand(generator=a), spec.rand(generator=b))
        assert x.dtype == dtype
        assert bool((x >= 0).all()) and bool((x < 1).all())

    def test_empty_has_the_shape_and_dtype_on_the_cpu(self):
        spec = TensorSpec((2, 3), torch.int16)
        x = spec.empty()
        assert x.shape == (2, 3)
        assert x.dtype == torch.int16
        assert x.device == torch.device("cpu")

    def test_empty_is_filled_in_place(self):
        x = TensorSpec((2,), torch.float64).empty().fill_(0.5)
        assert torch.equal(x, torch.full((2,), 0.5, dtype=torch.float64))

    @pytest.mark.parametrize("make", ["zeros", "empty", "rand"])
    def test_makes_the_value_on_the_given_device(self, make: str):
        # The meta device has no memory, so it needs no accelerator to test with.
        spec = TensorSpec((2, 3), torch.float32)
        for device in ("meta", torch.device("meta")):
            x = cast(torch.Tensor, getattr(spec, make)(device=device))
            assert x.device == torch.device("meta")
            assert x.shape == (2, 3) and x.dtype == torch.float32

    @pytest.mark.cuda
    def test_rand_on_cuda_follows_a_generator_on_cuda(self):
        spec = TensorSpec((1000,), torch.float32)
        x = spec.rand(generator=torch.Generator("cuda").manual_seed(7), device="cuda")
        assert x.device.type == "cuda"
        again = torch.Generator("cuda").manual_seed(7)
        assert torch.equal(x, spec.rand(generator=again, device="cuda"))

    @pytest.mark.parametrize(
        "dtype", [torch.complex64, torch.uint16, torch.float8_e4m3fn]
    )
    def test_rejects_unsupported_dtypes(self, dtype: torch.dtype):
        with pytest.raises(TypeError, match=r"TensorSpec\.dtypes"):
            TensorSpec((2,), dtype)

    def test_takes_the_dtypes_the_codec_lays_out(self):
        assert TensorSpec.dtypes == set(DLPACK)

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

    def test_empty_has_the_keys_shapes_and_dtypes_on_the_cpu(self):
        spec = feedback_spec()
        msg = spec.empty()
        assert list(msg) == ["error", "target"]
        for key, value in cast(dict[str, torch.Tensor], msg).items():
            assert value.shape == spec.field(key, TensorSpec).shape
            assert value.dtype == spec.field(key, TensorSpec).dtype
            assert value.device == torch.device("cpu")

    @pytest.mark.parametrize("make", ["zeros", "empty"])
    def test_makes_the_message_on_the_given_device(self, make: str):
        msg = cast(
            dict[str, torch.Tensor], getattr(feedback_spec(), make)(device="meta")
        )
        assert all(v.device == torch.device("meta") for v in msg.values())

    def test_missing_key(self):
        with pytest.raises(TypeError, match="missing \\['target'\\]"):
            DictSpec(Feedback, error=TensorSpec((2,)))

    def test_unknown_key(self):
        with pytest.raises(TypeError, match="unknown \\['extra'\\]"):
            DictSpec(  # pyright: ignore[reportCallIssue]
                Feedback,
                error=TensorSpec((2,)),
                target=TensorSpec((2,)),
                extra=TensorSpec((2,)),
            )

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

    @pytest.mark.parametrize("make", ["zeros", "empty"])
    def test_scalar_values_are_zero(self, make: str):
        msg = getattr(mixed_spec(), make)()
        assert msg["jumps"] == 0 and type(msg["jumps"]) is int
        assert msg["gain"] == 0.0 and type(msg["gain"]) is float
        assert msg["on"] is False
        assert msg["time"] == Timestamp(ns=0)

    def test_field_returns_the_spec_of_the_kind(self):
        spec = mixed_spec()
        assert spec.field("h", TensorSpec) == TensorSpec((2,))
        assert spec.field("jumps", IntSpec) == IntSpec()
        assert spec.field("time", TimestampSpec) == TimestampSpec()

    def test_field_of_another_kind(self):
        with pytest.raises(TypeError, match="'jumps' is IntSpec, not TensorSpec"):
            mixed_spec().field("jumps", TensorSpec)

    def test_field_of_no_key(self):
        with pytest.raises(KeyError, match="nope"):
            mixed_spec().field("nope", TensorSpec)


class TestIntSpec:
    def test_64_bits_signed_by_default(self):
        spec = IntSpec()
        assert (spec.bits, spec.signed) == (64, True)

    @pytest.mark.parametrize("bits", [8, 16, 32, 64])
    def test_takes_whole_bytes_up_to_64_bits(self, bits: int):
        assert IntSpec(bits, signed=False).bits == bits  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("bits", [0, 1, 12, 24, 128])
    def test_rejects_other_bits(self, bits: int):
        with pytest.raises(ValueError, match="8, 16, 32 or 64"):
            IntSpec(bits)  # pyright: ignore[reportArgumentType]

    def test_equality_includes_bits_and_signedness(self):
        assert IntSpec() == IntSpec(64, signed=True)
        assert hash(IntSpec(8)) == hash(IntSpec(8))
        assert IntSpec(8) != IntSpec(16)
        assert IntSpec(64) != IntSpec(64, signed=False)
        assert IntSpec() != TensorSpec((), torch.int64)

    def test_repr(self):
        assert repr(IntSpec(32, signed=False)) == "IntSpec(32, signed=False)"


class TestFloatBoolTimestampSpec:
    def test_equal_to_their_own_kind_only(self):
        kinds = [FloatSpec(), BoolSpec(), TimestampSpec()]
        for a in kinds:
            assert a == type(a)()
            assert hash(a) == hash(type(a)())
            assert all(a != b for b in kinds if b is not a)
        assert FloatSpec() != TensorSpec((), torch.float64)

    def test_repr(self):
        assert [repr(FloatSpec()), repr(BoolSpec()), repr(TimestampSpec())] == [
            "FloatSpec()",
            "BoolSpec()",
            "TimestampSpec()",
        ]
