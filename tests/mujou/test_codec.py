import struct
from collections.abc import Sequence
from typing import Any, TypedDict

import pytest
import torch

from mujou._native import Outlet, ReadOnlyTimer, VirtualTimer
from mujou.codec import fields, from_core, schema, to_core
from mujou.spec import (
    DLPACK,
    BoolSpec,
    DictSpec,
    FloatSpec,
    IntSpec,
    MessageSpec,
    TensorSpec,
    TimestampSpec,
)
from mujou.time import Timestamp

CPU = torch.device("cpu")


class Feedback(TypedDict):
    error: torch.Tensor
    target: torch.Tensor


class Mixed(TypedDict):
    h: torch.Tensor
    jumps: int
    gain: float
    on: bool
    time: Timestamp


def mixed_spec() -> DictSpec[Mixed]:
    return DictSpec(
        Mixed,
        h=TensorSpec((2,)),
        jumps=IntSpec(),
        gain=FloatSpec(),
        on=BoolSpec(),
        time=TimestampSpec(),
    )


def feedback_spec() -> DictSpec[Feedback]:
    return DictSpec(Feedback, error=TensorSpec((2, 3)), target=TensorSpec((2,)))


def sample(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Every bit pattern the dtype may hold (NaN payloads, the sign of zero),
    from random bytes; a bool holds 0 or 1."""
    g = torch.Generator().manual_seed(0)
    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), generator=g).bool()
    size = torch.empty((), dtype=dtype).element_size()
    n = size * int(torch.tensor(shape).prod())
    raw = torch.randint(0, 256, (n,), generator=g, dtype=torch.uint8)
    return raw.view(dtype).reshape(tuple(shape))


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1).view(torch.uint8)


def through_the_core(
    spec: MessageSpec[Any], msg: object, label: str = "a.next"
) -> object:
    """Send ``msg`` through an outlet of ``spec`` and read it back, as a port
    and the host do."""
    outlet = Outlet(schema(spec), 0, label, clock=ReadOnlyTimer(VirtualTimer()))
    outlet.send(to_core(spec, msg, label))
    return from_core(spec, outlet.outbox.read().values, CPU)


class TestFields:
    def test_a_tensor_is_the_one_field_named_empty(self):
        spec = TensorSpec((2, 3))
        assert fields(spec) == {"": spec}

    def test_a_dict_has_one_field_per_key_in_order(self):
        spec = feedback_spec()
        assert list(fields(spec).items()) == list(spec.specs.items())


class TestRoundTrip:
    @pytest.mark.parametrize("dtype", list(DLPACK), ids=str)
    def test_a_tensor_round_trips_every_dtype_bit_for_bit(self, dtype: torch.dtype):
        spec = TensorSpec((3, 5), dtype)
        x = sample(spec.shape, dtype)
        y = through_the_core(spec, x)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == dtype and y.shape == x.shape
        assert torch.equal(bits(y), bits(x))
        assert y.data_ptr() != x.data_ptr()

    @pytest.mark.parametrize("dtype", list(DLPACK), ids=str)
    def test_a_dict_round_trips_every_dtype_bit_for_bit(self, dtype: torch.dtype):
        spec = DictSpec(
            Feedback, error=TensorSpec((2, 3), dtype), target=TensorSpec((), dtype)
        )
        msg = {"target": sample((), dtype), "error": sample((2, 3), dtype)}
        back = through_the_core(spec, msg)
        assert isinstance(back, dict)
        assert list(back) == ["error", "target"]
        for key in ("error", "target"):
            assert torch.equal(bits(back[key]), bits(msg[key])), key

    def test_always_copies(self):
        spec = TensorSpec((2,))
        x = torch.zeros(2)
        tensors = to_core(spec, x, "a.next")
        outlet = Outlet(schema(spec), 0, "a", clock=ReadOnlyTimer(VirtualTimer()))
        outlet.send(tensors)
        x.add_(1.0)  # after the send: must not reach the message
        views = outlet.outbox.read().values
        first, second = from_core(spec, views, CPU), from_core(spec, views, CPU)
        assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
        assert torch.equal(first, torch.zeros(2))
        first.add_(1.0)  # a received value is the receiver's own
        assert torch.equal(second, torch.zeros(2))

    def test_detaches_from_the_graph(self):
        x = torch.ones(2, requires_grad=True)
        (y,) = to_core(TensorSpec((2,)), x * 2, "a.next")
        assert isinstance(y, torch.Tensor) and not y.requires_grad


def read_initial(spec: MessageSpec[Any]) -> object:
    """The value an outlet of ``spec`` holds before its first send."""
    outlet = Outlet(schema(spec), 0, "a", clock=ReadOnlyTimer(VirtualTimer()))
    return from_core(spec, outlet.outbox.read().values, CPU)


INT_BOUNDS = [
    (bits, signed, value)
    for bits in (8, 16, 32, 64)
    for signed, low, high in (
        (True, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1),
        (False, 0, 2**bits - 1),
    )
    for value in (low, high)
]


class TestScalars:
    @pytest.mark.parametrize(("bits", "signed", "value"), INT_BOUNDS)
    def test_an_int_round_trips_at_the_bounds(
        self, bits: int, signed: bool, value: int
    ):
        back = through_the_core(IntSpec(bits, signed=signed), value)  # pyright: ignore[reportArgumentType]
        assert back == value and type(back) is int

    @pytest.mark.parametrize(("bits", "signed", "value"), INT_BOUNDS)
    def test_an_int_beyond_the_bounds_overflows(
        self, bits: int, signed: bool, value: int
    ):
        beyond = value + 1 if value > 0 else value - 1  # the high or low bound
        spec = IntSpec(bits, signed=signed)  # pyright: ignore[reportArgumentType]
        with pytest.raises(OverflowError, match=f"'a.next/out'.*{beyond}"):
            through_the_core(spec, beyond)

    @pytest.mark.parametrize(
        "value", [0.0, -0.0, 1.5, -2.25e300, 5e-324, float("inf"), float("-inf")]
    )
    def test_a_float_round_trips_bit_for_bit(self, value: float):
        back = through_the_core(FloatSpec(), value)
        assert type(back) is float
        assert struct.pack("<d", back) == struct.pack("<d", value)

    def test_an_int_takes_a_bool_as_0_or_1(self):
        assert [through_the_core(IntSpec(8), b) for b in (False, True)] == [0, 1]

    def test_a_float_too_large_overflows_naming_the_slot(self):
        with pytest.raises(OverflowError, match="'a.next/out'.*float64"):
            through_the_core(FloatSpec(), 2**2000)

    def test_an_overflow_in_a_dict_names_the_field(self):
        msg = {**mixed_spec().zeros(), "jumps": 2**63}
        with pytest.raises(OverflowError, match="'a.next/out', field 'jumps'.*int64"):
            through_the_core(mixed_spec(), msg)

    def test_a_float_takes_an_int(self):
        back = through_the_core(FloatSpec(), 3)
        assert back == 3.0 and type(back) is float

    @pytest.mark.parametrize("value", [False, True])
    def test_a_bool_round_trips(self, value: bool):
        assert through_the_core(BoolSpec(), value) is value

    @pytest.mark.parametrize("ns", [0, 1, 2**64 - 1])
    def test_a_timestamp_round_trips(self, ns: int):
        back = through_the_core(TimestampSpec(), Timestamp(ns=ns))
        assert isinstance(back, Timestamp) and back.ns == ns

    def test_a_dict_mixes_tensors_and_scalars(self):
        msg = {
            "h": torch.tensor([1.0, 2.0]),
            "jumps": -3,
            "gain": 0.5,
            "on": True,
            "time": Timestamp(ns=7),
        }
        back = through_the_core(mixed_spec(), msg)
        assert isinstance(back, dict)
        assert list(back) == ["h", "jumps", "gain", "on", "time"]
        assert torch.equal(back["h"], msg["h"])
        assert [back[k] for k in ("jumps", "gain", "on", "time")] == [
            -3,
            0.5,
            True,
            Timestamp(ns=7),
        ]

    @pytest.mark.parametrize(
        ("spec", "zero"),
        [
            (IntSpec(8), 0),
            (FloatSpec(), 0.0),
            (BoolSpec(), False),
            (TimestampSpec(), Timestamp(ns=0)),
        ],
    )
    def test_the_initial_value_is_zero(self, spec: MessageSpec[Any], zero: object):
        initial = read_initial(spec)
        assert initial == zero and type(initial) is type(zero)

    def test_the_initial_dict_is_its_zeros(self):
        initial = read_initial(mixed_spec())
        zeros = mixed_spec().zeros()
        assert isinstance(initial, dict)
        assert torch.equal(initial["h"], zeros["h"])
        assert {k: v for k, v in initial.items() if k != "h"} == {
            k: v for k, v in zeros.items() if k != "h"
        }


class TestStructure:
    @pytest.mark.parametrize(
        ("spec", "msg", "message"),
        [
            (TensorSpec((2,)), [0.0, 0.0], "a.next: expected a torch.Tensor, got list"),
            (
                TensorSpec((2,)),
                {"error": torch.zeros(2)},
                r"a.next: expected a torch.Tensor, got a dict with the keys \['error'\]",
            ),
            (
                feedback_spec(),
                torch.zeros(2),
                r"a.next: expected a dict with the keys \['error', 'target'\], got "
                "Tensor",
            ),
            (
                feedback_spec(),
                {"error": torch.zeros(2, 3)},
                r"a.next: expected a dict with the keys \['error', 'target'\], got a "
                r"dict "
                r"with the keys \['error'\]",
            ),
            (
                feedback_spec(),
                {"error": torch.zeros(2, 3), "target": 1.0},
                r"a.next\['target'\]: expected a torch.Tensor, got float",
            ),
            (IntSpec(), torch.tensor(1), "a.next: expected an int, got Tensor"),
            (IntSpec(), 1.0, "a.next: expected an int, got float"),
            (FloatSpec(), "1", "a.next: expected a float, got str"),
            (BoolSpec(), 1, "a.next: expected a bool, got int"),
            (TimestampSpec(), 5, "a.next: expected a Timestamp, got int"),
            (
                mixed_spec(),
                {**mixed_spec().zeros(), "jumps": torch.tensor(0)},
                r"a.next\['jumps'\]: expected an int, got Tensor",
            ),
        ],
    )
    def test_names_the_message_and_what_it_got(
        self, spec: MessageSpec[Any], msg: object, message: str
    ):
        with pytest.raises(TypeError, match=f"^{message}$"):
            to_core(spec, msg, "a.next")

    def test_the_core_names_the_slot_and_field_of_a_wrong_shape_or_dtype(self):
        spec = feedback_spec()
        msg = {"error": torch.zeros(2, 3), "target": torch.zeros(3)}
        with pytest.raises(ValueError, match=r"'a\.next/out', field 'target'.*shape"):
            through_the_core(spec, msg)
        msg = {
            "error": torch.zeros(2, 3, dtype=torch.float64),
            "target": torch.zeros(2),
        }
        with pytest.raises(ValueError, match=r"'a\.next/out', field 'error'.*dtype"):
            through_the_core(spec, msg)
