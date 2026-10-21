from collections.abc import Sequence
from typing import TypedDict

import pytest
import torch

from mujou._core._native import Outlet, ReadOnlyTimer, VirtualTimer
from mujou._core.codec import fields, from_core, schema, to_core
from mujou._core.spec import DLPACK, DictSpec, Spec, TensorSpec

CPU = torch.device("cpu")


class Feedback(TypedDict):
    error: torch.Tensor
    target: torch.Tensor


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


def through_the_core(spec: Spec, msg: object, label: str = "a.next") -> object:
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
        assert not y.requires_grad


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
        ],
    )
    def test_names_the_message_and_what_it_got(
        self, spec: Spec, msg: object, message: str
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
