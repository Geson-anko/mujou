import threading
from collections.abc import Callable, Sequence
from types import CapsuleType

import pytest
import torch

from mujou._core._native import (
    GlobalTimer,
    InputGroup,
    Message,
    Outlet,
    Schema,
    Timer,
    Timestamp,
    timer as global_timer,
)

# The DLPack dtype `(code, bits, lanes)` that `Schema` takes.
type DLDataType = tuple[int, int, int]

# The DLPack dtypes of the supported torch dtypes. `mujou._spec` owns this table;
# it is repeated here so that the core is tested on its own.
DTYPES: dict[torch.dtype, DLDataType] = {
    torch.float16: (2, 16, 1),
    torch.bfloat16: (4, 16, 1),
    torch.float32: (2, 32, 1),
    torch.float64: (2, 64, 1),
    torch.int8: (0, 8, 1),
    torch.int16: (0, 16, 1),
    torch.int32: (0, 32, 1),
    torch.int64: (0, 64, 1),
    torch.uint8: (1, 8, 1),
    torch.bool: (6, 8, 1),
}


def schema(*fields: tuple[str, Sequence[int], torch.dtype]) -> Schema:
    return Schema([(name, shape, DTYPES[dtype]) for name, shape, dtype in fields])


def tensor_schema(shape: Sequence[int], dtype: torch.dtype = torch.float32) -> Schema:
    return schema(("", shape, dtype))


def values(msg: Message) -> list[torch.Tensor]:
    return [torch.from_dlpack(view).clone() for view in msg.buffers]


def bits(t: torch.Tensor) -> torch.Tensor:
    """The raw bytes, so that comparisons see NaN payloads and the sign of zero."""
    return t.reshape(-1).view(torch.uint8)


def sample(shape: Sequence[int], dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    """Values spread over the whole range of `dtype`, with the special floats."""
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.bool:
        return torch.randint(0, 2, tuple(shape), generator=g).bool()
    if dtype.is_floating_point:
        x = torch.randn(tuple(shape), generator=g, dtype=torch.float64) * 1e3
        x = x.to(dtype).reshape(-1)
        specials = torch.tensor([float("nan"), float("inf"), -0.0], dtype=dtype)
        n = min(len(specials), x.numel())
        x[:n] = specials[:n]
        return x.reshape(tuple(shape))
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min, info.max, tuple(shape), generator=g, dtype=torch.int64
    ).to(dtype)


class TestSchema:
    @pytest.mark.parametrize(
        "fields",
        [
            [],
            [("a", (2,), (2, 32, 1)), ("a", (3,), (2, 32, 1))],
            [("a", (2, 0), (2, 32, 1))],
            [("a", (2,), (2, 32, 4))],
            [("a", (2**61,), (2, 32, 1))],
            [("a", (2**32, 2**32), (2, 32, 1))],
        ],
        ids=["empty", "duplicate", "zero-extent", "vector-dtype", "huge", "overflow"],
    )
    def test_rejects_invalid_fields(
        self, fields: list[tuple[str, tuple[int, ...], DLDataType]]
    ):
        with pytest.raises(ValueError, match="invalid schema"):
            Schema(fields)


class TestOutlet:
    @pytest.mark.parametrize("dtype", list(DTYPES), ids=str)
    def test_round_trips_every_dtype_bit_for_bit(self, dtype: torch.dtype):
        x = sample((3, 5), dtype)
        outlet = Outlet(tensor_schema((3, 5), dtype), 1, "n.p", timer=global_timer)
        outlet.send([x])
        (y,) = values(outlet.outbox.read())
        assert y.dtype == dtype
        assert y.shape == x.shape
        assert torch.equal(bits(y), bits(x))

    def test_round_trips_zero_dimensional(self):
        outlet = Outlet(tensor_schema(()), 0, "p", timer=global_timer)
        outlet.send([torch.tensor(2.5)])
        (y,) = values(outlet.outbox.read())
        assert y.shape == ()
        assert y.item() == 2.5

    def test_round_trips_fields_in_key_order(self):
        s = schema(("error", (2,), torch.float64), ("mask", (3,), torch.bool))
        outlet = Outlet(s, 0, "p", timer=global_timer)
        error = torch.tensor([1.5, -2.0], dtype=torch.float64)
        mask = torch.tensor([True, False, True])
        outlet.send([error, mask])
        got = values(outlet.outbox.read())
        assert torch.equal(got[0], error)
        assert torch.equal(got[1], mask)

    @pytest.mark.parametrize(
        "make",
        [
            lambda x: x.T,
            lambda x: x[:, ::2],
            lambda x: x[1:],
            lambda x: x[0].expand(4, 6),
            lambda x: x[:, :1].expand(4, 6),
        ],
        ids=[
            "transposed",
            "sliced",
            "storage-offset",
            "broadcast-rows",
            "broadcast-cols",
        ],
    )
    def test_copies_non_contiguous_tensors(
        self, make: Callable[[torch.Tensor], torch.Tensor]
    ):
        x = make(torch.arange(24, dtype=torch.float32).reshape(4, 6))
        outlet = Outlet(tensor_schema(x.shape), 0, "p", timer=global_timer)
        outlet.send([x])
        (y,) = values(outlet.outbox.read())
        assert torch.equal(y, x.contiguous())

    def test_held_message_is_immutable(self):
        outlet = Outlet(tensor_schema((8,), torch.int64), 1, "p", timer=global_timer)
        outlet.send([torch.full((8,), 0)])
        held = outlet.channel(0).read()
        view = torch.from_dlpack(held.buffers[0])
        for k in range(1, 101):
            outlet.send([torch.full((8,), k)])
        assert torch.equal(view, torch.full((8,), 0))
        assert torch.equal(values(held)[0], torch.full((8,), 0))
        assert torch.equal(values(outlet.channel(0).read())[0], torch.full((8,), 100))

    def test_without_channels_writes_only_the_outbox(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p", timer=global_timer)
        outlet.send([torch.tensor([3.0])])
        assert torch.equal(values(outlet.outbox.read())[0], torch.tensor([3.0]))
        with pytest.raises(IndexError):
            outlet.channel(0)

    def test_accepts_producers_without_max_version(self):
        class OldProducer:
            def __init__(self, t: torch.Tensor) -> None:
                self._t = t

            def __dlpack__(self) -> CapsuleType:
                return self._t.__dlpack__()

        outlet = Outlet(tensor_schema((3,)), 0, "p", timer=global_timer)
        outlet.send([OldProducer(torch.tensor([1.0, 2.0, 3.0]))])
        assert torch.equal(
            values(outlet.outbox.read())[0], torch.tensor([1.0, 2.0, 3.0])
        )

    def test_lends_unversioned_capsules_to_old_consumers(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p", timer=global_timer)
        outlet.send([torch.tensor([1.0, 2.0])])
        (view,) = outlet.outbox.read().buffers
        y = torch.from_dlpack(view.__dlpack__()).clone()
        assert torch.equal(y, torch.tensor([1.0, 2.0]))
        assert view.__dlpack_device__() == (1, 0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"copy": True}, {"dl_device": (2, 0)}],
        ids=["copy", "cuda-device"],
    )
    def test_view_refuses_what_it_cannot_lend(self, kwargs: dict[str, object]):
        (view,) = (
            Outlet(tensor_schema((2,)), 0, "p", timer=global_timer)
            .outbox.read()
            .buffers
        )
        with pytest.raises(BufferError):
            view.__dlpack__(**kwargs)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (torch.zeros(3, 2), "shape (2, 3), got (3, 2)"),
            (torch.zeros(2, 3, dtype=torch.float64), "dtype float32, got float64"),
        ],
        ids=["shape", "dtype"],
    )
    def test_rejects_mismatch_naming_slot_and_field(
        self, value: torch.Tensor, expected: str
    ):
        s = schema(("x", (2, 3), torch.float32), ("y", (), torch.int64))
        outlet = Outlet(s, 1, "n.p", timer=global_timer)
        with pytest.raises(ValueError, match="field") as exc:
            outlet.send([value, torch.tensor(0)])
        assert "slot 'n.p/out'" in str(exc.value)
        assert "field 'x'" in str(exc.value)
        assert expected in str(exc.value)
        assert torch.equal(values(outlet.outbox.read())[0], torch.zeros(2, 3))

    def test_rejects_wrong_number_of_values(self):
        outlet = Outlet(
            schema(("x", (2,), torch.float32), ("y", (2,), torch.float32)),
            0,
            "p",
            timer=global_timer,
        )
        with pytest.raises(ValueError, match="expected 2 tensors") as exc:
            outlet.send([torch.zeros(2)])
        assert "slot 'p/out'" in str(exc.value)
        assert "'x', 'y'" in str(exc.value)

    def test_rejects_values_without_dlpack_naming_slot_and_field(self):
        outlet = Outlet(schema(("x", (2,), torch.float32)), 0, "p", timer=global_timer)
        with pytest.raises(TypeError, match="DLPack") as exc:
            outlet.send([[1.0, 2.0]])  # pyright: ignore[reportArgumentType]
        assert "slot 'p/out', field 'x'" in str(exc.value)

    def test_producer_errors_name_the_slot(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p", timer=global_timer)
        with pytest.raises(BufferError) as exc:
            outlet.send([torch.zeros(2, requires_grad=True)])
        assert any("slot 'p/out'" in note for note in exc.value.__notes__)

    def test_rejects_a_bare_tensor_for_the_sequence(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p", timer=global_timer)
        with pytest.raises(TypeError, match="sequence"):
            outlet.send(torch.zeros(2))  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("index", [3, -1])
    def test_rejects_unknown_channel(self, index: int):
        outlet = Outlet(tensor_schema((2,)), 3, "p", timer=global_timer)
        with pytest.raises(IndexError):
            outlet.send_to(index, [torch.zeros(2)])
        with pytest.raises(IndexError):
            outlet.channel(index)


class TestSlotWaitNewer:
    def test_returns_once_another_thread_writes(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p", timer=global_timer)
        last = outlet.outbox.read()
        started = threading.Event()

        def write() -> None:
            started.wait()
            outlet.send([torch.tensor([7.0])])

        writer = threading.Thread(target=write)
        writer.start()
        # With the GIL (the CI job on the default build), the writer can only run
        # while this thread waits if the wait drops it. The free-threaded build
        # checks nothing about the GIL here.
        started.set()
        msg = outlet.outbox.wait_newer(last, 10.0)
        writer.join()
        assert msg is not None
        assert torch.equal(values(msg)[0], torch.tensor([7.0]))

    def test_rejects_negative_timeout(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p", timer=global_timer)
        with pytest.raises(ValueError, match="timeout"):
            outlet.outbox.wait_newer(outlet.outbox.read(), -1.0)


class TestInputGroup:
    def test_invalid_item_changes_no_slot(self):
        group = InputGroup(timer=global_timer)
        x = group.input(tensor_schema((2,)), "x")
        y = group.input(tensor_schema((2,)), "y")
        with pytest.raises(ValueError, match="slot 'y/in'"):
            group.write([(x, [torch.ones(2)]), (y, [torch.ones(3)])])
        assert torch.equal(values(x.read())[0], torch.zeros(2))
        assert torch.equal(values(y.read())[0], torch.zeros(2))

    def test_rejects_slot_given_twice(self):
        group = InputGroup(timer=global_timer)
        x = group.input(tensor_schema((2,)), "x")
        with pytest.raises(ValueError, match="more than once"):
            group.write([(x, [torch.ones(2)]), (x, [torch.zeros(2)])])
        assert torch.equal(values(x.read())[0], torch.zeros(2))

    def test_rejects_slot_of_another_group(self):
        other = InputGroup(timer=global_timer).input(tensor_schema((2,)), "x")
        with pytest.raises(ValueError, match="another input group"):
            InputGroup(timer=global_timer).write([(other, [torch.ones(2)])])


class TestTimers:
    @pytest.mark.parametrize("own", [False, True], ids=["global", "own"])
    def test_slots_stamp_with_the_timer_they_are_given(
        self, virtual_timer: GlobalTimer, own: bool
    ):
        other = Timer()
        other.use_virtual_clock(True)
        other.load_state_dict({"time_ns": 0, "scale": 1.0})
        given, other = (other, virtual_timer) if own else (virtual_timer, other)
        other.advance(5.0)  # in no message
        given.advance(1.5)
        outlet = Outlet(tensor_schema((1,)), 1, "p", timer=given)
        group = InputGroup(timer=given)
        x = group.input(tensor_schema((1,)), "x")
        at_start = Timestamp(ns=1_500_000_000)
        assert outlet.channel(0).read().timestamp == x.read().timestamp == at_start

        given.advance(0.5)
        outlet.send([torch.ones(1)])
        written_at = group.write([(x, [torch.ones(1)])])
        sent = outlet.outbox.read().timestamp
        assert sent == x.read().timestamp == written_at == Timestamp(ns=2_000_000_000)
        assert float(sent) == 2.0
        assert at_start < sent
