import threading
from collections.abc import Callable, Sequence
from types import CapsuleType

import pytest
import torch

from mujou._core import (
    GlobalTimer,
    InputGroup,
    Message,
    Outlet,
    Schema,
)

# `mujou._core.DLDataType` exists only in the stub.
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
        ],
        ids=["empty", "duplicate", "zero-extent", "vector-dtype"],
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
        outlet = Outlet(tensor_schema((3, 5), dtype), 1, "n.p")
        outlet.send([x])
        (y,) = values(outlet.outbox.read())
        assert y.dtype == dtype
        assert y.shape == x.shape
        assert torch.equal(bits(y), bits(x))

    def test_round_trips_zero_dimensional(self):
        outlet = Outlet(tensor_schema(()), 0, "p")
        outlet.send([torch.tensor(2.5)])
        (y,) = values(outlet.outbox.read())
        assert y.shape == ()
        assert y.item() == 2.5

    def test_round_trips_fields_in_key_order(self):
        s = schema(("error", (2,), torch.float64), ("mask", (3,), torch.bool))
        outlet = Outlet(s, 0, "p")
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
            lambda x: x.unsqueeze(0).expand(2, -1, -1)[1],
        ],
        ids=["transposed", "sliced", "storage-offset", "expanded"],
    )
    def test_copies_non_contiguous_tensors(
        self, make: Callable[[torch.Tensor], torch.Tensor]
    ):
        x = make(torch.arange(24, dtype=torch.float32).reshape(4, 6))
        outlet = Outlet(tensor_schema(x.shape), 0, "p")
        outlet.send([x])
        (y,) = values(outlet.outbox.read())
        assert torch.equal(y, x.contiguous())

    def test_writer_changing_its_tensor_does_not_change_the_slot(self):
        x = torch.zeros(4)
        outlet = Outlet(tensor_schema((4,)), 0, "p")
        outlet.send([x])
        x.fill_(1.0)
        (y,) = values(outlet.outbox.read())
        assert torch.equal(y, torch.zeros(4))

    def test_reader_changing_its_tensor_does_not_change_other_reads(self):
        outlet = Outlet(tensor_schema((4,)), 0, "p")
        outlet.send([torch.ones(4)])
        msg = outlet.outbox.read()
        (first,) = values(msg)
        first.fill_(9.0)
        assert torch.equal(values(msg)[0], torch.ones(4))
        assert torch.equal(values(outlet.outbox.read())[0], torch.ones(4))

    def test_held_message_is_immutable(self):
        outlet = Outlet(tensor_schema((8,), torch.int64), 1, "p")
        outlet.send([torch.full((8,), 0)])
        held = outlet.channel(0).read()
        view = torch.from_dlpack(held.buffers[0])
        for k in range(1, 101):
            outlet.send([torch.full((8,), k)])
        assert torch.equal(view, torch.full((8,), 0))
        assert torch.equal(values(held)[0], torch.full((8,), 0))
        assert torch.equal(values(outlet.channel(0).read())[0], torch.full((8,), 100))

    def test_send_reaches_the_outbox_and_every_channel(
        self, virtual_timer: GlobalTimer
    ):
        outlet = Outlet(tensor_schema((2,)), 3, "p")
        virtual_timer.advance(1.5)
        outlet.send([torch.tensor([1.0, 2.0])])
        slots = [outlet.outbox] + [outlet.channel(k) for k in range(3)]
        for slot in slots:
            msg = slot.read()
            assert msg.time_ns == 1_500_000_000
            assert torch.equal(values(msg)[0], torch.tensor([1.0, 2.0]))

    def test_send_to_changes_only_that_channel(self, virtual_timer: GlobalTimer):
        outlet = Outlet(tensor_schema((2,)), 3, "p")
        virtual_timer.advance(1.0)
        outlet.send_to(1, [torch.ones(2)])
        assert outlet.channel(1).read().time_ns == 1_000_000_000
        assert torch.equal(values(outlet.channel(1).read())[0], torch.ones(2))
        for slot in [outlet.outbox, outlet.channel(0), outlet.channel(2)]:
            msg = slot.read()
            assert msg.time_ns == 0
            assert torch.equal(values(msg)[0], torch.zeros(2))

    def test_starts_with_zeros_and_names_its_slots(self, virtual_timer: GlobalTimer):
        virtual_timer.advance(0.25)
        outlet = Outlet(
            schema(("a", (2,), torch.int32), ("b", (), torch.bool)), 2, "x.p"
        )
        assert outlet.outbox.name == "x.p/out"
        assert [outlet.channel(k).name for k in range(2)] == ["x.p/ch0", "x.p/ch1"]
        msg = outlet.channel(1).read()
        assert msg.time_ns == 250_000_000
        a, b = values(msg)
        assert torch.equal(a, torch.zeros(2, dtype=torch.int32))
        assert torch.equal(b, torch.tensor(False))

    def test_without_channels_writes_only_the_outbox(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p")
        outlet.send([torch.tensor([3.0])])
        assert torch.equal(values(outlet.outbox.read())[0], torch.tensor([3.0]))
        with pytest.raises(IndexError):
            outlet.channel(0)

    def test_restore_sets_value_and_time(self):
        outlet = Outlet(tensor_schema((2,)), 1, "p")
        outlet.channel(0).restore([torch.tensor([4.0, 5.0])], 123)
        msg = outlet.channel(0).read()
        assert msg.time_ns == 123
        assert torch.equal(values(msg)[0], torch.tensor([4.0, 5.0]))

    def test_accepts_producers_without_max_version(self):
        class OldProducer:
            def __init__(self, t: torch.Tensor) -> None:
                self._t = t

            def __dlpack__(self) -> CapsuleType:
                return self._t.__dlpack__()

        outlet = Outlet(tensor_schema((3,)), 0, "p")
        outlet.send([OldProducer(torch.tensor([1.0, 2.0, 3.0]))])
        assert torch.equal(
            values(outlet.outbox.read())[0], torch.tensor([1.0, 2.0, 3.0])
        )

    def test_lends_unversioned_capsules_to_old_consumers(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p")
        outlet.send([torch.tensor([1.0, 2.0])])
        (view,) = outlet.outbox.read().buffers
        y = torch.from_dlpack(view.__dlpack__()).clone()
        assert torch.equal(y, torch.tensor([1.0, 2.0]))
        assert view.__dlpack_device__() == (1, 0)

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
        outlet = Outlet(s, 1, "n.p")
        with pytest.raises(ValueError, match="field") as exc:
            outlet.send([value, torch.tensor(0)])
        assert "slot 'n.p/out'" in str(exc.value)
        assert "field 'x'" in str(exc.value)
        assert expected in str(exc.value)
        assert torch.equal(values(outlet.outbox.read())[0], torch.zeros(2, 3))

    def test_rejects_wrong_number_of_values(self):
        outlet = Outlet(
            schema(("x", (2,), torch.float32), ("y", (2,), torch.float32)), 0, "p"
        )
        with pytest.raises(ValueError, match="field count") as exc:
            outlet.send([torch.zeros(2)])
        assert "'x', 'y'" in str(exc.value)

    def test_rejects_values_without_dlpack(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p")
        with pytest.raises(TypeError, match="DLPack"):
            outlet.send([[1.0, 2.0]])  # pyright: ignore[reportArgumentType]

    def test_rejects_a_bare_tensor_for_the_sequence(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p")
        with pytest.raises(TypeError, match="sequence"):
            outlet.send(torch.zeros(2))  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize("index", [3, -1])
    def test_rejects_unknown_channel(self, index: int):
        outlet = Outlet(tensor_schema((2,)), 3, "p")
        with pytest.raises(IndexError):
            outlet.send_to(index, [torch.zeros(2)])
        with pytest.raises(IndexError):
            outlet.channel(index)


class TestSlotWaitNewer:
    def test_returns_current_without_last(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p")
        msg = outlet.outbox.wait_newer(None, 0.0)
        assert msg is not None
        assert torch.equal(values(msg)[0], torch.zeros(1))

    def test_times_out_without_a_write(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p")
        last = outlet.outbox.read()
        assert outlet.outbox.wait_newer(last, 0.02) is None

    def test_returns_once_another_thread_writes(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p")
        last = outlet.outbox.read()
        started = threading.Event()

        def write() -> None:
            started.wait()
            outlet.send([torch.tensor([7.0])])

        writer = threading.Thread(target=write)
        writer.start()
        # The writer can only run while this thread waits if the wait drops the GIL.
        started.set()
        msg = outlet.outbox.wait_newer(last, 10.0)
        writer.join()
        assert msg is not None
        assert torch.equal(values(msg)[0], torch.tensor([7.0]))

    def test_rejects_negative_timeout(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p")
        with pytest.raises(ValueError, match="timeout"):
            outlet.outbox.wait_newer(outlet.outbox.read(), -1.0)


class TestInputGroup:
    def test_write_stamps_every_slot_with_one_time(self, virtual_timer: GlobalTimer):
        group = InputGroup()
        x = group.input(tensor_schema((2,)), "net.x")
        y = group.input(schema(("t", (), torch.int64)), "net.y")
        assert x.name == "net.x/in"
        virtual_timer.advance(2.0)
        group.write([(x, [torch.tensor([1.0, 2.0])]), (y, [torch.tensor(3)])])
        mx, my = x.read(), y.read()
        assert mx.time_ns == my.time_ns == 2_000_000_000
        assert torch.equal(values(mx)[0], torch.tensor([1.0, 2.0]))
        assert torch.equal(values(my)[0], torch.tensor(3))

    def test_invalid_item_changes_no_slot(self):
        group = InputGroup()
        x = group.input(tensor_schema((2,)), "x")
        y = group.input(tensor_schema((2,)), "y")
        with pytest.raises(ValueError, match="slot 'y/in'"):
            group.write([(x, [torch.ones(2)]), (y, [torch.ones(3)])])
        assert torch.equal(values(x.read())[0], torch.zeros(2))
        assert torch.equal(values(y.read())[0], torch.zeros(2))

    def test_rejects_slot_of_another_group(self):
        other = InputGroup().input(tensor_schema((2,)), "x")
        with pytest.raises(ValueError, match="another input group"):
            InputGroup().write([(other, [torch.ones(2)])])

    def test_restore_sets_value_and_time(self):
        group = InputGroup()
        x = group.input(tensor_schema((2,)), "x")
        group.write([(x, [torch.ones(2)])])
        x.restore([torch.tensor([5.0, 6.0])], 42)
        msg = x.read()
        assert msg.time_ns == 42
        assert torch.equal(values(msg)[0], torch.tensor([5.0, 6.0]))
