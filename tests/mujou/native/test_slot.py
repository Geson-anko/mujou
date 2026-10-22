import threading
from collections.abc import Callable, Sequence

import pytest
import torch

from mujou._native import (
    InputGroup,
    Message,
    Outlet,
    ReadOnlyTimer,
    Schema,
    Timestamp,
    VirtualTimer,
    global_timer,
)
from mujou.spec import DLPACK

# The clock of a run on the global timer, for the slots of the tests that do not
# look at the time.
CLOCK = ReadOnlyTimer(global_timer)

# The DLPack dtype `(code, bits, lanes)` that `Schema` takes.
type DLDataType = tuple[int, int, int]


def schema(*fields: tuple[str, Sequence[int], torch.dtype]) -> Schema:
    return Schema([(name, shape, DLPACK[dtype]) for name, shape, dtype in fields])


def tensor_schema(shape: Sequence[int], dtype: torch.dtype = torch.float32) -> Schema:
    return schema(("", shape, dtype))


def values(msg: Message) -> list[torch.Tensor]:
    return [torch.from_dlpack(view).clone() for view in msg.values]


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
    @pytest.mark.parametrize("dtype", list(DLPACK), ids=str)
    def test_round_trips_every_dtype_bit_for_bit(self, dtype: torch.dtype):
        x = sample((3, 5), dtype)
        outlet = Outlet(tensor_schema((3, 5), dtype), 1, "n.p", clock=CLOCK)
        outlet.send([x])
        (y,) = values(outlet.outbox.read())
        assert y.dtype == dtype
        assert y.shape == x.shape
        assert torch.equal(bits(y), bits(x))

    def test_round_trips_zero_dimensional(self):
        outlet = Outlet(tensor_schema(()), 0, "p", clock=CLOCK)
        outlet.send([torch.tensor(2.5)])
        (y,) = values(outlet.outbox.read())
        assert y.shape == ()
        assert y.item() == 2.5

    def test_round_trips_fields_in_key_order(self):
        s = schema(("error", (2,), torch.float64), ("mask", (3,), torch.bool))
        outlet = Outlet(s, 0, "p", clock=CLOCK)
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
        outlet = Outlet(tensor_schema(x.shape), 0, "p", clock=CLOCK)
        outlet.send([x])
        (y,) = values(outlet.outbox.read())
        assert torch.equal(y, x.contiguous())

    def test_held_message_is_immutable(self):
        outlet = Outlet(tensor_schema((8,), torch.int64), 1, "p", clock=CLOCK)
        outlet.send([torch.full((8,), 0)])
        held = outlet.channel(0).read()
        view = torch.from_dlpack(held.values[0])
        for k in range(1, 101):
            outlet.send([torch.full((8,), k)])
        assert torch.equal(view, torch.full((8,), 0))
        assert torch.equal(values(held)[0], torch.full((8,), 0))
        assert torch.equal(values(outlet.channel(0).read())[0], torch.full((8,), 100))

    def test_without_channels_writes_only_the_outbox(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p", clock=CLOCK)
        outlet.send([torch.tensor([3.0])])
        assert torch.equal(values(outlet.outbox.read())[0], torch.tensor([3.0]))
        with pytest.raises(IndexError):
            outlet.channel(0)

    def test_lends_only_over_dlpack_1(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p", clock=CLOCK)
        outlet.send([torch.tensor([1.0, 2.0])])
        (view,) = outlet.outbox.read().values
        with pytest.raises(BufferError, match="max_version"):
            view.__dlpack__()
        assert view.__dlpack_device__() == (1, 0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"copy": True}, {"dl_device": (2, 0)}],
        ids=["copy", "cuda-device"],
    )
    def test_view_refuses_what_it_cannot_lend(self, kwargs: dict[str, object]):
        (view,) = Outlet(tensor_schema((2,)), 0, "p", clock=CLOCK).outbox.read().values
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
        outlet = Outlet(s, 1, "n.p", clock=CLOCK)
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
            clock=CLOCK,
        )
        with pytest.raises(ValueError, match=r"^slot 'p/out': 1 values for 2 fields"):
            outlet.send([torch.zeros(2)])

    def test_rejects_values_without_dlpack_naming_slot_and_field(self):
        outlet = Outlet(schema(("x", (2,), torch.float32)), 0, "p", clock=CLOCK)
        with pytest.raises(TypeError, match="DLPack") as exc:
            outlet.send([[1.0, 2.0]])  # pyright: ignore[reportArgumentType]
        assert "slot 'p/out', field 'x'" in str(exc.value)

    def test_reads_a_python_scalar_only_into_a_scalar_dtype(self):
        # A float32 of shape () is a tensor field: no Python scalar is read as one.
        outlet = Outlet(schema(("x", (), torch.float32)), 0, "p", clock=CLOCK)
        with pytest.raises(TypeError, match="DLPack") as exc:
            outlet.send([1.0])
        assert "slot 'p/out', field 'x'" in str(exc.value)

    def test_names_the_slot_of_a_value_that_is_no_scalar(self):
        outlet = Outlet(schema(("n", (), torch.int64)), 0, "p", clock=CLOCK)
        with pytest.raises(TypeError) as exc:
            outlet.send(["1"])
        assert any("slot 'p/out', field 'n'" in note for note in exc.value.__notes__)

    def test_item_reads_a_scalar_field_only(self):
        outlet = Outlet(schema(("x", (2,), torch.int64)), 0, "p", clock=CLOCK)
        (view,) = outlet.outbox.read().values
        with pytest.raises(TypeError, match=r"shape \(2,\) and dtype int64"):
            view.item()

    def test_producer_errors_name_the_slot(self):
        outlet = Outlet(tensor_schema((2,)), 0, "p", clock=CLOCK)
        with pytest.raises(BufferError) as exc:
            outlet.send([torch.zeros(2, requires_grad=True)])
        assert any("slot 'p/out'" in note for note in exc.value.__notes__)

    @pytest.mark.parametrize("index", [3, -1])
    def test_rejects_unknown_channel(self, index: int):
        outlet = Outlet(tensor_schema((2,)), 3, "p", clock=CLOCK)
        with pytest.raises(IndexError):
            outlet.send_to(index, [torch.zeros(2)])
        with pytest.raises(IndexError):
            outlet.channel(index)


class TestSlotWaitNewer:
    def test_returns_once_another_thread_writes(self):
        outlet = Outlet(tensor_schema((1,)), 0, "p", clock=CLOCK)
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
        outlet = Outlet(tensor_schema((1,)), 0, "p", clock=CLOCK)
        with pytest.raises(ValueError, match="duration"):
            outlet.outbox.wait_newer(outlet.outbox.read(), -1.0)


class TestInputGroup:
    def test_invalid_item_changes_no_slot(self):
        group = InputGroup(clock=CLOCK)
        x = group.input(tensor_schema((2,)), "x")
        y = group.input(tensor_schema((2,)), "y")
        with pytest.raises(ValueError, match="slot 'y/in'"):
            group.write([(x, [torch.ones(2)]), (y, [torch.ones(3)])])
        assert torch.equal(values(x.read())[0], torch.zeros(2))
        assert torch.equal(values(y.read())[0], torch.zeros(2))


class TestTime:
    def test_only_what_the_writers_wrote_has_a_time(self):
        outlet = Outlet(tensor_schema((2,)), 1, "p", clock=CLOCK)
        group = InputGroup(clock=CLOCK)
        x = group.input(tensor_schema((2,)), "x")
        assert outlet.outbox.read().timestamp is None
        assert outlet.channel(0).read().timestamp is None
        assert x.read().timestamp is None
        outlet.send([torch.ones(2)])
        written_at = group.write([(x, [torch.ones(2)])])
        assert outlet.outbox.read().timestamp == outlet.channel(0).read().timestamp
        assert outlet.outbox.read().timestamp is not None
        assert x.read().timestamp == written_at

    @pytest.mark.parametrize("time_ns", [None, 5])
    def test_restore_sets_it(self, time_ns: int | None):
        outlet = Outlet(tensor_schema((2,)), 0, "p", clock=CLOCK)
        x = InputGroup(clock=CLOCK).input(tensor_schema((2,)), "x")
        outlet.outbox.restore([torch.ones(2)], time_ns)
        x.restore([torch.ones(2)], time_ns)
        expected = None if time_ns is None else Timestamp(ns=time_ns)
        assert outlet.outbox.read().timestamp == expected
        assert x.read().timestamp == expected


class TestClocks:
    def test_slots_stamp_with_the_clock_they_are_given(self):
        given, other = VirtualTimer(), VirtualTimer()
        other.advance(5.0)  # in no message
        given.advance(1.5)
        clock = ReadOnlyTimer(given)
        outlet = Outlet(tensor_schema((1,)), 1, "p", clock=clock)
        group = InputGroup(clock=clock)
        x = group.input(tensor_schema((1,)), "x")
        outlet.send([torch.ones(1)])
        written_at = group.write([(x, [torch.ones(1)])])
        sent = outlet.outbox.read().timestamp
        assert sent == x.read().timestamp == written_at == Timestamp(ns=1_500_000_000)
