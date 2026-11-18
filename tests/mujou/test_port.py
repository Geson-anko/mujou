import queue
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Never, TypedDict, cast, override

import pytest
import torch

from mujou.network import Group
from mujou.node import Node
from mujou.port import Channel, MultiPort, Port
from mujou.spec import DictSpec, IntSpec, TensorSpec
from mujou.time import Timestamp, VirtualTimer
from tests.mujou.nodes import Probe, probe_on

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual

H = 1e-3  # seconds per step
# The error of a port, or a node's timer, used outside its owner's hooks (K12).
OUTSIDE = "is used only in the hooks of its node while it runs"

type AnyPort = Port[Any, Any] | MultiPort[Any, Any]


class Feedback(TypedDict):
    error: torch.Tensor


class Other(TypedDict):
    error: torch.Tensor


class Tagged(TypedDict):
    x: torch.Tensor
    k: int


def h() -> TensorSpec:
    return TensorSpec((2, 3))


def fb(typed_dict: type[Feedback] = Feedback) -> DictSpec[Feedback]:
    return DictSpec(typed_dict, error=TensorSpec((2, 3)))


class TestConnect:
    def test_registers_the_channel_at_both_ends_in_order(self):
        a = MultiPort(send=h(), recv=fb())
        b = Port(send=fb(), recv=h())
        c = Port(send=fb(), recv=h())
        ab = a.connect(b)
        ca = c.connect(a)
        assert a.channels == (ab, ca)
        assert b.channels == (ab,)
        assert c.channels == (ca,)
        assert ab.other(a) is b and ab.other(b) is a
        assert ca.other(c) is a and ca.other(a) is c

    @pytest.mark.parametrize("make", [Port, MultiPort])
    def test_the_same_peer_again_gives_the_same_channel(self, make: type[AnyPort]):
        a, b = make(send=h(), recv=h()), make(send=h(), recv=h())
        ab = a.connect(b)
        assert a.connect(b) is ab
        assert b.connect(a) is ab
        assert a.channels == (ab,) and b.channels == (ab,)
        assert ab.other(a) is b
        # A port connects to itself once, too.
        loop = make(send=h(), recv=h())
        itself = loop.connect(loop)
        assert loop.connect(loop) is itself
        assert loop.channels == (itself,)
        assert itself.other(loop) is loop

    def test_one_way_ports_connect(self):
        sender = Port(send=h())
        receiver = Port(recv=h())
        ch = sender.connect(receiver)
        assert ch.other(sender) is receiver
        assert sender.recv_spec is None and receiver.send_spec is None

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # shape
            (Port(send=h()), Port(recv=TensorSpec((3, 2)))),
            # dtype
            (Port(send=h()), Port(recv=TensorSpec((2, 3), torch.float64))),
            # message type
            (Port(send=h()), Port(recv=fb())),
            # TypedDict type
            (Port(send=fb()), Port(recv=fb(Other))),  # pyright: ignore[reportArgumentType]
            # only one direction matches
            (Port(send=h(), recv=h()), Port(recv=h())),
            # same direction on both sides
            (Port(send=h()), Port(send=h())),
            # a MultiPort is checked alike
            (MultiPort(send=h()), Port(recv=fb())),
        ],
    )
    def test_rejects_specs_that_are_not_reversed(self, a: AnyPort, b: AnyPort):
        with pytest.raises(TypeError, match="do not match"):
            a.connect(b)
        assert a.channels == () and b.channels == ()

    @pytest.mark.parametrize("from_the_receiver", [True, False])
    def test_a_port_that_receives_takes_one_peer(self, from_the_receiver: bool):
        receiver = Port(recv=h())
        receiver.connect(Port(send=h()))
        other = MultiPort(send=h())
        # The message names the end that already has its peer.
        end = "this port" if from_the_receiver else "the other"
        with pytest.raises(
            ValueError,
            match=f"^{end} is a Port that receives and has a peer already; "
            "use a MultiPort to receive from several$",
        ):
            if from_the_receiver:
                receiver.connect(other)
            else:
                other.connect(receiver)
        assert len(receiver.channels) == 1 and other.channels == ()

    def test_a_port_that_only_sends_and_a_multi_port_take_any_number_of_peers(self):
        sender = Port(send=h())
        for _ in range(3):
            sender.connect(Port(recv=h()))
        merge = MultiPort(recv=h(), send=h())
        merge.connect(Port(send=h(), recv=h()))
        merge.connect(MultiPort(send=h(), recv=h()))
        assert len(sender.channels) == 3
        assert len(merge.channels) == 2


class TestNames:
    def test_ports_and_channels_are_shown_by_their_ends(self):
        a, b = Port(send=h()), MultiPort(recv=h())
        assert repr(a) == f"Port(send={h()!r}, recv=None)"
        assert repr(b) == f"MultiPort(send=None, recv={h()!r})"
        assert repr(a.connect(b)) == f"Channel({a!r} <-> {b!r})"

    def test_connect_errors_name_the_two_ends(self):
        a, b = Port(recv=h()), Port(recv=h())
        with pytest.raises(TypeError) as raised:
            a.connect(b)  # pyright: ignore[reportArgumentType]
        assert f"this port sends nothing, but the other receives {h()!r}" in str(
            raised.value
        )
        assert f"the other sends nothing, but this port receives {h()!r}" in str(
            raised.value
        )


def raises_outside(call: Callable[[], object]) -> RuntimeError | None:
    """What ``call`` raised if it was the error of K12, else ``None``."""
    try:
        call()
    except RuntimeError as e:
        return e if OUTSIDE in str(e) else None
    return None


class TestOutsideTheHooks:
    def test_before_a_run_naming_the_port(self):
        a = MultiPort(send=h(), recv=h())
        ch = a.connect(Port(send=h(), recv=h()))
        for call in (
            a.recv,
            a.recv_with_time,
            lambda: a.send(torch.zeros(2, 3)),
            lambda: a.send_to(ch, torch.zeros(2, 3)),
        ):
            error = raises_outside(call)
            assert error is not None
            assert str(error).startswith(f"{a!r} {OUTSIDE}")

    def test_from_the_host_while_it_runs_and_after_it(self, start_manual: StartManual):
        inp, out = Port(recv=h()), Port(send=h())
        run = start_manual(Group(node=Probe(inp=inp, out=out)))
        assert raises_outside(inp.recv) is not None
        assert raises_outside(lambda: out.send(torch.ones(2, 3))) is not None
        run.shutdown()
        assert raises_outside(lambda: out.send(torch.ones(2, 3))) is not None

    def test_from_another_node_and_for_a_port_made_in_a_hook(
        self, start_manual: StartManual
    ):
        theirs = Port(recv=h())
        owner = Probe(theirs=theirs)
        other = Probe(lambda: raises_outside(theirs.recv), own=Port(recv=h()))
        late = Probe(lambda: raises_outside(Port(send=h()).recv_with_time))
        run = start_manual(Group(owner=owner, other=other, late=late))
        run.step(H)
        assert isinstance(other.results[0], RuntimeError)
        assert isinstance(late.results[0], RuntimeError)


class Signalling(Probe):
    """A ``Probe`` that also signals each update it finished."""

    def __init__(
        self,
        update: Callable[[], object] = lambda: None,
        **ports: AnyPort,
    ) -> None:
        super().__init__(update, **ports)
        self.updated: queue.Queue[None] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.updated.put(None)


class TestOwnerOnAWorker:
    def test_only_the_owning_node_reads_and_writes_on_a_shared_worker(
        self, virtual: VirtualTimer, start: Start
    ):
        # specs: the owner is "the node the thread runs now", not the thread: one
        # worker runs all three nodes, one after another.
        inp, out = Port(recv=h()), Port(send=h())
        owner = Signalling(inp.recv, inp=inp, out=out)
        reader = Signalling(inp.recv, own=Port(recv=h()))
        writer = Signalling(lambda: out.send(torch.ones(2, 3)), own=Port(send=h()))
        run = start(
            Group(owner=owner, reader=reader, writer=writer), timer=virtual, workers=1
        )
        for _ in range(2):
            virtual.advance(0.01)
            for node in (owner, reader, writer):
                node.updated.get(timeout=5.0)
        run.shutdown()
        assert all(
            torch.equal(cast(torch.Tensor, r), torch.zeros(2, 3)) for r in owner.results
        )
        for node in (reader, writer):
            assert len(node.results) == 2
            for result in node.results:
                assert isinstance(result, RuntimeError)
                assert OUTSIDE in str(result)
        assert torch.equal(run.read(out), torch.zeros(2, 3))
        assert len(owner.results) == 2


class TestRecv:
    def test_unconnected_ports_give_zeros_or_nothing(self, start_manual: StartManual):
        single, multi = Port(recv=h()), MultiPort(recv=h())
        node = Probe(
            lambda: (
                single.recv(),
                multi.recv(),
                single.recv_with_time()[1],
                multi.recv_with_time(),
            ),
            single=single,
            multi=multi,
        )
        run = start_manual(Group(node=node))
        run.step(H)
        ((value, peers, time, timed_peers),) = cast(
            list[tuple[torch.Tensor, object, object, object]], node.results
        )
        assert torch.equal(value, torch.zeros(2, 3))
        assert peers == {} and timed_peers == {}
        assert time is None

    def test_many_peers_are_keyed_by_channel_in_connection_order(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        merge = MultiPort(recv=h())
        senders = [Port(send=h()) for _ in range(3)]
        channels = [s.connect(merge) for s in senders[::-1]]  # connect 2, 1, 0
        # Sender 0 stays quiet: its peer reads the zeros, with no time.
        nodes = {
            f"s{k}": Probe(
                lambda s=s, k=k: s.send(torch.full((2, 3), float(k))) if k else None,
                p=s,
            )
            for k, s in enumerate(senders)
        }
        sink = Probe(merge.recv_with_time, prev=merge)
        run = start_manual(Group(**nodes, sink=sink))
        run.step(H, *nodes.values(), sink)
        (received,) = sink.results
        received = cast(dict[Channel, tuple[torch.Tensor, Timestamp | None]], received)
        assert list(received) == channels == list(merge.channels)
        for channel, k in zip(channels, (2, 1, 0), strict=True):
            value, written_at = received[channel]
            assert torch.equal(value, torch.full((2, 3), float(k)))
            assert written_at == (virtual.now() if k else None)

    def test_gives_new_tensors_without_gradients(self, start_manual: StartManual):
        out, inp, other = Port(send=h()), Port(recv=h()), Port(recv=h())
        out.connect(inp)
        out.connect(other)
        sent = torch.ones(2, 3, requires_grad=True)
        source = Probe(lambda: out.send(sent), next=out)

        def change_what_was_received() -> tuple[torch.Tensor, torch.Tensor]:
            first = inp.recv()
            first.add_(10.0)  # in place: must not reach the slot
            return first, inp.recv()

        sink = Probe(change_what_was_received, prev=inp)
        neighbour = Probe(other.recv, prev=other)
        run = start_manual(Group(source=source, sink=sink, nb=neighbour))
        run.step(H, source, sink, neighbour)
        ((first, again),) = cast(list[tuple[torch.Tensor, torch.Tensor]], sink.results)
        assert not first.requires_grad and not again.requires_grad
        assert first.device == torch.device("cpu")
        assert torch.equal(again, torch.ones(2, 3))
        assert torch.equal(cast(torch.Tensor, neighbour.results[0]), torch.ones(2, 3))
        assert torch.equal(run.read(out), torch.ones(2, 3))

    def test_a_port_connected_to_itself_receives_what_it_sent(
        self, start_manual: StartManual
    ):
        loop = Port(send=h(), recv=h())
        loop.connect(loop)

        def count() -> torch.Tensor:
            x = loop.recv()
            loop.send(x + 1)
            return x

        node = Probe(count, loop=loop)
        run = start_manual(Group(node=node))
        for _ in range(3):
            run.step(H)
        firsts = [float(cast(torch.Tensor, x)[0, 0]) for x in node.results]
        assert firsts == [0.0, 1.0, 2.0]


def written(port: Port[Any, torch.Tensor]) -> bool:
    """Whether the writer of ``port`` has written what ``recv`` returns."""
    _, time = port.recv_with_time()
    return time is not None


class TestNoTimeUntilWritten:
    def test_the_initial_value_has_no_time_until_sent(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        out = Port(send=h(), initial=torch.ones(2, 3))
        inp = Port(recv=h())
        out.connect(inp)
        source = Probe(lambda: out.send(torch.zeros(2, 3)), next=out)
        sink = Probe(inp.recv_with_time, prev=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, sink)
        run.step(H, source, sink)  # sends the zeros: a value equal to the initial one
        (before, after) = cast(
            list[tuple[torch.Tensor, Timestamp | None]], sink.results
        )
        assert torch.equal(before[0], torch.ones(2, 3)) and before[1] is None
        assert torch.equal(after[0], torch.zeros(2, 3))
        assert after[1] == virtual.now()

    def test_what_the_host_writes_has_its_time(self, start_manual: StartManual):
        given, later = Port(recv=h()), Port(recv=h())
        node = Probe(
            lambda: (given.recv_with_time()[1], later.recv_with_time()[1]),
            given=given,
            later=later,
        )
        run = start_manual(Group(node=node), inputs={given: torch.ones(2, 3)})
        run.step(H)
        written_at = run.write({later: torch.ones(2, 3)})
        run.step(H)
        start = Timestamp(ns=0)
        assert node.results == [(start, None), (start, written_at)]

    @pytest.mark.parametrize("with_inputs", [False, True])
    def test_a_restored_run_keeps_what_was_not_written(
        self, start_manual: StartManual, with_inputs: bool
    ):
        def build(initial: float) -> tuple[Group, Probe, Probe, Port[Never, Any]]:
            out = MultiPort(send=h(), initial=torch.full((2, 3), initial))
            a, b = Port(recv=h()), Port(recv=h())
            to_a = out.connect(a)
            out.connect(b)
            host = Port(recv=h())
            source = Probe(lambda: out.send_to(to_a, torch.full((2, 3), 5.0)), next=out)
            sink = Probe(
                lambda: (a.recv_with_time(), b.recv_with_time(), written(host)),
                a=a,
                b=b,
                host=host,
            )
            return Group(source=source, sink=sink), source, sink, host

        net, source, _, _ = build(2.0)
        run = start_manual(net)
        run.step(H, source)  # sends to a only; nothing is written to host
        tree = run.state_dict()
        run.shutdown()

        # Another initial value: the restored run takes the values of the state.
        restored, _, sink, host = build(7.0)
        inputs = {host: torch.ones(2, 3)} if with_inputs else None
        run = start_manual(restored, state=tree, inputs=inputs)
        run.step(H, sink)
        ((at_a, at_b, host_written),) = cast(
            list[tuple[tuple[torch.Tensor, object], tuple[torch.Tensor, object], bool]],
            sink.results,
        )
        assert torch.equal(at_a[0], torch.full((2, 3), 5.0)) and at_a[1] is not None
        assert torch.equal(at_b[0], torch.full((2, 3), 2.0)) and at_b[1] is None
        assert host_written == with_inputs


class TestSend:
    @pytest.mark.parametrize("make", [Port, MultiPort])
    def test_copies_the_value_to_every_peer(
        self, start_manual: StartManual, make: type[AnyPort]
    ):
        out = make(send=h())
        sinks = {f"s{k}": Port(recv=h()) for k in range(3)}
        for inp in sinks.values():
            out.connect(inp)
        nodes = {name: Probe(inp.recv, prev=inp) for name, inp in sinks.items()}
        value = torch.ones(2, 3)

        def send_then_change() -> None:
            out.send(value)
            value.add_(1.0)  # after the send: must not reach the message

        source = Probe(send_then_change, next=out)
        run = start_manual(Group(source=source, **nodes))
        run.step(H, source, *nodes.values())  # the sinks read what source just sent
        for node in nodes.values():
            assert torch.equal(cast(torch.Tensor, node.results[0]), torch.ones(2, 3))
        assert torch.equal(run.read(out), torch.ones(2, 3))

    def test_rejects_a_tensor_that_is_not_dense(self, start_manual: StartManual):
        out = Port(send=TensorSpec((2, 2)))
        node = Probe(lambda: out.send(torch.eye(2).to_sparse()), out=out)
        run = start_manual(Group(node=node))
        run.step(H)
        assert isinstance(node.results[0], BufferError)
        assert torch.equal(run.read(out), torch.zeros(2, 2))

    def test_to_a_channel_of_another_port(self, start_manual: StartManual):
        out, inp = MultiPort(send=h()), Port(recv=h())
        foreign = Port(send=h()).connect(Port(recv=h()))
        out.connect(inp)
        node = Probe(lambda: out.send_to(foreign, torch.ones(2, 3)), next=out)
        run = start_manual(Group(node=node, sink=Probe(prev=inp)))
        run.step(H)
        assert isinstance(node.results[0], KeyError)


class TestInitial:
    def test_every_peer_and_the_host_read_it_until_a_send_to_them(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        twos = torch.full((2, 3), 2.0)
        out = MultiPort(send=h(), initial=twos)
        a, b = Port(recv=h()), Port(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        source = Probe(lambda: out.send_to(to_a, torch.ones(2, 3)), next=out)
        sink_a, sink_b = (
            Probe(a.recv_with_time, prev=a),
            Probe(b.recv_with_time, prev=b),
        )
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        run.step(H, sink_a, sink_b)  # the source does not send
        assert torch.equal(run.read(out), twos)
        run.step(H, source, sink_a, sink_b)  # send_to replaces it for a only
        results = cast(
            list[tuple[torch.Tensor, Timestamp | None]], sink_a.results + sink_b.results
        )
        expected = [(twos, None), (torch.ones(2, 3), virtual.now())] + [
            (twos, None)
        ] * 2
        for (value, written_at), (want, want_at) in zip(results, expected, strict=True):
            assert torch.equal(value, want)
            assert written_at == want_at  # None as the zeros: not sent by the port
        # The output the host reads changes only with send to every peer.
        assert torch.equal(run.read(out), twos)

    def test_of_a_dict_message(self, start_manual: StartManual):
        out = Port(send=fb(), initial={"error": torch.ones(2, 3)})
        inp = Port(recv=fb())
        out.connect(inp)
        sink = Probe(inp.recv, inp=inp)
        run = start_manual(Group(source=Probe(out=out), sink=sink))
        run.step(H, sink)
        received = cast(Feedback, sink.results[0])
        assert torch.equal(received["error"], torch.ones(2, 3))

    def test_is_copied_when_the_run_starts(self, start_manual: StartManual):
        value = torch.zeros(2, 3, requires_grad=True)
        out = Port(send=h(), initial=value)
        with torch.no_grad():
            value.add_(1.0)  # before the start: the run reads it
        run = start_manual(Group(node=Probe(out=out)))
        with torch.no_grad():
            value.add_(1.0)  # after the start: must not reach the slots
        received = run.read(out)
        assert torch.equal(received, torch.ones(2, 3))
        assert not received.requires_grad

    @pytest.mark.parametrize(
        ("spec", "value", "error", "match"),
        [
            (h(), [[0.0] * 3] * 2, TypeError, r"^node\.next: expected a torch\.Tensor"),
            (
                fb(),
                {"loss": torch.zeros(2, 3)},
                TypeError,
                r"^node\.next: expected a dict with",
            ),
            # The core checks the shape and the dtype when the run writes the value
            # to the port's slots, before any node starts.
            (
                h(),
                torch.zeros(3, 2),
                ValueError,
                r"^slot 'node\.next/out': expected shape",
            ),
            (
                h(),
                torch.zeros(2, 3, dtype=torch.float64),
                ValueError,
                r"^slot 'node\.next/out': expected dtype",
            ),
        ],
    )
    def test_a_run_rejects_a_value_that_does_not_match_by_its_path(
        self,
        start_manual: StartManual,
        spec: TensorSpec | DictSpec[Feedback],
        value: object,
        error: type[Exception],
        match: str,
    ):
        node = Probe(next=Port(send=spec, initial=value))  # pyright: ignore[reportArgumentType, reportUnknownArgumentType]
        with pytest.raises(error, match=match):
            start_manual(Group(node=node))


# For each argv[i] "<source>><sink>", steps a node on <source> (or the host,
# "host:<device>") that sends to a node on <sink> under torch.profiler, and
# prints "<case>|<name>" for each copy on a GPU. "<device>@<k>" sends while
# cuda:<k> is current.
_PROFILED = """
import sys
from typing import override

import torch
from torch.profiler import ProfilerActivity, profile

import mujou
from mujou import Group, Node, Port, TensorSpec

SPEC = TensorSpec((1024,))


class Source(Node):
    def __init__(self, device: str, current: str) -> None:
        super().__init__(device=device)
        self.out = Port(send=SPEC)
        self.current = current

    @override
    def on_update(self, dt: float) -> None:
        x = torch.ones(1024, device=self.device)
        if self.current:
            with torch.cuda.device(int(self.current)):
                self.out.send(x)
        else:
            self.out.send(x)


class Sink(Node):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.inp = Port(recv=SPEC)

    @override
    def on_update(self, dt: float) -> None:
        self.inp.recv()


def copies(source: str, on: str) -> list[str]:
    source, _, current = source.partition("@")
    sink = Sink(on)
    if source.startswith("host:"):
        device = source.removeprefix("host:")
        run = mujou.run_manually(sink)

        def step() -> None:
            run.write({sink.inp: torch.ones(1024, device=device)})
            run.step(0.001, sink)

    else:
        node = Source(source, current)
        node.out.connect(sink.inp)
        run = mujou.run_manually(Group(source=node, sink=sink))

        def step() -> None:
            run.step(0.001, node, sink)

    step()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
    run.shutdown()
    return [e.name for e in prof.events() if e.name.startswith("Memcpy")]


for case in sys.argv[1:]:
    for name in copies(*case.split(">")):
        print(case, name, sep="|")
"""


def profiled_copies(cases: list[str]) -> dict[str, list[str]]:
    """The copies on a GPU of each case (as ``_PROFILED`` takes them), in a new
    interpreter: the profiler loads modules that enable the GIL."""
    done = subprocess.run(
        [sys.executable, "-c", _PROFILED, *cases],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    found: dict[str, list[str]] = {case: [] for case in cases}
    for line in done.stdout.splitlines():
        case, _, name = line.partition("|")
        found[case].append(name)
    return found


def via_host_memory(copies: list[str]) -> bool:
    """Whether a copy from or to host memory is among ``copies``."""
    assert copies, "the profiler saw no copy"
    # "Memcpy HtoD (Pageable -> Device)": the ends are in the parentheses. A
    # copy between GPUs without peer access is "Memcpy DtoH (Device ->
    # Device)", which the driver may stage through the host by itself.
    ends = {end for name in copies for end in re.findall(r"\w+", name.split("(")[-1])}
    return bool(ends & {"Pageable", "Pinned"})


# (source, sink, whether the message goes through host memory, CUDA devices).
HOST_MEMORY_CASES = [
    ("cuda:0", "cuda:0", False, 1),
    ("cuda:0", "cuda:1", False, 2),
    # The control: the oracle sees a copy from host memory.
    ("cpu", "cuda:0", True, 1),
    # Read by nodes on the CPU only: sent to host memory once.
    ("cuda:0", "cpu", True, 1),
    # Sent while the hook has made another device current.
    ("cuda:0@1", "cuda:0", False, 2),
    # The host writes to a node on the device of the tensor, or not. (A copy
    # to a node on the CPU shows either way, where the host or the node makes
    # it: test_execution.py checks who waits.)
    ("host:cuda:0", "cuda:0", False, 1),
    ("host:cuda:0", "cpu", True, 1),
    ("host:cuda:1", "cuda:0", True, 2),
]


BIG = TensorSpec((1 << 20,))


def delay_the_stream() -> None:
    """Keep the current stream busy for about 0.1 s of GPU time, so that what
    is queued after it is still pending when the CPU moves on. ``_sleep`` is a
    private API of torch, which its own tests use for this."""
    torch.cuda._sleep(250_000_000)  # pyright: ignore[reportPrivateUsage]


class OnCuda0(Node):
    """A node on ``device`` (``cuda:0`` by default) on a thread of its own that,
    in ``on_start``, runs each kind of kernel the tests run on a CUDA device:
    CUDA loads the code of a kernel when it first runs, and waits for the whole
    device to load it, which would hide a missing wait for a stream."""

    def __init__(self, device: str = "cuda:0") -> None:
        super().__init__(device=device, dedicated_thread=True)

    @override
    def on_start(self) -> None:
        if self.device.type != "cuda":
            return
        x = torch.empty(BIG.shape, device=self.device)
        x.fill_(1.0)
        x.clone().to(self.device, copy=True).cpu()
        torch.cuda._sleep(1)  # pyright: ignore[reportPrivateUsage]
        torch.cuda.current_stream().synchronize()


class Late(OnCuda0):
    """Sends ``7 + current`` (7 without ``current``) that its stream fills
    late, once, and then sets ``sent``; sends while device ``current`` is
    current, if given. A value of each case of its own: memory of an earlier
    case may still hold another."""

    def __init__(self, current: int | None = None) -> None:
        super().__init__()
        self.out = Port(send=BIG)
        self.sent = threading.Event()
        self.current = current

    @override
    def on_update(self, dt: float) -> None:
        if self.sent.is_set():
            return
        x = torch.empty(BIG.shape, device=self.device)
        delay_the_stream()
        x.fill_(7.0 + (self.current or 0))
        if self.current is None:
            self.out.send(x)
        else:
            with torch.cuda.device(self.current):
                self.out.send(x)
        self.sent.set()


class Resends(OnCuda0):
    """Sends 1s and sets ``first``, then, once ``read`` is set, sends 2s, 3s and
    4s and sets ``again``: once."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=BIG)
        self.first, self.read, self.again = (threading.Event() for _ in range(3))

    @override
    def on_update(self, dt: float) -> None:
        if self.first.is_set():
            return
        self.out.send(torch.ones(BIG.shape, device=self.device))
        self.first.set()
        assert self.read.wait(10.0)
        for k in (2.0, 3.0, 4.0):
            self.out.send(torch.full(BIG.shape, k, device=self.device))
        self.again.set()


class Reads(OnCuda0):
    """Once ``ready`` is set: calls ``before``, receives one message, calls
    ``between`` and keeps the message on the CPU in ``received``. Once."""

    def __init__(
        self, ready: threading.Event, before: Callable[[], object] = lambda: None
    ) -> None:
        super().__init__()
        self.inp = Port(recv=BIG)
        self.ready = ready
        self.before = before
        self.between: Callable[[], object] = lambda: None
        self.received: torch.Tensor | None = None
        self.done = threading.Event()

    @override
    def on_update(self, dt: float) -> None:
        if self.done.is_set():
            return
        assert self.ready.wait(10.0)
        self.before()
        received = self.inp.recv()
        self.between()
        self.received = received.cpu()
        self.done.set()


class SendsThenWorks(OnCuda0):
    """Sends 1s, then queues long work on its stream, records ``busy`` after
    it and sets ``sent``: once."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=BIG)
        self.sent = threading.Event()
        self.busy: torch.cuda.Event | torch.Event | None = None

    @override
    def on_update(self, dt: float) -> None:
        if self.sent.is_set():
            return
        self.out.send(torch.ones(BIG.shape, device=self.device))
        delay_the_stream()
        self.busy = torch.cuda.current_stream().record_event()
        self.sent.set()


class ReadsAndCheck(OnCuda0):
    """Once ``writer`` has sent: receives, waits for its own stream, and keeps
    in ``writer_done`` whether the writer's work after the send had finished
    by then. Once."""

    def __init__(self, device: str, writer: SendsThenWorks) -> None:
        super().__init__(device)
        self.inp = Port(recv=BIG)
        # Not the node itself: a node holds no nodes.
        self.sent = writer.sent
        self.busy = lambda: cast("torch.cuda.Event | torch.Event", writer.busy)
        self.writer_done: bool | None = None
        self.received: torch.Tensor | None = None
        self.done = threading.Event()

    @override
    def on_update(self, dt: float) -> None:
        if self.done.is_set():
            return
        assert self.sent.wait(10.0)
        received = self.inp.recv()
        if received.is_cuda:
            torch.cuda.current_stream().synchronize()
        self.writer_done = self.busy().query()
        self.received = received.cpu()
        self.done.set()


@pytest.mark.cuda
class TestCuda:
    def test_receives_on_the_device_and_sends_to_the_host(
        self, start_manual: StartManual
    ):
        out, inp, back = Port(send=h()), Port(recv=h()), Port(send=h())
        out.connect(inp)

        def update() -> torch.Tensor:
            x = inp.recv()
            back.send(2 * x)
            return x

        source = Probe(lambda: out.send(torch.ones(2, 3)), next=out)

        class OnCuda(Probe):
            on_device = "cuda"

        sink = OnCuda(update, prev=inp, back=back)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        # The index-less device is fixed, so the message compares equal to it.
        assert received.device == sink.device == torch.device("cuda", 0)
        assert torch.equal(received.cpu(), torch.ones(2, 3))
        sent = run.read(back)
        assert sent.device == torch.device("cpu")
        assert torch.equal(sent, torch.full((2, 3), 2.0))

    @pytest.mark.parametrize(
        ("reader", "with_changes"),
        [
            ("cuda:0", True),
            pytest.param("cuda:1", True, marks=pytest.mark.cuda(2)),
            ("cpu", True),
            ("cpu", False),
        ],
        ids=["cuda:0", "cuda:1", "cpu-and-cuda:0", "cpu-only"],
    )
    def test_a_message_of_a_cuda_node_arrives_on_the_readers_device(
        self, start_manual: StartManual, reader: str, with_changes: bool
    ):
        # Another reader on cuda:0 changes its message in place before the
        # reader reads: each reader has a message of its own (and with the
        # reader on the CPU, the core holds the message on the device). Read
        # only on the CPU, the message goes through the host memory.
        out, changed, inp = Port(send=h()), Port(recv=h()), Port(recv=h())
        if with_changes:
            out.connect(changed)
        out.connect(inp)
        x = torch.arange(6.0).reshape(3, 2).t()  # (2, 3), not contiguous

        def send() -> None:
            w = torch.ones((), device="cuda:0", requires_grad=True)
            sent = x.to("cuda:0") * w
            out.send(sent)
            with torch.no_grad():
                sent.add_(1.0)  # after the send: must not reach the message

        source = probe_on("cuda:0", send, out=out)
        sink = probe_on(reader, inp.recv, inp=inp)
        if with_changes:
            changes = probe_on("cuda:0", lambda: changed.recv().add_(1.0), inp=changed)
            run = start_manual(Group(source=source, changes=changes, sink=sink))
            run.step(H, source, changes, sink)
            assert torch.equal(cast(torch.Tensor, changes.results[0]).cpu(), x + 1)
        else:
            run = start_manual(Group(source=source, sink=sink))
            run.step(H, source, sink)
        received = cast(torch.Tensor, sink.results[0])
        assert received.device == torch.device(reader)
        assert received.is_contiguous() and not received.requires_grad
        assert torch.equal(received.cpu(), x)
        assert torch.equal(run.read(out), x)

    def test_send_to_an_initial_value_and_a_dict_on_the_device(
        self, start_manual: StartManual
    ):
        twos = torch.full((2, 3), 2.0)
        out = MultiPort(send=h(), initial=twos.to("cuda:0"))
        a, b = Port(recv=h()), Port(recv=h())
        to_a = out.connect(a)
        out.connect(b)
        spec = DictSpec(Tagged, x=h(), k=IntSpec())
        tagged, tagged_in = Port(send=spec), Port(recv=spec)
        tagged.connect(tagged_in)

        def send() -> None:
            ones = torch.ones(2, 3, device="cuda:0")
            out.send_to(to_a, ones)
            tagged.send({"x": ones, "k": 5})

        source = probe_on("cuda:0", send, out=out, tagged=tagged)
        sink_a = probe_on(
            "cuda:0", lambda: (a.recv(), tagged_in.recv()), inp=a, tagged=tagged_in
        )
        # On the CPU: the initial value on the device reaches it there.
        sink_b = Probe(b.recv, inp=b)
        run = start_manual(Group(source=source, a=sink_a, b=sink_b))
        run.step(H, source, sink_a, sink_b)
        at_a, at_tagged = cast(tuple[torch.Tensor, Tagged], sink_a.results[0])
        assert at_a.device == torch.device("cuda", 0)
        assert torch.equal(at_a.cpu(), torch.ones(2, 3))
        assert at_tagged["k"] == 5
        assert at_tagged["x"].device == torch.device("cuda", 0)
        assert torch.equal(at_tagged["x"].cpu(), torch.ones(2, 3))
        at_b = cast(torch.Tensor, sink_b.results[0])
        assert at_b.device == torch.device("cpu")
        assert torch.equal(at_b, twos)
        assert torch.equal(run.read(out), twos)

    def test_rejects_a_tensor_that_is_not_dense(self, start_manual: StartManual):
        out = Port(send=TensorSpec((2, 2)))
        sparse = torch.eye(2, device="cuda:0").to_sparse()
        node = probe_on("cuda:0", lambda: out.send(sparse), out=out)
        run = start_manual(Group(node=node))
        run.step(H)
        assert isinstance(node.results[0], BufferError)

    @pytest.mark.cuda(2)
    def test_a_tensor_on_another_device_arrives(self, start_manual: StartManual):
        out, inp = Port(send=h()), Port(recv=h())
        out.connect(inp)
        x = torch.ones(2, 3, device="cuda:1")
        source = probe_on("cuda:0", lambda: out.send(x), out=out)
        sink = probe_on("cuda:0", inp.recv, inp=inp)
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]).cpu(), x.cpu())

    @pytest.mark.slow
    def test_a_message_skips_host_memory_only_from_a_cuda_device_to_a_node_on_it(
        self,
    ):
        # All cases in one interpreter: one CUDA context instead of one each.
        count = torch.cuda.device_count()
        expected = {
            f"{source}>{sink}": via
            for source, sink, via, needs in HOST_MEMORY_CASES
            if needs <= count
        }
        copies = profiled_copies(list(expected))
        assert {case: via_host_memory(copies[case]) for case in expected} == expected

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "current", [None, pytest.param(1, marks=pytest.mark.cuda(2))]
    )
    def test_a_reader_gets_what_a_late_stream_wrote(
        self, virtual: VirtualTimer, start: Start, current: int | None
    ):
        # The reader copies the message the moment the writer has sent it,
        # while the writer is still in its on_update: send waits for the stream,
        # the node's, also while the hook has made another device current.
        writer = Late(current)
        reader = Reads(writer.sent)
        writer.out.connect(reader.inp)
        start(Group(writer=writer, reader=reader), timer=virtual)
        virtual.advance(0.01)
        assert reader.done.wait(10.0)
        assert torch.equal(
            cast(torch.Tensor, reader.received),
            torch.full(BIG.shape, 7.0 + (current or 0)),
        )

    @pytest.mark.slow
    def test_a_copy_still_queued_keeps_its_message_from_reuse(
        self, virtual: VirtualTimer, start: Start
    ):
        # The reader's copy waits on its stream while the writer replaces the
        # message: the memory of the 1s goes back to the caching allocator, and
        # the next message of its size would take it, before the copy has run.
        writer = Resends()
        reader = Reads(writer.first, before=delay_the_stream)
        reader.between = lambda: (writer.read.set(), writer.again.wait(10.0))
        writer.out.connect(reader.inp)
        start(Group(writer=writer, reader=reader), timer=virtual)
        virtual.advance(0.01)
        assert reader.done.wait(10.0)
        assert torch.equal(cast(torch.Tensor, reader.received), torch.ones(BIG.shape))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "reader", ["cuda:0", pytest.param("cuda:1", marks=pytest.mark.cuda(2)), "cpu"]
    )
    def test_a_reader_does_not_wait_for_what_the_writer_queues_after_sending(
        self, virtual: VirtualTimer, start: Start, reader: str
    ):
        # S2: the writer waits for its stream before it publishes, so the
        # reader's copy depends on nothing of the writer's stream.
        writer = SendsThenWorks()
        sink = ReadsAndCheck(reader, writer)
        writer.out.connect(sink.inp)
        # A reader on CUDA too, so that the core holds the message on the
        # device for a reader on the CPU as well.
        on_cuda = Port(recv=BIG)
        writer.out.connect(on_cuda)
        other = probe_on("cuda:0", inp=on_cuda)
        start(Group(writer=writer, sink=sink, other=other), timer=virtual)
        virtual.advance(0.01)
        assert sink.done.wait(10.0)
        assert sink.writer_done is False
        assert torch.equal(cast(torch.Tensor, sink.received), torch.ones(BIG.shape))
