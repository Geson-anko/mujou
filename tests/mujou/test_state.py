import io
import os
import pickle
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, Never, TypedDict, cast, override

import pytest
import torch
from torch import nn

import mujou
from mujou import (
    DictSpec,
    Group,
    IntSpec,
    MultiPort,
    NetworkError,
    Node,
    Port,
    Run,
    TensorSpec,
    TimestampSpec,
)
from mujou.state import capture, cpu
from mujou.testing import ManualRun
from mujou.time import ReadOnlyTimer, Timestamp, VirtualTimer, global_timer
from mujou.tree import walk
from mujou.wiring import wire
from tests.mujou.nodes import (
    F64,
    Chain,
    Hidden,
    Leaky,
    Pair,
    Probe,
    assert_same_tree,
    entries,
    port_slots,
    probe_on,
    updates,
    vec,
)

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual

H = 1e-3


def u(k: float) -> torch.Tensor:
    return torch.tensor([k, -k], dtype=F64)


class Learned(Node):
    """Holds a module, a tensor and a registered count."""

    def __init__(self) -> None:
        super().__init__(seed=3)
        self.out = Port(send=vec())
        self.linear = nn.Linear(2, 2)
        self.x = torch.zeros(2, dtype=F64)
        self.count = 0
        self.register_state("count")


class Stamped(TypedDict):
    t: Timestamp
    n: int


class Stamper(Node):
    """Sends the time of its update and a count."""

    def __init__(self) -> None:
        super().__init__()
        self.out = Port(send=DictSpec(Stamped, t=TimestampSpec(), n=IntSpec()))

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.out.send({"t": self.timer.now(), "n": 3})


class TestShape:
    def test_holds_the_timer_and_each_node_by_its_path(self, start_manual: StartManual):
        net = Chain()
        run = start_manual(net, inputs={net.prev: u(1)})
        run.step(H)
        tree = run.state_dict()
        assert list(tree) == ["timer", "nodes"]
        assert tree["timer"] == {"time_ns": 1_000_000, "time_scale": 1.0}
        assert list(tree["nodes"]) == ["a", "b"]
        a = tree["nodes"]["a"]
        assert list(a) == ["state", "last_update_ns", "ports"]
        assert a["last_update_ns"] == 1_000_000
        assert a["state"]["updates"] == 1
        assert_same_tree(
            a["ports"],
            {
                "prev": {"in": {"value": u(1), "time_ns": 0}},
                "next": {
                    "out": {"value": net.a.x, "time_ns": 1_000_000},
                    "to": {"b.prev": {"value": net.a.x, "time_ns": 1_000_000}},
                },
            },
        )
        assert_same_tree(
            tree["nodes"]["b"]["ports"],
            {
                "prev": {},
                "next": {"out": {"value": net.b.x, "time_ns": 1_000_000}, "to": {}},
            },
        )

    def test_a_port_keys_its_slot_for_each_peer_by_the_peer_port(
        self, start_manual: StartManual
    ):
        source = Probe(out=Port(send=vec()), loop=MultiPort(send=vec(), recv=vec()))
        sinks = [Probe(inp=Port(recv=vec())) for _ in range(2)]
        for sink in sinks:
            source.out.connect(sink.inp)  # pyright: ignore[reportAttributeAccessIssue]
        source.loop.connect(source.loop)  # pyright: ignore[reportAttributeAccessIssue]
        net = Group(source=source, x=sinks[0], y=sinks[1], h=Hidden())
        tree = start_manual(net).state_dict()
        ports = tree["nodes"]["source"]["ports"]
        assert list(ports["out"]) == ["out", "to"]
        assert list(ports["out"]["to"]) == ["x.inp", "y.inp"]
        assert list(ports["loop"]["to"]) == ["source.loop"]
        # An unconnected Port that receives reads the host; a MultiPort does not.
        assert list(tree["nodes"]["h"]["ports"]["prev"]) == ["out", "to", "in"]
        assert list(tree["nodes"]["h"]["ports"]["next"]) == ["out", "to"]

    def test_a_slot_not_written_yet_has_no_time(self, start_manual: StartManual):
        tree = start_manual(Chain()).state_dict()
        assert tree["nodes"]["a"]["ports"]["prev"]["in"]["time_ns"] is None
        assert tree["nodes"]["b"]["ports"]["next"]["out"]["time_ns"] is None

    def test_send_to_writes_only_its_peer(self, start_manual: StartManual):
        spec = TensorSpec((2, 3))
        port = MultiPort(send=spec, recv=spec)
        b, c = Hidden(), Hidden()
        channel = port.connect(b.prev)
        port.connect(c.prev)
        node = Probe(lambda: port.send_to(channel, torch.ones(2, 3)), next=port)
        run = start_manual(Group(a=node, b=b, c=c))
        run.step(H, node)
        to = run.state_dict()["nodes"]["a"]["ports"]["next"]["to"]
        assert to["b.prev"]["time_ns"] == 1_000_000
        assert to["c.prev"]["time_ns"] is None

    def test_a_node_state_is_the_default_one(self, start_manual: StartManual):
        node = Learned()
        tree = start_manual(Group(node=node)).state_dict()
        state = tree["nodes"]["node"]["state"]
        assert list(state) == ["generator", "linear", "x", "count"]
        assert_same_tree(state["linear"], node.linear.state_dict())


class TestCopies:
    @pytest.mark.parametrize("ended", [False, True])
    def test_share_no_memory_with_the_run(self, start_manual: StartManual, ended: bool):
        node = Learned()
        run = start_manual(Group(node=node))
        if ended:
            run.shutdown()
        tree = run.state_dict()
        node.x += 1.0  # in place, as a node updates its state
        with torch.no_grad():
            node.linear.weight.add_(1.0)
        state = tree["nodes"]["node"]["state"]
        assert torch.equal(state["x"], torch.zeros(2, dtype=F64))
        assert not torch.equal(state["linear"]["weight"], node.linear.weight)

    @pytest.mark.parametrize("ended", [False, True])
    def test_hold_no_graph(self, start_manual: StartManual, ended: bool):
        class Derived(Node):
            """Returns a tensor that records its graph in its state."""

            def __init__(self) -> None:
                super().__init__()
                self.w = torch.ones(2, dtype=F64, requires_grad=True)

            @override
            def state_dict(self) -> dict[str, Any]:
                return {**super().state_dict(), "derived": [2 * self.w]}

        run = start_manual(Group(node=Derived()))
        if ended:
            run.shutdown()
        (derived,) = run.state_dict()["nodes"]["node"]["state"]["derived"]
        assert not derived.requires_grad and derived.grad_fn is None
        assert torch.equal(derived, torch.full((2,), 2.0, dtype=F64))

    def test_torch_load_reads_it_with_weights_only(self, start_manual: StartManual):
        node = Learned()
        tree = start_manual(Group(node=node)).state_dict()
        buffer = io.BytesIO()
        torch.save(tree, buffer)
        buffer.seek(0)
        assert_same_tree(torch.load(buffer, weights_only=True), tree)

    def test_holds_a_timestamp_of_a_message_as_its_ns(self, start_manual: StartManual):
        # As every time of the tree, so that weights_only reads it.
        run = start_manual(Group(node=Stamper()))
        run.step(H)
        tree = run.state_dict()
        out = port_slots(tree, "node", "out")["out"]
        assert out["value"] == {"t": 1_000_000, "n": 3}
        buffer = io.BytesIO()
        torch.save(tree, buffer)
        buffer.seek(0)
        assert_same_tree(torch.load(buffer, weights_only=True), tree)


class TestRestore:
    def test_goes_on_from_the_tree(self, start_manual: StartManual):
        first = Learned()
        run = start_manual(Group(node=first))
        first.count = 5
        # Moved on from the seed, which a new node starts from.
        torch.rand(3, generator=first.generator)
        run.step(H)
        tree = run.state_dict()
        run.shutdown()
        again = Learned()
        with torch.no_grad():
            again.linear.weight.fill_(7.0)
        restored = start_manual(Group(node=again), state=tree)
        assert again.count == 5
        assert torch.equal(again.linear.weight, first.linear.weight)
        # The generator goes on with the same sequence.
        assert torch.equal(
            torch.rand(3, generator=again.generator),
            torch.rand(3, generator=first.generator),
        )
        assert_same_tree(
            restored.state_dict()["nodes"]["node"]["ports"],
            tree["nodes"]["node"]["ports"],
        )

    def test_a_node_state_that_does_not_match_stops_the_run(
        self, start_manual: StartManual
    ):
        source = start_manual(Group(node=Learned()))
        tree = source.state_dict()
        source.shutdown()
        del tree["nodes"]["node"]["state"]["x"]
        with pytest.raises(
            NetworkError, match=r"node 'node' raised in load_state_dict"
        ) as raised:
            start_manual(Group(node=Learned()), state=tree)
        assert isinstance(raised.value.__cause__, ValueError)
        assert "missing ['x']" in str(raised.value.__cause__)

    def test_lists_what_does_not_match(self, start_manual: StartManual):
        source = start_manual(Chain())
        tree: dict[str, Any] = source.state_dict()
        source.shutdown()
        del tree["nodes"]["b"]
        del tree["nodes"]["a"]["ports"]["next"]["to"]["b.prev"]
        tree["nodes"]["a"]["ports"]["extra"] = {}
        net = Chain()
        with pytest.raises(ValueError) as raised:
            start_manual(net, state=tree)
        assert str(raised.value) == (
            "the state tree does not match this run: "
            "missing nodes['a']['ports']['next']['to']['b.prev'], "
            "unexpected nodes['a']['ports']['extra'], missing nodes['b']"
        )
        assert not net.a.started

    def test_lists_what_its_top_does_not_match(self, start_manual: StartManual):
        source = start_manual(Chain())
        tree: dict[str, Any] = source.state_dict()
        source.shutdown()
        del tree["timer"]["time_scale"]
        tree["extra"] = {}
        net = Chain()
        with pytest.raises(ValueError) as raised:
            start_manual(net, state=tree)
        assert str(raised.value) == (
            "the state tree does not match this run: "
            "missing timer['time_scale'], unexpected extra"
        )
        assert not net.a.started

    def test_restores_a_tree_of_the_same_node_class(self, start_manual: StartManual):
        first = Leaky()
        run = start_manual(first, inputs={first.prev: u(1)})
        run.step(H)
        tree = pickle.loads(pickle.dumps(run.state_dict()))  # saved and loaded
        run.shutdown()
        again = Leaky()
        restored = start_manual(again, state=tree)
        assert torch.equal(again.x, first.x) and again.updates == 1
        back = restored.state_dict()
        assert_same_tree(back, tree)

    def test_rejects_a_tree_of_another_network(self, start_manual: StartManual):
        source = start_manual(Chain())
        tree = source.state_dict()
        source.shutdown()
        other = Group(node=Pair())
        with pytest.raises(ValueError, match="^the state tree does not match this run"):
            start_manual(other, state=tree)

    def test_the_default_load_state_dict_copies_out_of_the_tree(
        self, start_manual: StartManual
    ):
        class Holds(Node):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.zeros(2, dtype=F64)

        run = start_manual(Group(node=Holds()))
        tree = run.state_dict()
        run.shutdown()
        restored = Holds()
        start_manual(Group(node=restored), state=tree)
        restored.x += 1.0  # in place, as a node updates its state
        assert torch.equal(
            tree["nodes"]["node"]["state"]["x"], torch.zeros(2, dtype=F64)
        )

    def test_restores_a_tree_saved_with_torch_save(self, start_manual: StartManual):
        source = Chain()
        run = start_manual(source, inputs={source.prev: u(1)})
        run.step(H)
        tree = run.state_dict()
        run.shutdown()
        buffer = io.BytesIO()
        torch.save(tree, buffer)
        buffer.seek(0)
        net = Chain()
        start_manual(net, state=torch.load(buffer, weights_only=True))
        assert torch.equal(net.a.x, tree["nodes"]["a"]["state"]["x"])
        assert torch.equal(net.b.x, tree["nodes"]["b"]["state"]["x"])

    @pytest.mark.parametrize("b_first", [False, True])
    def test_a_restored_network_continues_the_same(
        self, start_manual: StartManual, b_first: bool
    ):
        k, m = 5, 7
        net = Chain()
        run = start_manual(net)
        order = (net.b, net.a) if b_first else (net.a, net.b)
        for i in range(k):
            run.write({net.prev: u(i)})
            run.step(H, *order)
        run.step(H, net.a)  # b's last update is now one step behind
        run.write({net.prev: u(k)})  # the input slot is part of the state too
        tree = pickle.loads(pickle.dumps(run.state_dict()))
        for i in range(k, k + m):
            run.step(H, *order)
            run.write({net.prev: u(i + 1)})
        expected = (net.a.x.clone(), net.b.x.clone(), run.read_with_time(net.next))
        run.shutdown()

        restored = Chain()
        again_run = start_manual(restored, state=tree)
        # Right after the restore the state is the tree, including where each dt
        # starts and every channel slot (which Jacobi order reads first).
        assert_same_tree(again_run.state_dict(), tree)
        restored_order = (
            (restored.b, restored.a) if b_first else (restored.a, restored.b)
        )
        for i in range(k, k + m):
            again_run.step(H, *restored_order)
            again_run.write({restored.prev: u(i + 1)})
        assert torch.equal(restored.a.x, expected[0])
        assert torch.equal(restored.b.x, expected[1])
        out, t = again_run.read_with_time(restored.next)
        assert torch.equal(out, expected[2][0]) and t == expected[2][1]
        # The nodes got copies: their updates in place did not reach the tree.
        assert not torch.equal(tree["nodes"]["a"]["state"]["x"], restored.a.x)

    @pytest.mark.parametrize("b_first", [False, True])
    def test_a_network_restored_after_the_end_continues_the_same(
        self,
        virtual: VirtualTimer,
        start_manual: StartManual,
        b_first: bool,
    ):
        k, m, gap = 5, 7, 0.3

        def present(run: ManualRun, net: Chain, steps: range) -> None:
            order = (net.b, net.a) if b_first else (net.a, net.b)
            for i in steps:
                run.write({net.prev: u(i)})
                run.step(H, *order)
            run.step(H, net.a)  # b's last update is one step behind
            run.write({net.prev: u(steps.stop)})
            run.timer.advance(gap)  # the input stays presented a while

        # Saved after the end.
        ended = Chain()
        run = start_manual(ended)
        present(run, ended, range(k))
        run.shutdown()
        virtual.advance(100.0)  # time goes on after the end
        tree = pickle.loads(pickle.dumps(run.state_dict()))

        # The same run without stopping.
        through = Chain()
        run = start_manual(through)
        present(run, through, range(k))
        present(run, through, range(k + 1, k + 1 + m))
        expected = (through.a.x.clone(), through.b.x.clone())
        run.shutdown()

        restored = Chain()
        run = start_manual(restored, state=tree)
        present(run, restored, range(k + 1, k + 1 + m))
        assert torch.equal(restored.a.x, expected[0])
        assert torch.equal(restored.b.x, expected[1])


class TestStateDict:
    def test_after_a_normal_end_is_the_state_at_the_end(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        net = Chain()
        run = start_manual(net)
        for k in range(3):
            run.write({net.prev: u(k)})
            run.step(H)
        run.step(H, net.a)  # b's last update is one step behind
        run.write({net.prev: u(3)})
        running = run.state_dict()
        virtual.advance(0.3)  # the input stays presented for 0.3 s more
        virtual.set_time_scale(2.0)
        asked_at = virtual.now().ns
        run.shutdown()
        virtual.advance(100.0)  # time goes on after the end
        ended = run.state_dict()
        # Nothing changed after the last update: the same tree as just before the
        # end, at the time of the shutdown (the time after the end is not part of
        # the run).
        assert ended["timer"] == {"time_ns": asked_at, "time_scale": 2.0}
        assert_same_tree(ended["nodes"], running["nodes"])
        pickle.dumps(ended)

    def test_after_a_normal_end_of_a_free_running_network(self, start: Start):
        net = Chain(tau=0.01)
        run = start(net, inputs={net.prev: u(1)})
        # Free running: both nodes have updated once b sends a value other than 0.
        run.wait_for(net.next, lambda out: bool(out.any()), timeout=10.0)
        asked_after = global_timer.now().ns
        run.shutdown()
        stopped_before = global_timer.now().ns
        global_timer.sleep(0.05)  # time goes on after the end
        tree = run.state_dict()
        frozen_at = tree["timer"]["time_ns"]
        # As a pause at the request would: the time of the request, and nothing
        # in the tree is later than it.
        assert asked_after <= frozen_at <= stopped_before
        for updated_at in updates(tree).values():
            assert updated_at <= frozen_at
        for entry in entries(tree):
            assert entry["time_ns"] is not None and entry["time_ns"] <= frozen_at
        out = port_slots(tree, "b", "next")["out"]["value"]
        assert torch.equal(out, tree["nodes"]["b"]["state"]["x"])

    def test_is_taken_at_one_time(self, start: Start):
        net = Chain(tau=0.01)
        run = start(net, inputs={net.prev: u(1)})
        # Free running: both nodes have updated once b sends a value other than 0.
        run.wait_for(net.next, lambda out: bool(out.any()), timeout=10.0)
        tree = run.state_dict()
        frozen_at = tree["timer"]["time_ns"]
        assert list(tree["nodes"]) == ["a", "b"]
        assert len(entries(tree)) == 4  # a.prev in, a.next out and to b, b.next out
        for entry in entries(tree):
            assert entry["time_ns"] is not None and entry["time_ns"] <= frozen_at
        for updated_at in updates(tree).values():
            assert 0 < updated_at <= frozen_at
        # Each node sends at the end of its update, so a node paused between
        # updates has sent its state as it is.
        out = port_slots(tree, "b", "next")["out"]["value"]
        assert torch.equal(out, tree["nodes"]["b"]["state"]["x"])

    @pytest.mark.parametrize("ended", [False, True])
    def test_is_a_copy_that_pickles(self, start_manual: StartManual, ended: bool):
        net = Chain()
        run = start_manual(net)
        run.write({net.prev: u(1)})
        run.step(H)
        if ended:
            run.shutdown()
        tree = run.state_dict()
        x = net.a.x.clone()
        net.a.x += 1.0  # in place, as a node updates its state
        assert torch.equal(tree["nodes"]["a"]["state"]["x"], x)
        assert_same_tree(pickle.loads(pickle.dumps(tree)), tree)

    @pytest.mark.parametrize("ended", [False, True])
    def test_is_a_plain_dict(self, start_manual: StartManual, ended: bool):
        run = start_manual(Chain())
        if ended:
            run.shutdown()
        assert type(run.state_dict()) is dict

    @pytest.mark.parametrize("ended", [False, True])
    @pytest.mark.parametrize("use", ["recv", "send"])
    def test_the_ports_refuse_in_the_node_state_dict(
        self, start_manual: StartManual, ended: bool, use: str
    ):
        # Taking the state is not seen by the other nodes, nor by the host.
        class Using(Pair):
            @override
            def state_dict(self) -> dict[str, Any]:
                try:
                    if use == "recv":
                        self.x.recv()
                    else:
                        self.out.send(torch.ones(2, dtype=F64))
                except RuntimeError as e:
                    return {"error": str(e)}
                return {}

        node = Using()
        run = start_manual(Group(node=node))
        run.step(H)
        if ended:
            run.shutdown()
        state = run.state_dict()["nodes"]["node"]["state"]
        assert "a port is not used in state_dict" in state["error"]
        assert torch.equal(run.read(node.out), torch.zeros(2, dtype=F64))

    @pytest.mark.slow
    def test_two_threads_take_it_at_once_after_a_normal_end_with_the_gil(self):
        # With the GIL, a thread that waited for a node another thread captures
        # must not hold the GIL meanwhile. A short switch interval makes the
        # switch in the middle of state_dict likely.
        script = """
import sys, threading
sys.setswitchinterval(1e-6)
import mujou.testing
from mujou import Group, Node

class Busy(Node):
    def state_dict(self):
        s = 0
        for i in range(100_000):  # in Python, where the GIL may switch
            s += i
        return {"s": s}

for _ in range(5):
    run = mujou.testing.run(Group(a=Busy(), b=Busy()))
    run.step(0.001)
    run.shutdown()
    threads = [threading.Thread(target=run.state_dict) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
"""
        done = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHON_GIL": "1"},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert done.returncode == 0, done.stderr


class Blocks(Node):
    """Blocks in its first ``on_update`` until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        if not self.release.is_set():
            self.entered.set()
            self.release.wait(10.0)


class TestPauseTimeout:
    def test_state_dict_raises_when_a_node_does_not_pause_and_the_run_goes_on(
        self, started: list[Run]
    ):
        node = Blocks()
        run = mujou.run(node)
        started.append(run)
        try:
            assert node.entered.wait(10.0)
            with pytest.raises(TimeoutError):
                run.state_dict(timeout=0.05)
        finally:
            node.release.set()

        assert "nodes" in run.state_dict()  # the run kept running
        run.shutdown()
        run.state_dict()  # a normal end


class TestScalarMessages:
    def test_a_saved_timestamp_is_restored_as_a_timestamp(
        self, start_manual: StartManual
    ):
        run = start_manual(Group(node=Stamper()))
        run.step(H)
        buffer = io.BytesIO()
        torch.save(run.state_dict(), buffer)
        run.shutdown()
        buffer.seek(0)
        node = Stamper()
        restored = start_manual(
            Group(node=node), state=torch.load(buffer, weights_only=True)
        )
        assert restored.read(node.out) == {"t": Timestamp(ns=1_000_000), "n": 3}


def sends_on_cuda() -> tuple[Probe, Probe, Port[torch.Tensor, Never]]:
    """A node on ``cuda:0`` that sends ``[1, 2]``, and a node on ``cuda:0`` that
    receives it (so the core holds the message on the device); the port that
    sends."""
    out, inp = Port(send=vec()), Port(recv=vec())
    out.connect(inp)
    x = torch.tensor([1.0, 2.0], dtype=F64, device="cuda:0")
    source = probe_on("cuda:0", lambda: out.send(x), out=out)
    return source, probe_on("cuda:0", inp=inp), out


@pytest.mark.cuda
class TestCuda:
    @pytest.mark.parametrize("ended", [False, True])
    def test_a_port_value_on_the_device_is_saved_on_the_cpu_and_restored(
        self, start_manual: StartManual, ended: bool
    ):
        source, sink, out = sends_on_cuda()
        run = start_manual(Group(source=source, sink=sink))
        run.step(H, source)
        if ended:
            run.shutdown()
        tree = run.state_dict()
        expected = torch.tensor([1.0, 2.0], dtype=F64)
        for entry in entries(tree):
            assert entry["value"].device == torch.device("cpu")
        slots = port_slots(tree, "source", "out")
        for entry in (slots["out"], slots["to"]["sink.inp"]):
            assert torch.equal(entry["value"], expected)
        run.shutdown()

        source, sink, out = sends_on_cuda()
        restored = start_manual(Group(source=source, sink=sink), state=tree)
        assert torch.equal(restored.read(out), expected)
        sink.update = sink.inp.recv  # pyright: ignore[reportAttributeAccessIssue]
        restored.step(H, sink)
        assert torch.equal(cast(torch.Tensor, sink.results[0]).cpu(), expected)

    def test_a_paused_run_gives_the_values_on_the_device_uncopied(self):
        # What a pause reads, before Run.state_dict moves it to the CPU.
        source, sink, out = sends_on_cuda()
        tree = walk(Group(source=source, sink=sink))
        wiring = wire(tree, ReadOnlyTimer(VirtualTimer()))
        wiring.wires[0][out].send(torch.ones(2, dtype=F64, device="cuda:0"))
        (held,) = wiring.outlets[out].outbox.read().values
        taken = capture(wiring, [None, None], [0, 0], 0, 1.0)
        assert port_slots(taken, "source", "out")["out"]["value"] is held
        moved = port_slots(cpu(taken), "source", "out")["out"]["value"]
        assert moved.device == torch.device("cpu")
        assert torch.equal(moved, torch.ones(2, dtype=F64))
