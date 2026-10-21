import contextvars
import math
import queue
import subprocess
import sys
import threading
from typing import TYPE_CHECKING, Any, cast, override

import pytest
import torch

from mujou import Group, Network, NetworkError, Node, Port, TensorSpec
from mujou.time import VirtualTimer
from tests.mujou.nodes import F64, vec

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual

H = 1e-3  # seconds per step


def u(k: float) -> torch.Tensor:
    """The k-th input: different at every step."""
    return torch.tensor([math.sin(k), math.cos(k)], dtype=F64)


def euler(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The reference of one ``DeviceLeaky`` update with ``dt=H``."""
    return x + min(H, 1.0) * (target - x)


# Set by each node in its on_start; the host never sets it.
VAR: contextvars.ContextVar[str | None] = contextvars.ContextVar("var", default=None)


class Hooks(Node):
    """Records each hook it runs, with the value of ``VAR`` there; sets ``VAR``
    to its name in ``on_start``. Overrides every hook without calling
    ``super()``."""

    def __init__(self, name: str) -> None:
        super().__init__(period=0.01)
        self.out = Port(send=TensorSpec((1,)))
        self.name = name
        self.calls: list[tuple[str, str | None]] = []
        self.updated: queue.Queue[None] = queue.Queue()

    def record(self, hook: str) -> None:
        self.calls.append((hook, VAR.get()))

    @override
    def on_start(self) -> None:
        VAR.set(self.name)
        self.record("on_start")

    @override
    def load_state_dict(self, state: object) -> None:
        self.record("load_state_dict")

    @override
    def on_update(self, dt: float) -> None:
        self.record("on_update")
        self.updated.put(None)

    @override
    def on_pause(self) -> None:
        self.record("on_pause")

    @override
    def on_resume(self) -> None:
        self.record("on_resume")

    @override
    def state_dict(self) -> dict[str, Any]:
        self.record("state_dict")
        return {}

    @override
    def on_stop(self) -> None:
        self.record("on_stop")


class Logs(Node):
    """Appends ``(name, hook)`` to the shared ``log`` for each hook it is called
    in; raises in ``on_update`` once ``trigger`` is set."""

    def __init__(self, name: str, log: list[tuple[str, str]]) -> None:
        super().__init__()
        self.name = name
        self.log = log
        self.trigger = threading.Event()
        self.error: Exception | None = None

    def hooks(self) -> list[str]:
        return [hook for name, hook in self.log if name == self.name]

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.log.append((self.name, "on_update"))
        if self.trigger.is_set():
            self.error = ValueError(f"boom in {self.name}")
            raise self.error

    @override
    def on_pause(self) -> None:
        super().on_pause()
        self.log.append((self.name, "on_pause"))

    @override
    def on_resume(self) -> None:
        super().on_resume()
        self.log.append((self.name, "on_resume"))

    @override
    def state_dict(self) -> dict[str, Any]:
        self.log.append((self.name, "state_dict"))
        return super().state_dict()

    @override
    def on_exception(self) -> None:
        super().on_exception()
        self.log.append((self.name, "on_exception"))

    @override
    def on_stop(self) -> None:
        super().on_stop()
        self.log.append((self.name, "on_stop"))


class ThreeLogs(Network):
    def __init__(self, log: list[tuple[str, str]]) -> None:
        super().__init__()
        self.a = Logs("a", log)
        self.b = Logs("b", log)
        self.c = Logs("c", log)

    def nodes(self) -> list[Logs]:
        return [self.a, self.b, self.c]


class TestHooks:
    def test_run_in_order_also_when_restored(self, start_manual: StartManual):
        first = Hooks("a")
        run = start_manual(first)
        run.step(H)
        tree = run.state_dict()
        run.shutdown()
        node = Hooks("a")
        run = start_manual(node, state=tree)
        run.step(H)
        run.state_dict()
        run.shutdown()
        assert [hook for hook, _ in node.calls] == [
            "on_start",
            "load_state_dict",
            "on_update",
            "on_pause",
            "state_dict",
            "on_resume",
            "on_stop",
        ]

    def test_each_node_keeps_its_own_contextvars_on_a_shared_worker(
        self, virtual: VirtualTimer, start: Start
    ):
        nodes = Hooks("a"), Hooks("b")
        run = start(Group(a=nodes[0], b=nodes[1]), timer=virtual, workers=1)
        for _ in range(3):
            virtual.advance(0.01)
            for node in nodes:
                node.updated.get(timeout=5.0)
        run.state_dict()
        run.shutdown()
        # Every hook, state_dict included, sees what its own on_start set.
        for node in nodes:
            hooks = {hook for hook, _ in node.calls}
            assert hooks >= {"on_start", "on_update", "state_dict", "on_stop"}
            assert {value for _, value in node.calls} == {node.name}
        # Nor does the host see what a node set.
        assert VAR.get() is None

    def test_a_hook_replaced_at_any_time_takes_effect(self, start_manual: StartManual):
        node = Hooks("a")
        run = start_manual(node)
        run.step(H)

        def replaced(dt: float) -> None:
            node.record("replaced")

        node.on_update = replaced
        run.step(H)
        assert [hook for hook, _ in node.calls][-2:] == ["on_update", "replaced"]

    def test_state_dict_pauses_every_node_takes_the_state_then_resumes_each(
        self, start_manual: StartManual
    ):
        log: list[tuple[str, str]] = []
        net = ThreeLogs(log)
        run = start_manual(net)
        run.step(H)
        log.clear()

        run.state_dict()
        run.step(H)  # on_resume comes before the next on_update

        for node in net.nodes():
            assert node.hooks() == ["on_pause", "state_dict", "on_resume", "on_update"]
        hooks = [hook for _, hook in log]
        last_pause = max(i for i, h in enumerate(hooks) if h == "on_pause")
        states = [i for i, h in enumerate(hooks) if h == "state_dict"]
        first_resume = min(i for i, h in enumerate(hooks) if h == "on_resume")
        # The state is taken at one time: every node is paused before any state,
        # and none resumes before every state is taken.
        assert last_pause < min(states)
        assert max(states) < first_resume

    def test_every_node_gets_on_exception_then_on_stop_and_the_host_the_error(
        self, start_manual: StartManual
    ):
        log: list[tuple[str, str]] = []
        net = ThreeLogs(log)
        run = start_manual(net)
        run.step(H)
        net.b.trigger.set()

        with pytest.raises(NetworkError) as raised:
            run.step(H)
        with pytest.raises(NetworkError):
            run.join()  # every on_stop has returned

        assert raised.value.__cause__ is net.b.error
        for node in net.nodes():
            hooks = node.hooks()
            assert hooks[-2:] == ["on_exception", "on_stop"]
            assert hooks.count("on_exception") == 1
            assert hooks.count("on_stop") == 1


class OnDevice(Node):
    """Records the thread's CUDA settings in ``on_update``."""

    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.out = Port(send=TensorSpec((3,)))
        self.current_device: int | None = None
        self.own_stream: bool | None = None

    @override
    def on_update(self, dt: float) -> None:
        self.current_device = torch.cuda.current_device()
        default = torch.cuda.default_stream(self.device)
        self.own_stream = torch.cuda.current_stream() != default


class StreamRecorder(Node):
    """Records the current CUDA device and stream in each ``on_update``, and
    switches the stream away (to its device's default stream) at its end without
    restoring it."""

    def __init__(self, device: str) -> None:
        super().__init__(device=device, period=0.01)
        self.out = Port(send=TensorSpec((3,)))
        self.devices: list[int] = []
        self.streams: list[int] = []
        self.updated: queue.Queue[None] = queue.Queue()

    @override
    def on_update(self, dt: float) -> None:
        self.devices.append(torch.cuda.current_device())
        self.streams.append(torch.cuda.current_stream().cuda_stream)
        # Left behind for whatever runs next on this thread.
        torch.cuda.set_stream(torch.cuda.default_stream(self.device))
        self.updated.put(None)


class Busy(Node):
    """Each ``on_update`` adds 1 to a large tensor 20 times, in place, and first
    records whether the node's stream had finished the previous update."""

    def __init__(self) -> None:
        super().__init__(device="cuda:0")
        self.out = Port(send=TensorSpec((3,)))
        self.idle_at_update: list[bool] = []

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(1 << 26, device=self.device)  # 256 MB of float32

    @override
    def on_update(self, dt: float) -> None:
        self.idle_at_update.append(torch.cuda.current_stream().query())
        for _ in range(20):
            self.w.add_(1.0)


BIG = 1 << 26  # 256 MB of float32
ADDS = 20  # in-place updates of the whole tensor per on_update
CUDA0 = torch.device("cuda", 0)


class DeviceLeaky(Node):
    """``Leaky`` with ``tau=1`` on ``device``."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        self.prev = Port(recv=vec())
        self.next = Port(send=vec())

    @override
    def on_start(self) -> None:
        self.x = torch.zeros(2, dtype=F64, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        self.x += min(dt, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)


class DeviceChain(Network):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.a = DeviceLeaky(device)
        self.b = DeviceLeaky(device)
        self.a.next.connect(self.b.prev)
        self.prev = self.a.prev
        self.next = self.b.next


class CudaSource(Node):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.next = Port(send=vec())

    @override
    def on_update(self, dt: float) -> None:
        self.next.send(torch.tensor([1.0, 2.0], dtype=F64, device=self.device))


class CudaSink(Node):
    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.prev = Port(recv=vec())
        self.received: torch.Tensor | None = None

    @override
    def on_update(self, dt: float) -> None:
        self.received = self.prev.recv()


class SourceToSink(Network):
    def __init__(self, source: CudaSource, sink: CudaSink) -> None:
        super().__init__()
        self.source = source
        self.sink = sink
        source.next.connect(sink.prev)


class Counter(Node):
    """Keeps ``BIG`` floats that all equal ``k`` between updates; each update adds
    1 to every float ``ADDS`` times, in place."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        self.first: torch.Tensor | None = None  # w as the first update finds it

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(BIG, device=self.device)
        self.k = 0

    @override
    def on_update(self, dt: float) -> None:
        if self.first is None:
            self.first = self.w.clone()
        for _ in range(ADDS):
            self.w.add_(1.0)
        self.k += ADDS

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"w": self.w, "k": self.k}

    @override
    def load_state_dict(self, state: object) -> None:
        loaded = cast(dict[str, Any], state)
        self.w.copy_(loaded["w"])
        self.k = loaded["k"]


def is_uniform(tree: dict[str, Any]) -> bool:
    """Whether every float of the captured ``Counter`` equals its ``k``."""
    state = tree["nodes"]["node"]["state"]
    w, k = state["w"], state["k"]
    return w.min().item() == k and w.max().item() == k


HOST_ADDS = 50


class QueuesOnTheHost(Node):
    """Queues ``HOST_ADDS`` updates of a large tensor on the host's stream in
    ``__init__`` and reads it in ``on_start``."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)
        self.w = torch.zeros(BIG, device=self.device)
        for _ in range(HOST_ADDS):
            self.w.add_(1.0)
        self.seen: float | None = None

    @override
    def on_start(self) -> None:
        self.seen = self.w.min().item()


class BusyStart(Node):
    """Leaves ``HOST_ADDS`` updates of a large tensor on its stream in
    ``on_start`` (no ``on_update`` waits for them)."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(BIG, device=self.device)
        for _ in range(HOST_ADDS):
            self.w.add_(1.0)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"w": self.w}


class BusyStop(Node):
    """Leaves ``HOST_ADDS`` updates of a large tensor on its stream in
    ``on_stop``."""

    def __init__(self, device: torch.device) -> None:
        super().__init__(device=device)

    @override
    def on_start(self) -> None:
        self.w = torch.zeros(BIG, device=self.device)

    @override
    def on_stop(self) -> None:
        for _ in range(HOST_ADDS):
            self.w.add_(1.0)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"w": self.w}


@pytest.mark.cuda
class TestCuda:
    def test_a_node_on_the_cpu_does_not_initialize_cuda(self):
        # A fresh interpreter, as the tests before have initialized CUDA here.
        script = """
import torch
import mujou.testing
from mujou import Group, Node

run = mujou.testing.run(Group(node=Node()))
run.step(0.001)
run.state_dict()  # paused
run.shutdown()
run.state_dict()  # after the end, on this thread
print(torch.cuda.is_initialized())
"""
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert done.stdout.strip() == "False"

    @pytest.mark.parametrize("index", [0, pytest.param(1, marks=pytest.mark.cuda(2))])
    def test_hooks_run_on_the_device_on_a_stream_of_their_own(
        self, start_manual: StartManual, index: int
    ):
        node = OnDevice(f"cuda:{index}")
        run = start_manual(Group(node=node))
        run.step(H)
        assert node.current_device == index
        assert node.own_stream is True

    @pytest.mark.parametrize(
        "indices", [(0, 0), pytest.param((0, 1), marks=pytest.mark.cuda(2))]
    )
    def test_nodes_sharing_a_worker_keep_their_own_device_and_stream(
        self, virtual: VirtualTimer, start: Start, indices: tuple[int, int]
    ):
        a, b = (StreamRecorder(f"cuda:{i}") for i in indices)
        run = start(Group(a=a, b=b), timer=virtual, workers=1)
        for _ in range(3):
            virtual.advance(0.01)
            for node in (a, b):
                node.updated.get(timeout=5.0)
        run.shutdown()
        for node, index in zip((a, b), indices, strict=True):
            assert node.devices == [index] * 3
            assert len(set(node.streams)) == 1
            assert node.streams[0] != torch.cuda.default_stream(index).cuda_stream
        assert a.streams[0] != b.streams[0]

    @pytest.mark.parametrize("index", [0, pytest.param(1, marks=pytest.mark.cuda(2))])
    def test_the_state_after_the_end_leaves_the_hosts_device_and_stream(
        self, start_manual: StartManual, index: int
    ):
        # After the end, the node's state_dict runs on the host's thread.
        run = start_manual(Group(node=OnDevice(f"cuda:{index}")))
        run.step(H)
        run.shutdown()
        device, stream = torch.cuda.current_device(), torch.cuda.current_stream()
        run.state_dict()
        assert torch.cuda.current_device() == device
        assert torch.cuda.current_stream() == stream

    @pytest.mark.slow
    def test_waits_for_its_stream_after_each_update(self, start_manual: StartManual):
        # Busy overrides on_start and on_update without calling super(): the
        # framework's work around the hooks does not depend on it.
        node = Busy()
        run = start_manual(Group(node=node))
        for _ in range(5):
            run.step(H)
        # The first update follows on_start, after which nothing waits.
        assert node.idle_at_update[1:] == [True] * 4

    @pytest.mark.cuda(2)
    def test_a_message_goes_from_one_gpu_to_another(self, start_manual: StartManual):
        source, sink = CudaSource("cuda:0"), CudaSink("cuda:1")
        net = SourceToSink(source, sink)
        run = start_manual(net)
        run.step(H, source, sink)
        received = sink.received
        assert received is not None
        assert received.device == torch.device("cuda", 1)
        assert torch.equal(received.cpu(), torch.tensor([1.0, 2.0], dtype=F64))

    def test_the_state_is_on_the_cpu_and_restores_to_the_device(
        self, start_manual: StartManual
    ):
        source = DeviceChain(CUDA0)
        run = start_manual(source, inputs={source.prev: u(1)})
        run.step(H, source.a, source.b)
        tree = run.state_dict()
        run.shutdown()
        x = tree["nodes"]["a"]["state"]["x"]
        assert x.device == torch.device("cpu")
        net = DeviceChain(CUDA0)
        start_manual(net, state=tree)
        assert net.a.x.device == CUDA0
        assert torch.equal(net.a.x.cpu(), x)

    def test_manual_steps_match_the_discrete_algorithm(self, start_manual: StartManual):
        net = DeviceChain(CUDA0)
        run = start_manual(net)
        xa = torch.zeros(2, dtype=F64, device=CUDA0)
        xb = torch.zeros(2, dtype=F64, device=CUDA0)
        for k in range(20):
            run.write({net.prev: u(k)})
            run.step(H, net.a, net.b)
            xa = euler(xa, u(k).to(CUDA0))
            xb = euler(xb, xa)
        assert torch.equal(run.read(net.next), xb.cpu())

    @pytest.mark.slow
    def test_state_dict_is_whole_while_a_node_updates_in_place(self, start: Start):
        run = start(Group(node=Counter(CUDA0)))
        run.timer.sleep(0.2)
        trees: list[dict[str, Any]] = []
        for _ in range(40):
            trees.append(run.state_dict())
            run.timer.sleep(0.01)
        torn = [i for i, tree in enumerate(trees) if not is_uniform(tree)]
        # The copies stay as taken while the node goes on updating in place.
        run.timer.sleep(0.2)
        changed = [i for i, tree in enumerate(trees) if not is_uniform(tree)]
        run.shutdown()
        assert torn == []
        assert changed == []
        assert trees[-1]["nodes"]["node"]["state"]["k"] > 0

    @pytest.mark.slow
    def test_on_start_sees_the_work_the_host_queued(self, start_manual: StartManual):
        seen: list[float | None] = []
        for _ in range(5):
            # Run each node right after making it, while its work is still queued.
            node = QueuesOnTheHost(CUDA0)
            start_manual(Group(node=node)).shutdown()
            seen.append(node.seen)
        assert seen == [float(HOST_ADDS)] * 5

    @pytest.mark.slow
    def test_state_dict_waits_for_the_work_of_on_start(self, start_manual: StartManual):
        captured: list[tuple[float, float]] = []
        for _ in range(5):
            run = start_manual(Group(node=BusyStart(CUDA0)))
            w = run.state_dict()["nodes"]["node"]["state"]["w"]
            run.shutdown()
            captured.append((w.min().item(), w.max().item()))
        assert captured == [(float(HOST_ADDS), float(HOST_ADDS))] * 5

    @pytest.mark.slow
    def test_state_dict_after_the_end_waits_for_the_work_of_on_stop(
        self, start_manual: StartManual
    ):
        captured: list[tuple[float, float]] = []
        for _ in range(5):
            run = start_manual(Group(node=BusyStop(CUDA0)))
            run.shutdown()
            w = run.state_dict()["nodes"]["node"]["state"]["w"]
            captured.append((w.min().item(), w.max().item()))
        assert captured == [(float(HOST_ADDS), float(HOST_ADDS))] * 5

    @pytest.mark.slow
    def test_a_restored_node_starts_from_the_tree(
        self,
        start_manual: StartManual,
    ):
        net = Group(node=Counter(CUDA0))
        run = start_manual(net)
        for _ in range(3):
            run.step(H)
        tree = run.state_dict()
        run.shutdown()

        node = Counter(CUDA0)
        restored = start_manual(Group(node=node), state=tree)
        restored.step(H)
        assert node.first is not None
        expected = torch.full((BIG,), float(3 * ADDS), device=CUDA0)
        assert torch.equal(node.first, expected)
        assert torch.equal(tree["nodes"]["node"]["state"]["w"].to(CUDA0), expected)
