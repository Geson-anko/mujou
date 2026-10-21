import contextvars
import queue
import subprocess
import sys
from typing import TYPE_CHECKING, Any, override

import pytest
import torch

from mujou import Group, Node, Port, TensorSpec
from mujou.time import VirtualTimer

if TYPE_CHECKING:
    from tests.mujou.conftest import Start, StartManual

H = 1e-3  # seconds per step

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
