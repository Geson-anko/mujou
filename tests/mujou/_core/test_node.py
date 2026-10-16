import threading
from typing import TYPE_CHECKING, override

import pytest
import torch

from mujou import Network, Node, Port, TensorSpec
from mujou._core.node import thread_ident
from mujou.testing import VirtualTimer
from mujou.time import Timer, Timestamp

if TYPE_CHECKING:
    from tests.mujou.conftest import StartManual


class Probe(Node):
    """Overrides ``on_start`` without calling the base, and records the thread's
    settings there."""

    def __init__(self, seed: int | None = None, *, init_base: bool = True) -> None:
        if init_base:
            super().__init__(seed=seed)
        self.out = Port(send=TensorSpec((3,)))
        self.grad_enabled: bool | None = None
        self.draws: torch.Tensor | None = None
        self.ident: int | None = None

    @override
    def on_start(self) -> None:
        self.grad_enabled = torch.is_grad_enabled()
        self.draws = torch.rand(3, generator=self.generator)
        self.ident = threading.get_ident()


class Solo(Network):
    def __init__(self, node: Node) -> None:
        self.node = node


# The tests run the networks in manual mode only so that idle nodes do not spin:
# on_start runs either way.


class TestThreadSetup:
    def test_on_start_runs_without_gradients_on_its_own_thread(
        self, start_manual: StartManual
    ):
        node = Probe(seed=0)
        start_manual(Solo(node))
        assert node.grad_enabled is False
        assert node.ident not in (None, threading.get_ident())
        # The setting is per thread: the host still records gradients.
        assert torch.is_grad_enabled()

    def test_the_generator_follows_the_seed(self, start_manual: StartManual):
        a, b, c = Probe(seed=7), Probe(seed=7), Probe(seed=8)
        for node in (a, b, c):
            start_manual(Solo(node))
        assert a.draws is not None and b.draws is not None and c.draws is not None
        expected = torch.rand(3, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a.draws, expected)
        assert torch.equal(b.draws, expected)
        assert not torch.equal(c.draws, expected)

    def test_works_without_calling_the_base_init(self, start_manual: StartManual):
        node = Probe(init_base=False)
        start_manual(Solo(node))
        assert node.device == torch.device("cpu")
        assert node.grad_enabled is False
        assert node.draws is not None

    def test_the_thread_is_forgotten_after_the_end(self, start_manual: StartManual):
        node = Probe()
        run = start_manual(Solo(node))
        assert thread_ident(node) == node.ident
        run.shutdown()
        assert thread_ident(node) is None

    def test_leaves_the_hooks_of_the_node_alone(self, start_manual: StartManual):
        node = Probe(seed=0)
        run = start_manual(Solo(node))
        run.state_dict()
        assert not {"on_update", "state_dict"} & set(vars(node))

    def test_the_generator_is_made_on_the_thread(self):
        with pytest.raises(RuntimeError, match="node's thread"):
            _ = Probe(seed=0).generator


class Clocked(Node):
    """Keeps the time of its run's timer in each hook."""

    def __init__(self) -> None:
        self.out = Port(send=TensorSpec((3,)))
        self.timers: list[Timer] = []
        self.times: list[Timestamp] = []

    def keep(self) -> None:
        self.timers.append(self.timer)
        self.times.append(self.timer.now())

    @override
    def on_start(self) -> None:
        self.keep()

    @override
    def on_update(self, dt: float) -> None:
        self.keep()

    @override
    def on_stop(self) -> None:
        self.keep()


class TestTimer:
    def test_is_the_timer_of_the_run_from_on_start_to_on_stop(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Clocked()
        virtual.advance(1.0)
        run = start_manual(Solo(node), timer=virtual)
        virtual.advance(0.5)
        run.step()
        run.shutdown()
        assert all(timer is virtual for timer in node.timers)
        assert node.times == [
            Timestamp(ns=1_000_000_000),
            Timestamp(ns=1_500_000_000),
            Timestamp(ns=1_500_000_000),
        ]

    def test_raises_outside_a_run(self, start_manual: StartManual):
        node = Clocked()
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer
        start_manual(Solo(node)).shutdown()
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer

    def test_is_released_when_a_run_is_refused(self, start_manual: StartManual):
        tree = start_manual(Solo(Clocked())).state_dict()
        node = Clocked()
        with pytest.raises(ValueError):
            # Refused at the start: the timer of the run that is still running
            # cannot be set back.
            start_manual(Solo(node), state=tree)
        with pytest.raises(RuntimeError, match="run"):
            _ = node.timer


class TestDevice:
    def test_defaults_to_the_cpu(self):
        assert Probe().device == torch.device("cpu")

    def test_takes_a_string(self):
        class OnCpu(Node):
            def __init__(self) -> None:
                super().__init__(device="cpu")

        assert OnCpu().device == torch.device("cpu")

    @pytest.mark.skipif(torch.cuda.is_available(), reason="needs no CUDA")
    def test_rejects_cuda_without_cuda(self):
        with pytest.raises(ValueError, match="CUDA is not available"):
            Node(device="cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_fixes_the_cuda_index(self):
        assert Node(device="cuda").device.index == torch.cuda.current_device()

    def test_rejects_devices_other_than_cpu_and_cuda(self):
        with pytest.raises(ValueError, match="CPU or CUDA"):
            Node(device="meta")


class OnDevice(Node):
    """Records the thread's CUDA settings in ``on_update``."""

    def __init__(self, device: str) -> None:
        super().__init__(device=device)
        self.out = Port(send=TensorSpec((3,)))
        self.current_device: int | None = None
        self.own_stream: bool | None = None
        self.grad_enabled: bool | None = None

    @override
    def on_update(self, dt: float) -> None:
        self.current_device = torch.cuda.current_device()
        default = torch.cuda.default_stream(self.device)
        self.own_stream = torch.cuda.current_stream() != default
        self.grad_enabled = torch.is_grad_enabled()


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


def needs_gpus(n: int) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < n,
        reason=f"needs {n} CUDA device(s)",
    )


@needs_gpus(1)
class TestCuda:
    @pytest.mark.parametrize("index", [0, pytest.param(1, marks=needs_gpus(2))])
    def test_hooks_run_on_the_device_on_a_stream_of_their_own(
        self, virtual: VirtualTimer, start_manual: StartManual, index: int
    ):
        node = OnDevice(f"cuda:{index}")
        run = start_manual(Solo(node))
        virtual.advance(0.001)
        run.step()
        assert node.current_device == index
        assert node.own_stream is True
        assert node.grad_enabled is False

    @pytest.mark.slow
    def test_waits_for_its_stream_after_each_update(
        self, virtual: VirtualTimer, start_manual: StartManual
    ):
        node = Busy()
        run = start_manual(Solo(node))
        for _ in range(5):
            virtual.advance(0.001)
            run.step()
        # The first update follows on_start, after which nothing waits.
        assert node.idle_at_update[1:] == [True] * 4
