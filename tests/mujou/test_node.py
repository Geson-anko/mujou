import threading
from typing import TYPE_CHECKING, override

import pytest
import torch

from mujou import Network, Node, Port, TensorSpec
from mujou._node import thread_ident

if TYPE_CHECKING:
    from tests.mujou.conftest import RunNetwork


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
        self, run_network: RunNetwork
    ):
        node = Probe(seed=0)
        run_network(Solo(node), _manual=True)
        assert node.grad_enabled is False
        assert node.ident not in (None, threading.get_ident())
        # The setting is per thread: the host still records gradients.
        assert torch.is_grad_enabled()

    def test_the_generator_follows_the_seed(self, run_network: RunNetwork):
        a, b, c = Probe(seed=7), Probe(seed=7), Probe(seed=8)
        for node in (a, b, c):
            run_network(Solo(node), _manual=True)
        assert a.draws is not None and b.draws is not None and c.draws is not None
        expected = torch.rand(3, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a.draws, expected)
        assert torch.equal(b.draws, expected)
        assert not torch.equal(c.draws, expected)

    def test_works_without_calling_the_base_init(self, run_network: RunNetwork):
        node = Probe(init_base=False)
        run_network(Solo(node), _manual=True)
        assert node.device == torch.device("cpu")
        assert node.grad_enabled is False
        assert node.draws is not None

    def test_the_thread_is_forgotten_after_the_end(self, run_network: RunNetwork):
        node = Probe()
        net = run_network(Solo(node), _manual=True)
        assert thread_ident(node) == node.ident
        net.shutdown()
        assert thread_ident(node) is None

    def test_leaves_the_hooks_of_the_node_alone(self, run_network: RunNetwork):
        node = Probe(seed=0)
        net = run_network(Solo(node), _manual=True)
        net.state_dict()
        assert not {"on_update", "state_dict"} & set(vars(node))

    def test_the_generator_is_made_on_the_thread(self):
        with pytest.raises(RuntimeError, match="node's thread"):
            _ = Probe(seed=0).generator


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
