"""The node, the unit of asynchronous computation."""

import threading
from typing import override

import torch

from mujou import _core
from mujou._network import Component


class Node(_core.Node, Component):
    """A unit of asynchronous computation: the leaf of a network.

    Declare the ports as instance attributes in ``__init__`` (``build`` rejects
    ports declared on the class). When the network runs, the framework calls
    the node's lifecycle hooks (``on_start``, ``on_update(dt)``, ...) on a thread of
    its own; in the hooks the node reads its ports with ``recv`` and writes them
    with ``send``. A node must not hold other nodes or networks as attributes; put
    them side by side in a ``Network`` instead.

    Before ``on_start``, the framework sets up the node's thread
    (``_on_thread_start``): it turns off gradient recording, makes a CUDA stream of
    the node's own the current one when ``device`` is CUDA, and creates
    ``generator`` from ``seed``. Overriding ``on_start`` does not skip this.

    Calling ``super().__init__()`` is optional: without it the node computes on
    the CPU and seeds its generator nondeterministically.

    Example::

        class Hidden(Node):
            def __init__(self, n_in: int, n_out: int) -> None:
                super().__init__(seed=0)
                x, h = TensorSpec((32, n_in)), TensorSpec((32, n_out))
                # Send W^T eps back and receive x from the previous side (one peer).
                self.prev = PrevPort(send=x, recv=x)
                # Send h forward and receive W^T eps from each next peer (many).
                self.next = NextPort(send=h, recv=h)

            def on_update(self, dt: float) -> None:
                x = self.prev.recv()  # torch.Tensor
                feedback = self.next.recv()  # dict[Channel, torch.Tensor]
                ...
                self.next.send(h)
                self.prev.send(wt_eps)


        l1, l2 = Hidden(784, 256), Hidden(256, 256)
        l1.connect(l2)  # l1.next <-> l2.prev

    Args:
        device: Where the node computes: the CPU or a CUDA device. Received
            messages arrive on it.
        seed: The seed of ``generator``. ``None`` seeds it nondeterministically.

    Raises:
        ValueError: ``device`` is neither the CPU nor a CUDA device.
    """

    # Class-level defaults, so that subclasses need not call super().__init__().
    _device: torch.device = torch.device("cpu")
    _seed: int | None = None
    _generator: torch.Generator | None = None
    _thread_ident: int | None = None

    # The core's __new__ takes any arguments and drops them, so subclasses can take
    # their own.
    def __init__(  # pyright: ignore[reportInconsistentConstructor]
        self, *, device: torch.device | str = "cpu", seed: int | None = None
    ) -> None:
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError(f"a node computes on the CPU or CUDA, got {device}")
        self._device = device
        self._seed = seed

    @property
    def device(self) -> torch.device:
        """The device the node computes on and receives messages on."""
        return self._device

    @property
    def generator(self) -> torch.Generator:
        """The node's own random generator on ``device``, seeded with ``seed``.

        Use it instead of the global generator, which all nodes share.

        Raises:
            RuntimeError: The node's thread has not started yet.
        """
        if self._generator is None:
            raise RuntimeError(
                "the generator is created on the node's thread; use it from on_start "
                "or a later hook"
            )
        return self._generator

    @override
    def _on_thread_start(self) -> None:
        # Every setting here is per thread, so it does not touch other nodes.
        torch.set_grad_enabled(False)
        if self._device.type == "cuda":
            torch.cuda.set_stream(torch.cuda.Stream(self._device))
        generator = torch.Generator(self._device)
        if self._seed is None:
            generator.seed()
        else:
            generator.manual_seed(self._seed)
        self._generator = generator
        # The OS thread id: the Python thread object may differ between hooks.
        self._thread_ident = threading.get_ident()


def thread_ident(node: Node) -> int | None:
    """Return the id of the thread ``node`` runs on, or ``None`` before it starts."""
    return node._thread_ident  # pyright: ignore[reportPrivateUsage]  # Node's own module
