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
    (``_on_thread_start``): it turns off gradient recording, creates ``generator``
    from ``seed``, and, when ``device`` is CUDA, makes the device and a stream of
    the node's own current, and waits for that stream at the end of each
    ``on_update``. Overriding ``on_start`` does not skip this.

    ``state_dict`` runs on the runtime's control thread while the node is paused,
    so it cannot ``recv`` or ``send``; return the node's own state from it.

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
    _stream: torch.cuda.Stream | None = None

    def __init__(
        self, *, device: torch.device | str = "cpu", seed: int | None = None
    ) -> None:
        device = torch.device(device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError(f"a node computes on the CPU or CUDA, got {device}")
        self._device = device
        self._seed = seed

    @property
    def device(self) -> torch.device:
        """The device the node computes on and receives messages on.

        Allocate with it (``torch.zeros(..., device=self.device)``): the thread's
        current device is this one, but an explicit device states the intent.
        """
        return self._device

    @property
    def generator(self) -> torch.Generator:
        """The node's own random generator on ``device``, seeded with ``seed``.

        Use it instead of the global generator, which all nodes share. It is
        created anew from ``seed`` whenever the network starts, also when it is
        restored; to continue the same sequence after a restore, put
        ``self.generator.get_state()`` in ``state_dict`` and ``set_state`` it in
        ``load_state_dict``.

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
            torch.cuda.set_device(self._device)
            stream = torch.cuda.Stream(self._device)
            torch.cuda.set_stream(stream)
            self._stream = stream
            self._synchronize_each_update(stream)
        generator = torch.Generator(self._device)
        if self._seed is None:
            generator.seed()
        else:
            generator.manual_seed(self._seed)
        self._generator = generator
        # The OS thread id: the Python thread object may differ between hooks.
        self._thread_ident = threading.get_ident()
        _hooks.inside = True
        self._mark_state_dict()

    # The runtime looks the hooks below up after _on_thread_start, on_start and
    # load_state_dict, so instance attributes set here replace them.

    def _synchronize_each_update(self, stream: torch.cuda.Stream) -> None:
        """End each ``on_update`` by waiting for the node's own stream, so that the
        CPU never runs ahead of the GPU and a paused node has no pending work."""
        update = self.on_update

        def on_update(dt: float) -> None:
            update(dt)
            stream.synchronize()

        self.on_update = on_update

    def _mark_state_dict(self) -> None:
        """Mark the control thread as inside a hook while it runs ``state_dict``."""
        state_dict = self.state_dict

        def marked() -> object:
            outer = _hooks.inside
            _hooks.inside = True
            try:
                return state_dict()
            finally:
                _hooks.inside = outer

        self.state_dict = marked


class _Hooks(threading.local):
    # Whether this thread is running a node's hook (a node's own thread always is).
    inside = False


_hooks = _Hooks()


def in_hook() -> bool:
    """Whether the calling thread is running a node's hook."""
    return _hooks.inside


def thread_ident(node: Node) -> int | None:
    """Return the id of the thread ``node`` runs on, or ``None`` when it does not
    run."""
    return node._thread_ident  # pyright: ignore[reportPrivateUsage]  # Node's own module


def end_thread(node: Node) -> None:
    """Forget the thread of ``node`` once its network has stopped, so that a later
    thread that reuses the id is not taken for it."""
    node._thread_ident = None  # pyright: ignore[reportPrivateUsage]  # Node's own module


def synchronize(node: Node) -> None:
    """Wait for the work queued on the CUDA stream of ``node``, if it has one."""
    stream = node._stream  # pyright: ignore[reportPrivateUsage]  # Node's own module
    if stream is not None:
        stream.synchronize()
