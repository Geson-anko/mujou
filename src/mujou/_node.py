"""The node, the unit of asynchronous computation."""

import threading
from typing import cast, override

import torch

from mujou import _core
from mujou._network import Component


class Node(_core.Node, Component):
    """A unit of asynchronous computation: the leaf of a network.

    Declare the ports as instance attributes in ``__init__`` (``build`` rejects
    ports declared on the class). A node must not hold other nodes or networks as
    attributes; put them side by side in a ``Network`` instead.

    Override the lifecycle hooks below; each does nothing by default. The framework
    calls them on the node's own thread, except ``state_dict``:

    - ``on_start()``: once, when the network runs (also when it was restored).
      Allocate and initialize the state here.
    - ``load_state_dict(state)``: only when the network was restored, after
      ``on_start`` and before the first ``on_update``. ``state`` is a copy of what
      this node's ``state_dict`` returned.
    - ``on_update(dt)``: repeatedly. ``dt`` is the global time in seconds since
      the node's last update. Read the ports with ``recv``, update in proportion
      to ``dt``, and write the ports with ``send``. It never waits for other nodes.
    - ``on_pause()``: when the node has stopped for a pause of the network (as
      ``Network.state_dict`` does). ``on_resume()``: when the pause ends, before
      the next ``on_update``.
    - ``on_exception()``: once on every node, when a hook of any node has raised,
      before ``on_stop``. It must not change what ``state_dict`` returns.
    - ``on_stop()``: once, when the network stops, also after an exception.
    - ``state_dict()``: when the network's state is taken, on the runtime's control
      thread while the node is paused, so ``recv`` and ``send`` raise
      ``RuntimeError`` there. Return the node's own state as a picklable object;
      it may hold the live tensors, because the framework copies it before the
      node resumes.

    If any hook raises, the whole network stops. A hook must not call the methods
    of any ``Network`` (``write``, ``read``, ``state_dict``, ...): they raise
    ``RuntimeError`` there, because a node talks only through its own ports.

    Before ``on_start``, the framework sets up the node's thread: it turns off
    gradient recording (for backpropagation inside the node, wrap both the forward
    computation and ``backward`` in ``with torch.enable_grad():`` marked
    ``# mujou: local-autograd``), creates
    ``generator`` from ``seed``, and, when ``device`` is CUDA, makes the device
    and a stream of the node's own current. When ``device`` is CUDA it also waits
    for that stream after each ``on_update`` and around ``state_dict``. The
    framework does this in hooks of its own, ``_on_thread_start``,
    ``_after_update`` and ``_capture_state``, which subclasses must not override.
    So overriding ``on_start``, ``on_update`` or ``state_dict`` does not skip it.

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
        ValueError: ``device`` is neither the CPU nor a CUDA device, or is CUDA
            while CUDA is not available.
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
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    f"the node's device is {device}, but CUDA is not available"
                )
            # torch's stub types the index as int, but it is None for "cuda".
            index = cast(int | None, device.index)
            if index is None:
                # Fix the device now: "cuda" means the host's current device, and
                # set_device needs an index.
                device = torch.device("cuda", torch.cuda.current_device())
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

        Use it instead of the global generator, which all nodes share (do not
        call ``torch.manual_seed`` in a node). It is created anew from ``seed``
        whenever the network starts, also when it is restored; to continue the
        same sequence after a restore, put ``self.generator.get_state()`` in
        ``state_dict`` and ``set_state`` it in ``load_state_dict``.

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
        generator = torch.Generator(self._device)
        if self._seed is None:
            generator.seed()
        else:
            generator.manual_seed(self._seed)
        self._generator = generator
        # The OS thread id: the Python thread object may differ between hooks.
        self._thread_ident = threading.get_ident()
        _hooks.inside = True

    @override
    def _after_update(self) -> None:
        # The CPU never runs ahead of the GPU. (Other hooks may still leave work on
        # the stream; _capture_state waits for it before taking the state.)
        if self._stream is not None:
            self._stream.synchronize()

    @override
    def _capture_state(self) -> object:
        # On the control thread: mark it as inside a hook for the call.
        outer = _hooks.inside
        _hooks.inside = True
        try:
            stream = self._stream
            if stream is None:
                return self.state_dict()
            # Copies that state_dict makes queue on the node's stream, after the
            # work of every hook (not only on_update); wait for them before the
            # framework copies the tree on its own stream.
            with torch.cuda.stream(stream):
                stream.synchronize()
                state = self.state_dict()
                stream.synchronize()
            return state
        finally:
            _hooks.inside = outer


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
