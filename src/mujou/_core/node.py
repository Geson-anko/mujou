"""The node, the unit of asynchronous computation."""

import threading
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast, override

import torch
from torch import nn

from mujou._core import _native
from mujou._core.network import Component


class Node(_native.Node, Component):
    """A unit of asynchronous computation: the leaf of a network.

    Declare the ports as instance attributes in ``__init__`` (``build`` rejects
    ports declared on the class). A node must not hold other nodes or networks as
    attributes; put them side by side in a ``Network`` instead.

    Override the lifecycle hooks below; each does nothing by default, except
    ``state_dict`` and ``load_state_dict``. The framework calls them on the
    node's own thread, except ``state_dict``:

    - ``on_thread_start()``: once, right before ``on_start``, after the
      framework has set up the thread (see below). Make the node's own
      per-thread settings here; there is no need to call ``super()``.
    - ``on_start()``: once, when the network runs (also when it was restored).
      Allocate and initialize the state here.
    - ``load_state_dict(state)``: only when the network was restored, after
      ``on_start`` and before the first ``on_update``. ``state`` is a copy of what
      this node's ``state_dict`` returned. By default it copies the state into the
      collected tensors and modules (see below). When ``device`` is CUDA, the
      framework waits for the node's stream after it.
    - ``on_update(dt)``: repeatedly. ``dt`` is the global time in seconds since
      the node's last update. Read the ports with ``recv``, update in proportion
      to ``dt``, and write the ports with ``send``. It never waits for other nodes.
    - ``on_pause()``: when the node has stopped for a pause of the network (as
      ``Network.state_dict`` does). ``on_resume()``: when the pause ends, before
      the next ``on_update``.
    - ``on_exception()``: once on every node, when a hook of any node has raised,
      before ``on_stop``.
    - ``on_stop()``: once, when the network stops, also after an exception. It
      must not change what ``state_dict`` returns: the state taken after a normal
      end is the state at the stop.
    - ``state_dict()``: when the network's state is taken, on the runtime's control
      thread while the node is paused (or, after a normal end, on the host's
      thread that called ``Network.state_dict``), so ``recv`` and ``send`` raise
      ``RuntimeError`` there. Return the node's own state as a picklable object;
      it may hold the live tensors, because the framework copies it before the
      node resumes. By default it returns the collected tensors and modules (see
      below).

    The node collects its state by itself, as ``nn.Module`` does: every attribute
    assigned a ``torch.Tensor`` (``nn.Parameter`` included) or an ``nn.Module``,
    whether or not its name starts with ``_``, is registered under its name.
    Assigning any other value (``None`` too) under that name, or ``del``, removes
    it. Ports, numbers, lists and dicts of tensors are not collected, and neither
    are properties (what a setter stores is, under its own name). The default
    ``state_dict`` and ``load_state_dict`` save and restore what is registered;
    override both to choose the state yourself (they may call ``super()``).
    Allocate the tensors of the state by ``on_start`` and update them in place
    afterwards: the default ``load_state_dict`` copies into them in place.

    If any hook raises, the whole network stops. A hook must not call the methods
    of any ``Network`` (``write``, ``read``, ``state_dict``, ...): they raise
    ``RuntimeError`` there, because a node talks only through its own ports.

    Before ``on_start``, the framework sets up the node's thread: it turns off
    gradient recording (for backpropagation inside the node, wrap both the forward
    computation and ``backward`` in ``with torch.enable_grad():`` marked
    ``# mujou: local-autograd``), creates
    ``generator`` from ``seed``, and, when ``device`` is CUDA, makes the device
    and a stream of the node's own current. When ``device`` is CUDA it also
    waits for that stream after each ``on_update``, after ``load_state_dict`` and
    around ``state_dict``. It does all this in hooks of its own,
    ``_on_thread_start``, ``_after_update``, ``_restore_state`` and
    ``_capture_state``, which subclasses must not override. So overriding a
    public hook never skips it.

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
        same sequence after a restore, add ``self.generator.get_state()`` to what
        ``super().state_dict()`` returns, and in ``load_state_dict`` take it out
        and ``set_state`` it before passing the rest to ``super()``.

        Raises:
            RuntimeError: The node's thread has not started yet.
        """
        if self._generator is None:
            raise RuntimeError(
                "the generator is created on the node's thread; use it from on_start "
                "or a later hook"
            )
        return self._generator

    if not TYPE_CHECKING:
        # Hidden from the type checker, which would otherwise accept assignments
        # to undeclared attributes on every subclass.

        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if _is_data_descriptor(type(self), name):
                return  # A property: what its setter stores registers itself.
            names = _state_names(self)
            if isinstance(value, torch.Tensor | nn.Module):
                if name not in names:
                    _set_state_names(self, {**names, name: None})
            elif name in names:
                _set_state_names(self, {k: v for k, v in names.items() if k != name})

        def __delattr__(self, name: str) -> None:
            super().__delattr__(name)
            names = _state_names(self)
            if name in names:
                _set_state_names(self, {k: v for k, v in names.items() if k != name})

    @override
    def state_dict(self) -> Any:  # Any: an override returns any picklable object.
        """Return the collected state: a dict with, for each registered attribute
        in the order of registration, the tensor ``detach``-ed or the module's
        ``state_dict()``, under the attribute's name.

        The tensors share memory with the attributes (the framework copies the
        tree). ``requires_grad`` and ``.grad`` are not part of the state, as in
        ``nn.Module.state_dict``: detaching also makes a tensor that is part of an
        autograd graph picklable.
        """
        return {
            name: value.detach()
            if isinstance(value, torch.Tensor)
            else value.state_dict()
            for name, value in _state_items(self)
        }

    @override
    def load_state_dict(self, state: object) -> None:
        """Copy ``state`` (from the default ``state_dict``) into the registered
        attributes in place: ``copy_`` into each tensor and ``load_state_dict``
        into each module, without recording gradients.

        The objects stay the same, so references taken in ``__init__`` or
        ``on_start`` see the restored values, and a tensor that records gradients
        stays a leaf that does. ``copy_`` casts the dtype and moves the device.
        A tensor in ``state`` whose name is not registered (one first assigned
        after ``on_start``) is set as a new attribute, a copy of the value, and
        registered. Everything is checked before anything is copied, so on an
        error nothing has changed.

        Raises:
            TypeError: ``state`` or the state of a module is not a mapping, or
                the value for a tensor is not a tensor.
            ValueError: A registered attribute is missing from ``state``, a key
                that is not registered holds no tensor, or a tensor's shape
                differs or the tensor has elements that share memory.
            RuntimeError: A module's state does not have the keys or the shapes
                of the module's ``state_dict``, as ``nn.Module.load_state_dict``
                (strict) raises.
        """
        tree = _mapping(state, "the default load_state_dict takes")
        items = _state_items(self)
        missing = [name for name, _ in items if name not in tree]
        names = _state_names(self)
        unexpected = [
            key
            for key, value in tree.items()
            if key not in names and not isinstance(value, torch.Tensor)
        ]
        if missing or unexpected:
            raise ValueError(
                "the state does not match the node's tensors and modules: "
                + "; ".join(
                    f"{what} {keys}"
                    for what, keys in (
                        ("missing", missing),
                        ("not tensors and not registered", unexpected),
                    )
                    if keys
                )
            )
        for name, value in items:
            if isinstance(value, torch.Tensor):
                _check_tensor(name, value, tree[name])
            else:
                _check_module(name, value, tree[name])
        with torch.no_grad():
            for name, value in items:
                if isinstance(value, torch.Tensor):
                    value.copy_(cast(torch.Tensor, tree[name]))
                else:
                    value.load_state_dict(cast(Mapping[str, Any], tree[name]))
            for key, value in tree.items():
                if key not in names:
                    setattr(self, key, cast(torch.Tensor, value).clone())

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
    def _restore_state(self, state: object) -> None:
        self.load_state_dict(state)
        # The copies queue on the node's stream; wait for them, so that the host
        # may drop the tree it passed (whose memory they read) once run returns.
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


# The attribute that holds a node's registry of state names. Long, so that it does
# not meet a name of the user's.
_STATE_NAMES = "_mujou_state_names"


def _state_names(node: Node) -> Mapping[str, None]:
    """The names of the attributes of ``node`` that hold state, in the order of
    registration (an ordered set).

    The registry is replaced, never changed, so that a shallow copy of a node
    (``copy.copy``) does not share changes with the original.
    """
    return cast(Mapping[str, None], vars(node).get(_STATE_NAMES, {}))


def _set_state_names(node: Node, names: dict[str, None]) -> None:
    vars(node)[_STATE_NAMES] = names  # Not through __setattr__.


def _state_items(node: Node) -> list[tuple[str, torch.Tensor | nn.Module]]:
    """Return ``(attribute name, value)`` for the tensors and modules registered on
    ``node``, in the order of registration."""
    return [(name, getattr(node, name)) for name in _state_names(node)]


def _is_data_descriptor(klass: type, name: str) -> bool:
    """Whether ``name`` on ``klass`` is a data descriptor other than a slot (a
    property, say), which stores the value somewhere else."""
    for base in klass.__mro__:
        if name in vars(base):
            attribute: object = vars(base)[name]
            return hasattr(type(attribute), "__set__") and not isinstance(
                attribute, types.MemberDescriptorType
            )
    return False


def _mapping(value: object, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} a mapping, got {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _check_tensor(name: str, tensor: torch.Tensor, source: object) -> None:
    """Check that ``tensor.copy_(source)`` restores ``tensor`` element by element."""
    if not isinstance(source, torch.Tensor):
        raise TypeError(
            f"the state's {name!r} is a {type(source).__name__}, not a tensor"
        )
    if source.shape != tensor.shape:
        raise ValueError(
            f"the state's {name!r} has the shape {tuple(source.shape)}, but the "
            f"tensor has {tuple(tensor.shape)}"
        )
    # copy_ refuses to write into elements that share memory (from expand, say).
    if any(
        stride == 0 and size > 1
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    ):
        raise ValueError(
            f"the tensor {name!r} has elements that share memory, so it cannot be "
            "restored in place"
        )


def _check_module(name: str, module: nn.Module, source: object) -> None:
    """Check ``source`` against ``module.state_dict()`` as the strict
    ``module.load_state_dict`` would, before anything is copied."""
    tree = _mapping(source, f"the state's {name!r} of a module must be")
    own = module.state_dict()
    missing = [key for key in own if key not in tree]
    unexpected = [key for key in tree if key not in own]
    shapes = [
        key
        for key, value in own.items()
        if key in tree
        and isinstance(tree[key], torch.Tensor)
        and cast(torch.Tensor, tree[key]).shape != value.shape
    ]
    if missing or unexpected or shapes:
        raise RuntimeError(
            f"the state's {name!r} does not match the module: missing {missing}, "
            f"unexpected {unexpected}, other shapes {shapes}"
        )


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


def capture_state(node: Node) -> object:
    """Take the state of ``node`` as the runtime does while it is paused, on the
    calling thread (for a network that has ended)."""
    return node._capture_state()  # pyright: ignore[reportPrivateUsage]  # Node's own module
