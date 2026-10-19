"""The node, the unit of asynchronous computation."""

import inspect
import math
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, cast, override

import torch
from torch import nn

from mujou._core import _native
from mujou._core.component import Component, is_data_descriptor
from mujou._core.hooks import enter_node_thread, running_hook
from mujou._core.timer import ReadOnlyTimer


class Node(_native.Node, Component):
    """A unit of asynchronous computation: the leaf of a network.

    Declare the ports as instance attributes in ``__init__``, after
    ``super().__init__()`` (``build`` rejects ports declared on the class). The
    node registers them as they are assigned. A node must not hold other nodes or
    networks as attributes (assigning one raises ``ValueError``); put them side by
    side in a ``Network`` instead.

    Override the lifecycle hooks below; each does nothing by default, except
    ``state_dict`` and ``load_state_dict``. As in any subclass, call
    ``super().__init__()`` first in ``__init__`` and the same hook of ``super()``
    first in each override. The framework calls the hooks on the node's own
    thread, except ``state_dict``:

    - ``on_thread_start()``: once, right before ``on_start``, after the
      framework has set up the thread (see below). Make the node's own
      per-thread settings here.
    - ``on_start()``: once, when the network runs (also when it was restored).
      Allocate and initialize the state here.
    - ``load_state_dict(state)``: only when the network was restored, after
      ``on_start`` and before the first ``on_update``. ``state`` is a copy of what
      this node's ``state_dict`` returned. By default it copies the state into the
      collected tensors and modules (see below). When ``device`` is CUDA, the
      framework waits for the node's stream after it.
    - ``on_update(dt)``: repeatedly, once every ``period`` on average (see
      below). ``dt`` is the time in seconds on the run's ``timer`` since the
      node's last update. Read the ports with ``recv``, update in proportion to
      ``dt``, and write the ports with ``send``. It never waits for other nodes.
    - ``on_pause()``: when the node has stopped for a pause of the network (as
      ``Run.state_dict`` does). ``on_resume()``: when the pause ends, before
      the next ``on_update``.
    - ``on_exception()``: once on every node, when a hook of any node has raised,
      before ``on_stop``.
    - ``on_stop()``: once, when the network stops, also after an exception. It
      must not change what ``state_dict`` returns: the state taken after a normal
      end is the state at the stop.
    - ``state_dict()``: when the network's state is taken, on the runtime's control
      thread while the node is paused (or, after a normal end, on the host's
      thread that called ``Run.state_dict``), so ``recv`` and ``send`` raise
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

    If any hook raises, the whole network stops. A hook must not call
    ``mujou.run`` or the methods of any ``Run`` (``write``, ``read``,
    ``state_dict``, ...): they raise ``RuntimeError`` there, because a node talks
    only through its own ports.

    Before ``on_start``, the framework sets up the node's thread: it turns off
    gradient recording (to backpropagate inside the node, wrap the forward
    computation and ``backward`` in ``with torch.enable_grad():`` marked
    ``# mujou: local-autograd``), creates ``generator`` from ``seed`` on
    ``device``, and, when ``device`` is CUDA, makes the device and a stream of the
    node's own current. On CUDA it also waits for that stream after
    ``on_update``, after ``load_state_dict`` and around ``state_dict``. It does this in hooks of its own, whose names start with
    ``_``: do not override those, and overriding a public hook never skips them.

    The node updates once every ``period`` (0.01 s by default) of the run's timer
    on average: the time an update takes
    counts toward the period, and a node that falls a whole ``period`` behind does
    not update several times in a row to catch up. A pause does not count. ``dt``
    is still the time that actually passed since the last update, longer than
    ``period`` when the node is late, so keep updating in proportion to it and
    stay stable for the longest ``dt`` it can get. A manual run
    (``ManualRun.step``) ignores the period.

    ``period=0.0`` updates as soon as any time has passed since the last update.
    Set it only when the node must: such a node keeps a CPU core busy, and takes
    time from the other nodes, whose ``dt`` it makes irregular.

    ``device``, ``generator``, ``period`` and ``timer`` (the reading side of the
    run's timer) are reserved names: do not assign attributes of those names
    (a class attribute ``period`` makes the run raise ``TypeError``).

    Example::

        class Hidden(Node):
            def __init__(self, n_in: int, n_out: int) -> None:
                super().__init__(seed=0)
                x, h = TensorSpec((32, n_in)), TensorSpec((32, n_out))
                # Send W^T eps back and receive x from the previous side (one peer).
                self.prev = PrevPort(send=x, recv=x)
                # Send h forward and receive W^T eps from each next peer (many).
                self.next = NextPort(send=h, recv=h)

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                x = self.prev.recv()  # torch.Tensor
                feedback = self.next.recv()  # dict[Channel, torch.Tensor]
                ...
                self.next.send(h)
                self.prev.send(wt_eps)


        l1, l2 = Hidden(784, 256), Hidden(256, 256)
        l1.next.connect(l2.prev)

    Args:
        device: Where the node computes: the CPU or a CUDA device (no other
            device type is supported). Received messages arrive on it.
            ``"cuda"`` without an index is fixed to the host's current CUDA
            device.
        seed: The seed of ``generator``. ``None`` seeds it nondeterministically.
        period: The time between two updates on average, in seconds of the
            run's timer. ``0.0`` updates as soon as any time has passed.

    Raises:
        TypeError: ``period`` is a ``bool``.
        ValueError: ``device`` is neither the CPU nor a CUDA device, or is CUDA
            while CUDA is not available, or ``period`` is negative or not
            finite.
    """

    # Class-level defaults, for a subclass that does not call super().__init__().
    _device: torch.device = torch.device("cpu")
    _seed: int | None = None
    _period: float = 0.01
    _generator: torch.Generator | None = None
    _thread_ident: int | None = None
    _stream: torch.cuda.Stream | None = None
    _timer: ReadOnlyTimer | None = None

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        seed: int | None = None,
        period: float = 0.01,
    ) -> None:
        if isinstance(period, bool):
            raise TypeError(f"the period must be a number of seconds, got {period}")
        if not (math.isfinite(period) and period >= 0):
            raise ValueError(f"the period must be finite and >= 0, got {period}")
        super().__init__()
        self._device = _check_device(torch.device(device))
        self._seed = seed
        self._period = period

    @property
    def device(self) -> torch.device:
        """The device the node computes on and receives messages on.

        Allocate with it (``torch.zeros(..., device=self.device)``): the thread's
        current device is this one, but an explicit device states the intent.
        """
        return self._device

    @property
    def period(self) -> float:
        """The time between two updates on average, in seconds of the run's
        timer. ``0.0`` updates as soon as any time has passed."""
        return self._period

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
            names = _state_names(self)
            if isinstance(value, _STATE_TYPES):
                # A property is not registered: what its setter stores is. Looked
                # up only for a new name, as the lookup walks the MRO.
                if name not in names and not is_data_descriptor(type(self), name):
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

    @property
    def timer(self) -> ReadOnlyTimer:
        """The reading side of the timer of the run this node is in, which
        measures its ``dt``.

        Read the time with ``self.timer.now()``. The timer is the one given to
        ``mujou.run(..., timer=...)``, or the global timer by default. A
        ``ReadOnlyTimer`` only reads and does not hold the timer: a node does not
        change the time of its run (``Timer.sleep``, ``set_time_scale`` and
        ``VirtualTimer.advance`` raise ``RuntimeError`` in a hook). ``timer`` is a
        reserved name: do not assign an attribute of that name.

        It can be read from the start of the run (also in ``on_thread_start``
        and ``on_start``). Once the host has seen a normal end (e.g. after
        ``Run.shutdown`` returned), it is replaced by a ``ReadOnlyTimer`` that
        stays at the time of the state, also in the ``state_dict`` that the host
        takes then. A ``ReadOnlyTimer`` kept from before the end keeps reading
        the running timer.

        Raises:
            RuntimeError: The node has not run: before its run starts the nodes,
                after a start that was refused, or once the host has seen an
                end by an exception.
        """
        if self._timer is None:
            raise RuntimeError(
                "the node has no timer: it is read from the start of a run, and is "
                "removed when a start is refused or a run ends by an exception"
            )
        return self._timer

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
        enter_node_thread()

    @override
    def _after_update(self) -> None:
        # The CPU never runs ahead of the GPU. (Other hooks may still leave work on
        # the stream; _capture_state waits for it before taking the state.)
        self._synchronize()

    @override
    def _restore_state(self, state: object) -> None:
        self.load_state_dict(state)
        # The copies queue on the node's stream; wait for them, so that the host
        # may drop the tree it passed (whose memory they read) once run returns.
        self._synchronize()

    @override
    def _capture_state(self) -> object:
        # On the control thread: mark it as inside a hook for the call.
        with running_hook():
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

    def _synchronize(self) -> None:
        """Wait for the node's own stream on CUDA; nothing on the CPU."""
        if self._stream is not None:
            self._stream.synchronize()


def _check_device(device: torch.device) -> torch.device:
    """Check that a node can compute on ``device`` and fix its index.

    A node computes on the CPU or on CUDA. Other device types (MPS, XPU, ...)
    are not supported.

    Raises:
        ValueError: ``device`` is neither the CPU nor a CUDA device, or is CUDA
            while CUDA is not available.
    """
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(
            f"a node computes on the CPU or CUDA (only these are supported), got "
            f"{device}"
        )
    if not torch.cuda.is_available():
        raise ValueError(f"the node's device is {device}, but CUDA is not available")
    # torch's stub types the index as int, but it is None for "cuda".
    if cast(int | None, device.index) is None:
        # Fix the device now: "cuda" means the host's current device, and
        # set_device needs an index.
        device = torch.device("cuda", torch.cuda.current_device())
    return device


# The types of the values collected into the state. A tuple, not a union:
# isinstance checks a tuple several times faster, and Node.__setattr__ checks every
# assignment.
_STATE_TYPES: Final = (torch.Tensor, nn.Module)

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


def thread_ident(node: Node) -> int | None:
    """Return the id of the thread ``node`` runs on, or ``None`` when it does not
    run."""
    return node._thread_ident  # pyright: ignore[reportPrivateUsage]  # Node's own module


def end_thread(node: Node) -> None:
    """Forget the thread of ``node`` once its network has stopped, so that a later
    thread that reuses the id is not taken for it."""
    node._thread_ident = None  # pyright: ignore[reportPrivateUsage]  # Node's own module


def set_timer(node: Node, timer: ReadOnlyTimer | None) -> None:
    """Give ``node`` the view of its run's timer, or take it back with ``None``."""
    node._timer = timer  # pyright: ignore[reportPrivateUsage]  # Node's own module


def checked_period(node: Node) -> float | None:
    """Return the period of ``node`` in seconds for the runtime, which rounds it
    to ns, or ``None`` for ``0.0`` (no wait between updates).

    Raises:
        TypeError: A subclass of ``Node`` replaced the reserved ``period``.
    """
    if inspect.getattr_static(type(node), "period") is not vars(Node)["period"]:
        raise TypeError(
            f"{type(node).__qualname__} replaces period, a reserved name of Node: "
            "pass the period to Node.__init__ instead"
        )
    period = node._period  # pyright: ignore[reportPrivateUsage]  # Node's own module
    return period if period > 0 else None


def capture_state(node: Node) -> object:
    """Take the state of ``node`` as the runtime does while it is paused, on the
    calling thread (for a network that has ended)."""
    return node._capture_state()  # pyright: ignore[reportPrivateUsage]  # Node's own module
