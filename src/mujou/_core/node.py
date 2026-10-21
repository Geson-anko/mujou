"""The node, the unit of asynchronous computation."""

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeIs, cast, final

import torch

from mujou._core._native import ReadOnlyTimer
from mujou._core.component import Component
from mujou._core.current import outside, running


@final
@dataclass(frozen=True, slots=True)
class _Config:
    """The settings of a node, from ``Node.__init__``."""

    device: torch.device
    period: float
    dedicated_thread: bool
    generator: torch.Generator


class Node(Component):
    """A unit of asynchronous computation: the leaf of a network.

    Call ``super().__init__(...)`` first in ``__init__`` (``mujou.run`` raises
    ``TypeError`` for a node that did not), then declare the node's ports and
    state as attributes of the instance. A run reads the attributes when it
    starts. A node must not hold other nodes or networks as attributes, nor
    ports in a list, a tuple or a dict (the run raises ``TypeError``); put
    nodes side by side in a ``Network`` instead.

    Override the lifecycle hooks below; each does nothing by default. As in any
    subclass, call the same hook of ``super()`` first in each override. The
    framework calls the hooks on the thread that runs the node (see "Threads"
    below), in this order:

    - ``on_start()``: once, when the run starts (also when it is restored).
      Open the node's I/O here. A node with ``dedicated_thread=True`` gets it on
      its own thread, so it can also make its per-thread settings here. Do not
      send anything that depends on the state here: a restore loads the state
      after it.
    - ``load_state_dict(state)``: only when the run was restored, after
      ``on_start`` and before the first ``on_update``.
    - ``on_update(dt)``: repeatedly, once every ``period`` on average (see
      below). ``dt`` is the time in seconds on the run's timer since the
      node's last update. Read the ports with ``recv``, update in proportion to
      ``dt``, and write the ports with ``send``. It never waits for other nodes.
    - ``on_pause()``: when the node has stopped for the run to take its state
      (``Run.state_dict``). ``on_resume()``: when that pause ends, before the
      next ``on_update``. A pause of the timer (``Timer.pause``) calls neither:
      the node only sees no time pass.
    - ``on_exception()``: once on every node, when a hook of any node has raised,
      before ``on_stop``.
    - ``on_stop()``: once, when the run stops, also after an exception. It
      must not change what ``state_dict`` returns: the state taken after a normal
      end is the state at the stop.

    ``state_dict()`` is called when the run's state is taken, while the node is
    paused (or, after a normal end, on the host's thread that called
    ``Run.state_dict``). Its ports refuse to receive or send there, so that
    taking the state is not seen by the other nodes. The framework copies what
    it returns before the node resumes.

    The state of a node is, as for an ``nn.Module``, every attribute that holds
    a ``torch.Tensor`` (``nn.Parameter`` included) or an object with
    ``state_dict`` and ``load_state_dict`` (an ``nn.Module``, a
    ``pc.LearnClock``; not a class), every attribute registered with
    ``register_state``, and the state of ``generator``. What an attribute holds
    when the state is taken or loaded decides. The settings that ``__init__``
    computes from its arguments (a time constant, say) are no state: a node
    restored with other arguments keeps its own. Register the plain values that
    change while the node runs (a counter, the time last seen), and the
    containers of tensors (a list as a replay buffer, a dict of tensors): a
    list or a dict is no state by itself, even of tensors. Make the state
    in ``__init__`` and update it in place: ``load_state_dict`` copies into the
    tensors that the node holds. To choose the state yourself, override both
    ``state_dict`` and ``load_state_dict``: extend what ``super().state_dict()``
    returns, and pass ``super().load_state_dict`` the state without the keys
    you added.

    If any hook raises, the whole run stops. A hook must not call ``mujou.run``
    or the methods of any ``Run`` (``write``, ``read``, ``state_dict``, ...):
    they raise ``RuntimeError`` there, because a node talks only through its own
    ports.

    Threads: by default the node runs on the workers, a pool of threads that
    the run shares among its nodes (``mujou.run(..., workers=...)``). Each hook
    may then run on a different thread from the previous one, but the hooks of
    a node never run at the same time, and each sees what the previous one
    wrote. Whether the next hook sees what a hook set per thread
    (``threading.local``, ``threading.current_thread()``, settings of torch
    such as ``torch.set_grad_enabled`` or ``torch.autocast``) depends on the
    schedule, and the next node on the same worker may see it too: set such
    settings back before the hook returns (inside a ``with``, say).
    ``contextvars`` do persist: each node runs its hooks in a
    ``contextvars.Context`` of its own, which starts empty (it does not inherit
    the host's). A hook that blocks (``time.sleep``, waiting for I/O or a
    lock) holds its worker meanwhile, and the other nodes' ``dt`` grows; give
    such a node ``dedicated_thread=True``, so that it runs on a thread of its
    own, with every hook on that thread.

    The framework leaves gradient recording as it is, on by default as in
    plain PyTorch: a hook may call ``backward`` directly, and computes without
    a graph inside ``with torch.no_grad():``. No graph crosses nodes: a port
    sends a detached copy. Before each hook, when ``device`` is CUDA, the
    framework makes the device and a stream of the node's own current: the
    only per-thread settings it makes, and around ``state_dict`` (which may
    run on the host's thread) it sets the thread's own back. On CUDA it also
    waits for the work queued on that stream after ``on_update`` (holding its
    worker meanwhile), after ``load_state_dict`` and around ``state_dict``. It
    does this around the hooks, so overriding one never skips it.

    The node updates once every ``period`` (0.01 s by default) of the run's
    timer on average: the time an update takes counts toward the period, and a
    node that falls a whole ``period`` behind does not update several times in
    a row to catch up. A pause does not count. ``dt`` is still the time that
    actually passed since the last update, longer than ``period`` when the
    node is late, so keep updating in proportion to it and stay stable for the
    longest ``dt`` it can get. A manual run (``ManualRun.step``) ignores the
    period.

    ``period=0.0`` updates as soon as any time has passed since the last update.
    Set it only when the node must: such a node keeps a CPU core busy, and takes
    time from the other nodes, whose ``dt`` it makes irregular.
    ``period=math.inf`` never updates by itself: only ``ManualRun.step`` updates
    such a node.

    Example::

        class Hidden(Node):
            def __init__(self, n_in: int, n_out: int) -> None:
                super().__init__(seed=0)
                x, h = TensorSpec((32, n_in)), TensorSpec((32, n_out))
                # Send W^T eps back and receive x from the previous side (one peer).
                self.prev = Port(send=x, recv=x)
                # Send h forward and receive W^T eps from each next peer (many).
                self.next = MultiPort(send=h, recv=h)

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
        period: The time between two updates on average, in seconds of the
            run's timer. ``0.0`` updates as soon as any time has passed.
        seed: The seed of ``generator``. ``None`` seeds it nondeterministically.
        device: Where the node computes: the CPU or a CUDA device (no other
            device type is supported). Received messages arrive on it.
            ``"cuda"`` without an index is fixed to the host's current CUDA
            device.
        dedicated_thread: Run the node on a thread of its own instead of the
            run's workers: for a node whose hooks block, or that relies on
            per-thread state set once (in ``on_start``).

    Raises:
        ValueError: ``device`` is neither the CPU nor a CUDA device, or is CUDA
            while CUDA is not available, or ``period`` is negative or NaN.
    """

    def __init__(
        self,
        *,
        period: float = 0.01,
        seed: int | None = None,
        device: torch.device | str = "cpu",
        dedicated_thread: bool = False,
    ) -> None:
        super().__init__()
        if not period >= 0:
            raise ValueError(f"the period must be >= 0 seconds, got {period}")
        fixed = _check_device(torch.device(device))
        generator = torch.Generator(fixed)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)
        self.__config = _Config(fixed, period, dedicated_thread, generator)
        # The names given to register_state.
        self.__registered: set[str] = set()

    @property
    def device(self) -> torch.device:
        """The device the node computes on and receives messages on.

        Allocate with it (``torch.zeros(..., device=self.device)``): the current
        device in the hooks is this one, but an explicit device states the
        intent.
        """
        return self.__config.device

    @property
    def period(self) -> float:
        """The time between two updates on average, in seconds of the run's
        timer. ``0.0`` updates as soon as any time has passed."""
        return self.__config.period

    @property
    def dedicated_thread(self) -> bool:
        """Whether the node runs on a thread of its own instead of the run's
        workers."""
        return self.__config.dedicated_thread

    @property
    def generator(self) -> torch.Generator:
        """The node's own random generator on ``device``, seeded with ``seed``.

        ``seed`` sets nothing else: use this generator instead of the global
        one, which all nodes share (do not call ``torch.manual_seed`` in a
        node), also to initialize modules (``nn.init.normal_(w,
        generator=self.generator)``). ``super().__init__`` creates it, so it can
        be used in ``__init__`` already, and the hooks go on with the same
        sequence. Its state is part of the node's state, so a restored node
        goes on with the same sequence too.
        """
        return self.__config.generator

    @property
    def timer(self) -> ReadOnlyTimer:
        """The reading side of the timer of the run whose hook runs, which
        measures the ``dt`` of its nodes.

        Read the time with ``self.timer.now()`` in the hooks (``state_dict``
        included). The timer is the one given to ``mujou.run(..., timer=...)``,
        the global timer by default, or, for a run restored from a state, a
        new timer that started at the time of the state (``Run.timer``). A
        ``ReadOnlyTimer`` only reads: a node does not change the time of its
        run (``Timer.sleep``, ``set_time_scale`` and ``VirtualTimer.advance``
        raise ``RuntimeError`` in a hook). It stops at the time the run ends,
        so the hooks that run after the end (``on_exception``, ``on_stop``)
        see the time stand still; measure real time there instead. The host
        reads ``Run.timer``.

        Raises:
            RuntimeError: This is not read in a hook of this node while it runs
                (but in ``__init__``, by the host, or by another node).
        """
        current = running()
        if current is None or current.node is not self:
            raise outside(f"{type(self).__name__}.timer", "the host reads run.timer")
        return current.clock

    def register_state(self, name: str) -> None:
        """Add the attribute ``name`` to the state, whatever value it holds.

        Use it for the plain values that change while the node runs (a
        counter, the time last seen) and for the containers of tensors (a list
        as a replay buffer), which are not state by themselves. The
        attribute is state while it is assigned, whatever it holds. The default
        ``state_dict`` returns the value as it is, and the default
        ``load_state_dict`` assigns a deep copy of the saved value (or copies
        into a tensor, or loads into an object with ``load_state_dict``, that
        the attribute holds then).

        Register in ``__init__``: a restore loads into new instances, which
        must have the same state.

        The state is data: when the run takes it, it detaches every tensor of
        what ``state_dict`` returns (in its dicts, lists and tuples) and copies
        the result, so no graph goes with it.

        ``torch.load(..., weights_only=True)`` reads only basic types (``int``,
        ``float``, ``bool``, ``str``, ``None``, and lists, tuples and dicts of
        them and of tensors): keep the registered values to those if the state
        is saved with ``torch.save``.

        Raises:
            AttributeError: The node has no attribute ``name`` (not assigned
                yet, misspelled, or a property); assign it first.
        """
        if name not in vars(self):
            raise AttributeError(
                f"{type(self).__name__}.{name} is not assigned: assign it, then "
                "register it"
            )
        self.__registered.add(name)

    def on_start(self) -> None:
        """Called once when the run starts (also when it is restored), before any
        other hook. Do not send anything that depends on the state here: a
        restore loads it later."""

    def on_update(self, dt: float) -> None:
        """Called repeatedly with ``dt``, the seconds on the run's timer since the
        node's last update (always > 0). Read the ports, update in proportion to
        ``dt`` and send."""

    def on_pause(self) -> None:
        """Called when the node has stopped for the run to take its state
        (``Run.state_dict``), at the end of its current ``on_update``. A pause
        of the timer (``Timer.pause``) does not call it."""

    def on_resume(self) -> None:
        """Called when the pause for taking the state ends, before the next
        ``on_update``."""

    def on_exception(self) -> None:
        """Called once on every node when a hook of any node has raised, before
        ``on_stop``. What it raises is logged and ignored."""

    def on_stop(self) -> None:
        """Called once when the run stops, also after an exception. It must not
        change what ``state_dict`` returns. What it raises is logged and
        ignored."""

    def state_dict(self) -> dict[str, Any]:
        """Return the node's state: ``"generator"``, the state of
        ``generator``, then each attribute of the state in the order of
        assignment, under its name: a tensor ``detach``-ed, the ``state_dict()``
        of an object that has one, or a registered value as it is.

        The tensors share memory with the attributes (the framework copies the
        tree). ``requires_grad`` and ``.grad`` are not part of the state, as in
        ``nn.Module.state_dict``.
        """
        state: dict[str, Any] = {"generator": self.generator.get_state()}
        for name, value in self.__items():
            state[name] = _state_of(value)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore ``state``, as the default ``state_dict`` returned it, into the
        attributes of the state, without recording gradients: ``copy_`` into
        each tensor, ``load_state_dict`` into each object that has one, and
        assign a ``copy.deepcopy`` of the saved value to any other registered
        attribute, with the tensors in it (in its dicts, lists and tuples)
        moved to ``device``.

        The objects stay the same, so references taken in ``__init__`` or
        ``on_start`` see the restored values, and a tensor that records
        gradients stays a leaf that does. ``copy_`` casts the dtype and moves
        the device. The state of ``generator`` is set as it is saved, and a
        generator keeps its state in a format of its device's type: the state
        must come from a node on the same type of device (the CPU or CUDA).

        An override that keeps a tensor of ``state`` as it is shares its memory
        with the tree given to ``mujou.run(..., state=...)``: copy it instead.

        Raises:
            ValueError: The keys of ``state`` are not those of this node's
                state, or a tensor's shape differs.
        """
        items = self.__items()
        expected = ["generator", *(name for name, _ in items)]
        missing = [key for key in expected if key not in state]
        unexpected = [key for key in state if key not in expected]
        if missing or unexpected:
            raise ValueError(
                f"the state does not match {type(self).__name__}: missing "
                f"{missing}, unexpected {unexpected}"
            )
        with torch.no_grad():
            for name, value in items:
                self.__load(name, value, state[name])
        self.generator.set_state(state["generator"])

    def __items(self) -> list[tuple[str, object]]:
        """``(name, value)`` for each attribute of the state, in the order of
        assignment."""
        registered = self.__registered
        return [
            (name, value)
            for name, value in vars(self).items()
            if name in registered or is_state(value)
        ]

    def __load(self, name: str, value: object, saved: Any) -> None:
        if isinstance(value, torch.Tensor):
            shape = tuple(cast(torch.Tensor, saved).shape)
            if shape != tuple(value.shape):
                raise ValueError(
                    f"{type(self).__name__}.{name}: the state has the shape "
                    f"{shape}, the tensor {tuple(value.shape)}"
                )
            value.copy_(saved)
        elif _has_state(value):
            value.load_state_dict(saved)
        else:
            # The tree keeps its tensors on the CPU; the node holds them on its
            # device.
            copied = copy.deepcopy(saved)
            setattr(self, name, map_tensors(copied, lambda t: t.to(self.device)))


def initialized(node: Node) -> bool:
    """Whether ``Node.__init__`` has run on ``node``."""
    return "_Node__config" in vars(node)


class _Stateful(Protocol):
    def state_dict(self) -> Any: ...
    def load_state_dict(self, state: Any, /) -> Any: ...


def _has_state(value: object) -> TypeIs[_Stateful]:
    """Whether ``value`` has a state of its own: it has ``state_dict`` and
    ``load_state_dict`` and is not a class (whose methods these would be)."""
    return (
        not isinstance(value, type)
        and hasattr(value, "state_dict")
        and hasattr(value, "load_state_dict")
    )


def is_state(value: object) -> bool:
    """Whether an attribute that holds ``value`` is part of a node's state by
    itself: a tensor, or an object with a state of its own."""
    return isinstance(value, torch.Tensor) or _has_state(value)


def _state_of(value: object) -> object:
    """What the default ``state_dict`` returns for an attribute that holds
    ``value``, as ``Node.__load`` loads it back."""
    if isinstance(value, torch.Tensor):
        return value.detach()
    if _has_state(value):
        return value.state_dict()
    return value


def map_tensors(value: object, f: Callable[[torch.Tensor], torch.Tensor]) -> object:
    """``value``, a state, with ``f`` applied to each tensor in its dicts,
    lists and tuples, in new containers (a dict of the same type: an
    ``OrderedDict`` keeps the metadata of a module's ``state_dict``). The run
    detaches the states it takes with it (``driver.py``) and moves them to the
    CPU (``state.py``)."""
    if isinstance(value, torch.Tensor):
        return f(value)
    if isinstance(value, dict):
        items = copy.copy(cast(dict[object, object], value))
        for key, item in items.items():
            items[key] = map_tensors(item, f)
        return items
    if isinstance(value, list):
        return [map_tensors(element, f) for element in cast(list[object], value)]
    if type(value) is tuple:
        elements = cast(tuple[object, ...], value)
        return tuple(map_tensors(element, f) for element in elements)
    return value


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
