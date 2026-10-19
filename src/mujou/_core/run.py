"""Running networks: ``mujou.run`` and the ``Run`` it returns (``specs/network.md``)."""

import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import Any, Never, Self, cast, override

import torch

from mujou._core import _native
from mujou._core._native import Timestamp  # by name: the API reference shows it
from mujou._core.graph import build_node, start_once
from mujou._core.hooks import in_hook
from mujou._core.network import Network, RunGroup, unbuild
from mujou._core.node import Node
from mujou._core.port import Port
from mujou._core.session import Session
from mujou._core.state import StateDict
from mujou._core.timer import GLOBAL, Timer, VirtualTimer


class Run:
    """A run of networks, as ``mujou.run`` returns it: what the host uses to talk
    to the running nodes.

    The nodes run in the background, on the run's worker threads (or threads of
    their own); the host's thread only writes inputs, reads outputs and takes the state:

    - ``write`` inputs, ``read`` / ``read_with_time`` / ``subscribe`` outputs,
      take a ``state_dict``.
    - ``shutdown()`` to stop the run, or ``join()`` to wait until it stops.
    - After a normal end: take the final ``state_dict``.

    A ``with`` statement shuts the run down at its end. These methods are for
    the host: calling them from a node's hook raises ``RuntimeError``.

    If a node raises, the whole run stops. If that happens while ``mujou.run``
    starts the nodes, ``mujou.run`` raises ``NetworkError``; otherwise every later
    call of ``write``, ``read``, ``read_with_time``, ``subscribe`` (and its
    iterator), ``state_dict``, ``shutdown`` and ``join`` raises it. Its ``__cause__`` is the node's
    exception. The state at the exception is not kept.

    A run keeps running when the host drops it, until ``shutdown`` or, at the
    latest, the interpreter's exit (which waits for ``on_stop``). At the exit,
    the runs start stopping once the main thread has ended, before the
    interpreter joins the non-daemon threads, and the exit waits until they have
    stopped.

    Example::

        from typing import override

        import torch

        import mujou
        from mujou import NextPort, Node, Port, PrevPort, TensorSpec

        VECTOR = TensorSpec((2,))


        class Error(Node):  # sends its input minus its teacher
            def __init__(self) -> None:
                super().__init__()
                self.prev = PrevPort(recv=VECTOR)
                self.teacher = Port(recv=VECTOR)
                self.next = NextPort(send=VECTOR)

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                self.next.send(self.prev.recv() - self.teacher.recv())


        node = Error()
        x, y = torch.ones(2), torch.zeros(2)
        with mujou.run(node, inputs={node.prev: x, node.teacher: y}) as run:
            run.write({node.prev: 2 * x, node.teacher: x})  # switch together
            mujou.time.sleep(0.1)  # present the input for 0.1 s of global time
            error, t = run.read_with_time(node.next)
        state = run.state_dict()  # the state at the end, to save
    """

    _session: Session
    _timer: Timer

    def __init__(self) -> Never:
        raise TypeError("a Run is made by mujou.run() or mujou.testing.run()")

    @property
    def timer(self) -> Timer:
        """The run's timer: the ``timer`` given to ``mujou.run``, or the global
        timer (the one ``mujou.time`` reads) when none was given.

        Every node's ``dt`` and every time of the run is on it."""
        return self._timer

    # P: a dict built beforehand has a precise key type, which a Mapping keyed by
    # Port[Any, Any, Any] would reject (Mapping keys are invariant). A dict message
    # is a Mapping[str, object]: a TypedDict is not a Mapping[str, torch.Tensor].
    def write[P: Port[Any, Any, Any]](
        self, values: Mapping[P, torch.Tensor | Mapping[str, object]]
    ) -> Timestamp:
        """Write messages to unconnected single-peer ports that receive, and
        return the time they carry.

        The values are detached and copied. All of them switch together and carry
        the same time: at any one moment, a node sees either all of them or none.
        (A node that reads two of them one after the other may still see a later
        write in between; compare the times to tell.) Any port of the tree can be
        written, exposed by the root or not.

        The time is the run's timer at the write, as a node's ``recv_with_time``
        and the ``/in`` slots of ``state_dict`` show it; ``read_with_time`` shows
        the times of sends, not of writes. An empty ``values`` changes nothing and returns
        the time all the same.

        Raises:
            RuntimeError: The run has stopped normally (no node reads what is
                written then).
            ValueError: A port is outside the run, connected, takes many peers,
                or receives nothing; or a shape or the keys differ.
            TypeError: A message or dtype differs from the spec.
            NetworkError: The run stopped because a node raised.
        """
        return _host(self).write(values)

    def read[S](self, port: Port[S, Any, Any]) -> S:
        """Return the latest message ``port`` sent.

        Any port that sends can be read, so a connected port shows an inner
        node. Only ``send`` (to every peer) updates the value read here; before
        the first one it is the port's ``initial`` (zeros by default) or the
        value restored from ``state``. The tensors are new copies on the CPU.
        ``read_with_time`` also returns the time it was sent at.

        Raises:
            ValueError: ``port`` is outside the run or sends nothing.
            NetworkError: The run stopped because a node raised.
        """
        return cast(S, _host(self).read(port)[0])

    def read_with_time[S](self, port: Port[S, Any, Any]) -> tuple[S, Timestamp]:
        """Return the latest message ``port`` sent, as ``read`` does, and the time
        it was sent at.

        The time is on the run's timer. A ``Timestamp`` does not say which timer
        it is on: compare it only with times of the same timer (with
        ``mujou.time`` only when the run is on the global timer).

        Raises:
            ValueError: ``port`` is outside the run or sends nothing.
            NetworkError: The run stopped because a node raised.
        """
        return cast(tuple[S, Timestamp], _host(self).read(port))

    def subscribe[S](
        self, port: Port[S, Any, Any], timeout: float = math.inf
    ) -> Iterator[tuple[S, Timestamp]]:
        """Iterate over the messages ``port`` sends, with their times.

        The first step yields the latest message at the time of this call (before
        the first ``send``, the port's ``initial`` or the restored value) without
        waiting. Each
        later step waits until a newer message is sent and yields the latest;
        messages sent in between are skipped. Only the consumer waits: the
        sending node never does. The iterator ends when the run shuts down
        normally. Shut the run down before the program ends: without a
        ``timeout``, a thread iterating here ends only with the run. In a
        ``ManualRun`` nothing is sent unless ``step`` is called, so a wait here
        without a ``timeout`` on the thread that steps never ends.

        To wait for one message that meets a condition, use ``wait_for``.

        Args:
            port: The port to follow.
            timeout: Real seconds, counted from this call, that the whole
                iteration may take, the time spent in the loop's body included.
                ``math.inf`` iterates as long as the run runs.

        Raises:
            ValueError: ``port`` is outside the run or sends nothing; or
                ``timeout`` is negative or NaN.
            TimeoutError: From the iterator, at a step after the first that
                begins once ``timeout`` has passed, or that would wait past it.
                The iterator ends; the run keeps running.
            NetworkError: The run stopped because a node raised (also from the
                iterator).
        """
        return cast(Iterator[tuple[S, Timestamp]], _host(self).subscribe(port, timeout))

    def wait_for[S](
        self,
        port: Port[S, Any, Any],
        condition: Callable[[S], bool],
        timeout: float = math.inf,
    ) -> tuple[S, Timestamp]:
        """Wait until ``port`` sends a message that meets ``condition``, and return
        it with its time.

        Checks the latest message first, then each newer one as ``subscribe``
        yields them (messages sent in between are skipped). Only the host waits:
        the nodes never do. In a ``ManualRun`` nothing is sent unless ``step`` is
        called, so a wait without a ``timeout`` on the thread that steps never
        ends. Here with the node ``Error`` of the example of
        ``Run``::

            node = Error()
            x = torch.ones(2)
            with mujou.run(node, inputs={node.prev: x}) as run:
                error, t = run.wait_for(
                    node.next, lambda e: torch.equal(e, x), timeout=5.0
                )

        Args:
            port: The port to follow.
            condition: Called on each message on the host's thread; ``True``
                ends the wait.
            timeout: Real seconds, counted from this call, to wait at most.
                ``math.inf`` waits as long as the run runs.

        Raises:
            ValueError: As ``subscribe`` raises.
            TimeoutError: ``timeout`` passed first. The run keeps running.
            RuntimeError: The run shut down normally first.
            NetworkError: The run stopped because a node raised.
        """
        messages = self.subscribe(port, timeout)
        # Valid: subscribe has checked the port.
        name = _host(self).label(port)
        while True:
            try:
                message, time = next(messages)
            except StopIteration:
                break
            except TimeoutError:
                # Name this call, not the subscribe it is built on. Only the
                # iteration's own timeout: one that condition raises goes as it is.
                raise TimeoutError(
                    f"no message of the port {name!r} met the condition of "
                    f"wait_for within {timeout} s; the run keeps running"
                ) from None
            if condition(message):
                return message, time
        raise RuntimeError(
            f"the run shut down before the port {name!r} sent a message that meets "
            "the condition"
        )

    def state_dict(self, timeout: float = 30.0) -> StateDict:
        """Copy the state of the run at one time, while it runs or after it ended
        normally.

        While it runs, this pauses the run's timer and nodes, copies the state
        and resumes, and returns once the run has resumed. It waits at most
        ``timeout`` for every node to pause. After a normal end,
        this copies the state as the nodes left it, calling each node's
        ``state_dict`` on this thread (an exception it raises reaches the caller
        as it is). The tree's timer holds the time at which ``shutdown`` asked
        the run to stop, as if the run had paused then (or a later update or send
        of an ``on_stop``), so that a restored run goes on from there without the
        time that passed after the end.

        The tree is a copy (later updates of the nodes do not change it), a
        ``StateDict`` (a ``dict``) that pickles; saving it is up to the host.
        Pass it to ``mujou.run(..., state=tree)`` to restore it::

            {
                "nodes": {"body.l1": ..., ...},  # each node's state_dict()
                "slots": {"body.l1.next/out": (value, time_ns, sent), ...},
                "last_update_ns": {"body.l1": int, ...},
                "timer": {"time_ns": int, "scale": float},
            }

        A node is keyed by its attribute path from the root. A slot is keyed by
        the path of the port through its owning node (``"body.l1.prev/in"``, not
        an exposed name such as ``"prev/in"``) and holds its latest message:
        ``/out`` what ``read`` returns, ``/ch0``, ``/ch1``, ... each channel, and
        ``/in`` what the host wrote. A node run alone is the root: it is keyed
        ``""`` and its slots ``"prev/in"``, ``"next/out"``, ....

        The tensors of the tree stay on their nodes' devices. ``tree.cpu()``
        moves them all to the CPU, for example before saving the tree.

        Args:
            timeout: Real seconds to wait for every node to pause (the end of
                its current ``on_update``): finite and > 0. After the end there
                is nothing to wait for, and it is only checked.

        Raises:
            RuntimeError: This is called from a node's hook.
            ValueError: ``timeout`` is not finite and > 0.
            PauseTimeoutError: A node did not pause within ``timeout``; no state
                is taken and the run keeps running.
            NetworkError: The run stopped because a node raised; its state is
                not kept.
        """
        host = _host(self)
        if not (math.isfinite(timeout) and timeout > 0):
            raise ValueError(f"timeout must be finite and > 0 seconds, got {timeout}")
        return host.state_dict(timeout)

    def shutdown(self, timeout: float = math.inf) -> None:
        """Stop every node and return once every ``on_stop`` has returned.

        Does nothing after a normal end.

        Args:
            timeout: Real seconds to wait for the end. ``math.inf`` waits as
                long as it takes.

        Raises:
            ValueError: ``timeout`` is negative or NaN; nothing was asked.
            TimeoutError: The run did not stop within ``timeout``. The stop stays
                asked: the nodes take no more updates and the run goes on
                stopping in the background (an ``on_stop`` may still be
                running). ``join`` waits for the end.
            NetworkError: The run stopped because a node raised.
        """
        _host(self).shutdown(timeout)

    def join(self, timeout: float = math.inf) -> None:
        """Wait until the run stops, without asking it to stop.

        It returns once another thread called ``shutdown`` and every ``on_stop``
        has returned; if a node raises, it raises instead. If nothing stops the
        run and ``timeout`` is ``math.inf``, it never returns.

        Args:
            timeout: Real seconds to wait for the end. ``math.inf`` waits as
                long as it takes.

        Raises:
            ValueError: ``timeout`` is negative or NaN.
            TimeoutError: The run did not stop within ``timeout``. Nothing
                changes: a running run keeps running.
            NetworkError: The run stopped because a node raised.
        """
        _host(self).join(timeout)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Shut the run down. When the body raised, its exception goes on and a
        ``NetworkError`` of the shutdown is dropped (the node's exception is in
        the log)."""
        if exc is None:
            self.shutdown()
            return
        try:
            self.shutdown()
        except _native.NetworkError:
            pass


class ManualRun[T: Timer = VirtualTimer](Run):
    """A run whose nodes update only when ``step`` lets them, for tests.

    ``mujou.testing.run`` returns it. The nodes start and stop as in any run, but
    between updates each waits for ``step``. On a ``VirtualTimer`` (the default),
    every ``dt`` is then exactly the time the test advanced with ``timer.advance``,
    and the result follows the discrete algorithm of the chosen order.
    """

    @property
    @override
    def timer(self) -> T:
        """The run's timer: the ``VirtualTimer`` that ``mujou.testing.run`` made,
        or the timer given to it."""
        # The T that start_manual gave start_run.
        return cast(T, self._timer)

    def step(self, *nodes: Node) -> None:
        """Update each of ``nodes`` once, one after another in the given order (by
        default every node, in the order of the tree), and return when done.

        ``step`` does not move the time: advance the timer first. A node whose
        time on the run's timer has not moved since its last update is skipped
        (it would see ``dt = 0``): a ``step`` without an ``advance`` before it
        updates nothing, and a node given twice in one ``step`` updates once.

        Raises:
            ValueError: A node is not in this run.
            RuntimeError: The run has stopped normally.
            NetworkError: A node raised; the run has stopped.
        """
        _host(self).step(nodes)


# P: as in Run.write.
def run[P: Port[Any, Any, Any]](
    component: Node | Network,
    *components: Node | Network,
    timer: Timer | None = None,
    inputs: Mapping[P, torch.Tensor | Mapping[str, object]] | None = None,
    state: Mapping[str, Any] | None = None,
    workers: int | None = None,
) -> Run:
    """Build the given nodes and networks and run them in the background; return
    once every node has started.

    One component alone is the root of the run: a network runs as it is (its
    nodes at ``"l1"``, ...), and a node is at the path ``""`` with its ports at
    ``"prev"``, ``"next"``, .... Several run as ``Group(component,
    *components)``, nodes and networks mixed or not, so their paths start with
    their positions: ``"0"`` for a node given first, ``"1.l1"``, ... for the
    nodes of a network given second. Either way the host writes and reads the
    ports themselves (``run.write({node.prev: x})``).

    Every node's ``on_start`` (and, with ``state``, the node's
    ``load_state_dict``) has returned, and the run's timer runs when this
    returns. A network or node runs once.

    The run's nodes share a pool of ``workers`` threads, which it creates for
    itself; a node with ``dedicated_thread=True`` runs on a thread of its own
    instead. A worker runs one hook of one node at a time, the updates that are
    due earliest first. A hook that blocks holds its worker, so the other
    nodes' ``dt`` grows meanwhile.

    Args:
        component: The node or network to run.
        components: More nodes and networks to run with it.
        timer: The time axis of the run: every node's ``dt``, every message's
            time and every pause of the run is on it. ``None`` is the global
            timer, which ``mujou.time`` reads and every run without a timer of
            its own shares.
        inputs: The initial values of unconnected single-peer ports that
            receive, as ``Run.write`` takes them. Written after ``state``, they
            replace its values and carry its time.
        state: A tree from ``Run.state_dict`` to restore: the ``StateDict``,
            or any mapping of the same items (such as a plain ``dict``). The slots, the nodes'
            states (through each node's ``load_state_dict``, after its
            ``on_start``), their last update times and the timer continue from
            the time the tree was taken. Build the networks as the ones the tree
            came from: the same node paths and slots.
        workers: The number of worker threads of the run (>= 1). ``None`` is
            as many as the CPUs the process may use (the logical cores, or
            fewer when the CPU affinity or a quota limits them), or half of
            them (at least 1) while the GIL is enabled at run time
            (``sys._is_gil_enabled()``), never more than the nodes on the
            workers. With no such node, the run has no
            worker.

    Example, with the node ``Error`` of the example of ``Run``::

        node = Error()
        with mujou.run(node, inputs={node.prev: torch.ones(2)}) as run:
            mujou.time.sleep(0.1)
            error = run.read(node.next)

    Raises:
        RuntimeError: A network or node has already run (``"Leaky has already
            run; create a new instance to run again"``), or part of one belongs
            to another network that was built but has not run; or this is called
            from a node's hook.
        TypeError: A value is neither a ``Node`` nor a ``Network``, or
            ``workers`` is not an ``int`` (a ``bool`` included).
        ValueError: ``workers`` is less than 1 (or more than ``sys.maxsize``).
        ValueError, TypeError: As ``Network.build``, ``Group`` and
            ``Run.write`` (for ``inputs``) raise; ``state`` does not match the
            networks, or the timer it would set back has another run on it.
            Nothing has started: the same networks, given the same way, may run
            once the cause is gone.
        NetworkError: A node raised while starting; the run has stopped.
    """
    return start_run(
        Run,
        (component, *components),
        timer=timer,
        inputs=inputs,
        state=state,
        manual=False,
        workers=workers,
    )


def start_run[R: Run](
    kind: type[R],
    components: Sequence[Node | Network],
    *,
    timer: Timer | None,
    inputs: Mapping[Any, object] | None,
    state: Mapping[str, Any] | None,
    manual: bool,
    workers: int | None,
) -> R:
    """Run ``components`` as ``mujou.run`` does and return a ``kind`` for it."""
    if in_hook():
        raise _from_hook()
    if workers is not None:
        # The annotation is not checked at runtime.
        if isinstance(workers, bool) or not isinstance(workers, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"workers must be an int or None, got {type(workers).__name__}"
            )
        # The upper bound: the core counts the workers in a pointer-sized int.
        if not 1 <= workers <= sys.maxsize:
            raise ValueError(
                f"workers must be at least 1 and at most {sys.maxsize}, got {workers}"
            )
    for given in components:
        # The annotation is not checked at runtime.
        if not isinstance(given, Node | Network):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"mujou.run() takes nodes and networks, got {type(given).__name__}"
            )
    run_timer = GLOBAL if timer is None else timer
    # ours: a tree mujou.run built only for this run, undone if the run is refused.
    match components:
        case [Network() as network]:
            graph, ours = network.build(), False
        case [Node() as node]:
            graph, ours = build_node(node), True
        case _:
            graph, ours = RunGroup(*components).build(), True

    def go() -> Session:
        # Made only once the graph may run: it binds the ports to its slots.
        session = Session(graph, run_timer)
        session.start(manual=manual, state=state, inputs=inputs, workers=workers)
        return session

    try:
        session = start_once(graph, go)
    except Exception as e:
        # Refused before any node started: the tree is ours and is dropped, so its
        # build must not keep what was given from being built again.
        if ours and not isinstance(e, _native.NetworkError):
            unbuild(graph)
        raise
    handle = object.__new__(kind)
    handle._session = session  # pyright: ignore[reportPrivateUsage]  # Run's own module
    handle._timer = run_timer  # pyright: ignore[reportPrivateUsage]  # Run's own module
    return handle


def start_manual[T: Timer](
    components: Sequence[Node | Network],
    *,
    timer: T,
    inputs: Mapping[Any, object] | None,
    state: Mapping[str, Any] | None,
) -> ManualRun[T]:
    """Run ``components`` as ``mujou.testing.run`` does, on ``timer``."""
    # The class itself: a subscripted generic is not a class to instantiate.
    return start_run(
        cast(type[ManualRun[T]], ManualRun),
        components,
        timer=timer,
        inputs=inputs,
        state=state,
        manual=True,
        workers=None,
    )


def _host(run: Run) -> Session:
    if in_hook():
        raise _from_hook()
    return run._session  # pyright: ignore[reportPrivateUsage]  # Run's own module


def _from_hook() -> RuntimeError:
    return RuntimeError(
        "a node's hook cannot call the host methods of a run; a node talks only "
        "through its own ports"
    )
