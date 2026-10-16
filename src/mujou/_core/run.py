"""Running networks: ``mujou.run`` and the ``Run`` it returns (``specs/network.md``)."""

import math
from collections.abc import Iterator, Mapping, Sequence
from types import TracebackType
from typing import Any, Never, Self, cast

import torch

from mujou._core import _native
from mujou._core.network import Compose, Network, start_once
from mujou._core.node import Node, in_hook
from mujou._core.port import Port
from mujou._core.session import Session
from mujou._core.timer import GLOBAL, Timer


class Run:
    """A run of networks, as ``mujou.run`` returns it: what the host uses to talk
    to the running nodes.

    The nodes run in the background, each on its own thread; the host's thread
    only writes inputs, reads outputs and takes the state:

    - ``write`` inputs, ``read`` / ``subscribe`` outputs, take a ``state_dict``.
    - ``shutdown()`` to stop the run, or ``join()`` to wait until it stops.
    - After a normal end: take the final ``state_dict``.

    A ``with`` statement shuts the run down at its end. These methods are for
    the host: calling them from a node's hook raises ``RuntimeError``.

    If a node raises, the whole run stops. If that happens while ``mujou.run``
    starts the nodes, ``mujou.run`` raises ``NetworkError``; otherwise every later
    call of ``write``, ``read``, ``subscribe`` (and its iterator), ``state_dict``,
    ``shutdown`` and ``join`` raises it. Its ``__cause__`` is the node's
    exception. The state at the exception is not kept.

    A run keeps running when the host drops it, until ``shutdown`` or, at the
    latest, the interpreter's exit (which waits for ``on_stop``). At the exit,
    the runs start stopping once the main thread has ended, before the
    interpreter joins the non-daemon threads, and the exit waits until they have
    stopped.

    Example::

        net = Classifier()
        with mujou.run(net, inputs={net.prev: x0, net.teacher: y0}) as run:
            run.write({net.prev: x, net.teacher: y})  # switch together
            mujou.time.sleep(0.1)  # present the input for 0.1 s of global time
            prediction, t = run.read(net.head.next)
        state = run.state_dict()  # the state at the end, to save
    """

    _session: Session

    def __init__(self) -> Never:
        raise TypeError("a Run is made by mujou.run()")

    # P: a dict built beforehand has a precise key type, which a Mapping keyed by
    # Port[Any, Any, Any] would reject (Mapping keys are invariant).
    def write[P: Port[Any, Any, Any]](
        self, values: Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]]
    ) -> None:
        """Write messages to unconnected single-peer ports that receive.

        The values are detached and copied. All of them switch together and carry
        the same time: at any one moment, a node sees either all of them or none.
        (A node that reads two of them one after the other may still see a later
        write in between; compare the times to tell.) Any port of the tree can be
        written, exposed by the root or not.

        Raises:
            RuntimeError: The run has stopped normally (no node reads what is
                written then).
            ValueError: A port is outside the run, connected, takes many peers,
                or receives nothing; or a shape or the keys differ.
            TypeError: A message or dtype differs from the spec.
            NetworkError: The run stopped because a node raised.
        """
        _host(self).write(values)

    def read[S](self, port: Port[S, Any, Any]) -> tuple[S, _native.Timestamp]:
        """Return the latest message ``port`` sent and the time it was sent at.

        Any port that sends can be read, so a connected port shows an inner
        node. Only ``send`` (to every peer) updates the value read here; before
        the first one it is zeros (or the value restored from ``state``). The
        tensors are new copies on the CPU. The time is on the run's timer.

        Raises:
            ValueError: ``port`` is outside the run or sends nothing.
            NetworkError: The run stopped because a node raised.
        """
        return cast(tuple[S, _native.Timestamp], _host(self).read(port))

    def subscribe[S](
        self, port: Port[S, Any, Any]
    ) -> Iterator[tuple[S, _native.Timestamp]]:
        """Iterate over the messages ``port`` sends, with their times.

        The first step yields the latest message at the time of this call (zeros
        or the restored value before the first ``send``) without waiting. Each
        later step waits until a newer message is sent and yields the latest;
        messages sent in between are skipped. Only the consumer waits: the
        sending node never does. The iterator ends when the run shuts down
        normally. Shut the run down before the program ends: a thread iterating
        here ends only with the run.

        Raises:
            ValueError: ``port`` is outside the run or sends nothing.
            NetworkError: The run stopped because a node raised (also from the
                iterator).
        """
        return cast(Iterator[tuple[S, _native.Timestamp]], _host(self).subscribe(port))

    def state_dict(self) -> dict[str, Any]:
        """Copy the state of the run at one time, while it runs or after it ended
        normally.

        While it runs, this pauses the run's timer and nodes, copies the state
        and resumes, and returns once the run has resumed. After a normal end,
        this copies the state as the nodes left it, calling each node's
        ``state_dict`` on this thread (an exception it raises reaches the caller
        as it is). The tree's timer holds the time at which ``shutdown`` asked
        the run to stop, as if the run had paused then (or a later update or send
        of an ``on_stop``), so that a restored run goes on from there without the
        time that passed after the end.

        The tree is a copy (later updates of the nodes do not change it) and
        pickles; saving it is up to the host. Pass it to ``mujou.run(...,
        state=tree)`` to restore it::

            {
                "nodes": {"body.l1": ..., ...},  # each node's state_dict()
                "slots": {"body.l1.next/out": (value, time_ns), ...},
                "last_update_ns": {"body.l1": int, ...},
                "timer": {"time_ns": int, "scale": float},
            }

        A node is keyed by its attribute path from the root. A slot is keyed by
        the path of the port through its owning node (``"body.l1.prev/in"``, not
        an exposed name such as ``"prev/in"``) and holds its latest message:
        ``/out`` what ``read`` returns, ``/ch0``, ``/ch1``, ... each channel, and
        ``/in`` what the host wrote.

        Raises:
            RuntimeError: This is called from a node's hook.
            PauseTimeoutError: A node did not pause within ``pause_timeout``; the
                run keeps running.
            NetworkError: The run stopped because a node raised; its state is
                not kept.
        """
        return _host(self).state_dict()

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


class ManualRun(Run):
    """A run whose nodes update only when ``step`` lets them, for tests.

    ``mujou.testing.run`` returns it. The nodes start and stop as in any run, but
    between updates each waits for ``step``. With a ``VirtualTimer``, every
    ``dt`` is then exactly the time the test advanced, and the result follows
    the discrete algorithm of the chosen order.
    """

    def step(self, *nodes: Node) -> None:
        """Update each of ``nodes`` once, one after another in the given order (by
        default every node, in the order of the tree), and return when done.

        A node whose time on the run's timer has not advanced since its last
        update is not updated. A node updated twice sees ``dt = 0`` the second
        time and is skipped.

        Raises:
            ValueError: A node is not in this run.
            NetworkError: A node raised; the run has stopped.
        """
        _host(self).step(nodes)


# P: as in Run.write.
def run[P: Port[Any, Any, Any]](
    *networks: Network,
    timer: Timer | None = None,
    inputs: Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]] | None = None,
    state: Mapping[str, Any] | None = None,
    pause_timeout: float = 30.0,
) -> Run:
    """Build ``networks`` and run them in the background; return once every node
    has started.

    One network runs as it is; several run as ``Compose(*networks)``, so their
    paths start with their positions (``"0.l1"``, ``"1.l1"``, ...). Every node's
    thread is set up, ``on_start`` (and, with ``state``, the node's
    ``load_state_dict``) has returned, and the run's timer runs when this
    returns. A network runs once.

    Args:
        networks: The networks to run.
        timer: The time axis of the run: every node's ``dt``, every message's
            time and every pause of the run is on it. ``None`` is the global
            timer, which ``mujou.time`` reads and every run without a timer of
            its own shares.
        inputs: The initial values of unconnected single-peer ports that
            receive, as ``Run.write`` takes them. Written after ``state``, they
            replace its values and carry its time.
        state: A tree from ``Run.state_dict`` to restore. The slots, the nodes'
            states (through each node's ``load_state_dict``, after its
            ``on_start``), their last update times and the timer continue from
            the time the tree was taken. Build the networks as the ones the tree
            came from: the same node paths and slots.
        pause_timeout: Real seconds that ``Run.state_dict`` waits for every node
            to pause (the end of its current ``on_update``).

    Example::

        net = Classifier()
        with mujou.run(net, inputs={net.prev: x}) as run:
            mujou.time.sleep(0.1)
            y, t = run.read(net.head.next)

    Raises:
        RuntimeError: A network has already run, or this is called from a node's
            hook.
        ValueError, TypeError: As ``Network.build``, ``Compose`` and
            ``Run.write`` (for ``inputs``) do; or ``state`` does not match the
            networks, or it cannot set the timer back because another run is
            running on it. Nothing started: the networks may run once the cause
            is gone.
        NetworkError: A node raised while starting; the run has stopped.
    """
    return start(
        Run,
        networks,
        timer=timer,
        inputs=inputs,
        state=state,
        pause_timeout=pause_timeout,
        manual=False,
    )


def start[R: Run](
    kind: type[R],
    networks: Sequence[Network],
    *,
    timer: Timer | None,
    inputs: Mapping[Any, object] | None,
    state: Mapping[str, Any] | None,
    pause_timeout: float,
    manual: bool,
) -> R:
    """Run ``networks`` as ``mujou.run`` does and return a ``kind`` for it."""
    if in_hook():
        raise _from_hook()
    network = networks[0] if len(networks) == 1 else Compose(*networks)
    graph = network.build()

    def go() -> Session:
        # Made only once the graph may run: it binds the ports to its slots.
        session = Session(graph, GLOBAL if timer is None else timer)
        session.start(
            manual=manual, state=state, inputs=inputs, pause_timeout=pause_timeout
        )
        return session

    session = start_once(graph, go)
    handle = object.__new__(kind)
    handle._session = session  # pyright: ignore[reportPrivateUsage]  # Run's own module
    return handle


def _host(run: Run) -> Session:
    if in_hook():
        raise _from_hook()
    return run._session  # pyright: ignore[reportPrivateUsage]  # Run's own module


def _from_hook() -> RuntimeError:
    return RuntimeError(
        "a node's hook cannot call the host methods of a run; a node talks only "
        "through its own ports"
    )
