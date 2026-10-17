"""Timers: the time axes runs measure their dynamics on (``specs/execution-model.md``).

``Timer`` wraps the native timer and shows only what the host needs; the framework
reaches the native one through ``native``. A node sees its run's timer only
through a ``ReadOnlyTimer``, which reads the time and changes nothing.
"""

import itertools
from collections.abc import Callable
from typing import Final, final, override

from mujou._core import _native
from mujou._core._native import Timestamp  # by name: the API reference shows it
from mujou._core.hooks import in_hook


class Timer:
    """A time axis that a run's nodes measure their ``dt`` on.

    ``Timer()`` is a new time axis that starts at 0 on the real clock, at scale 1.
    Give it to ``mujou.run(..., timer=...)`` to run a network on a time of its
    own: a pause of that run (as ``Run.state_dict`` takes) then stops only this
    timer, not the global one that every other run uses by default. The nodes of
    the run read its time through ``Node.timer``, a ``ReadOnlyTimer``.

    Times are ``Timestamp``. Use ``sleep`` instead of the standard
    ``time.sleep`` wherever a duration should follow this time, e.g. to present
    an input to a run on this timer.

    Only the host controls a timer: ``sleep``, ``set_time_scale``, ``pause`` and
    ``resume`` raise ``RuntimeError`` in a node's hook, where a node would wait
    for (or change the ``dt`` of) the other nodes.

    ``pause`` stops the time of every run on the timer at one time T until
    ``resume``. Take the states of several runs on one timer while it is paused,
    so that all of them are of the same time T: taken one after another on a
    running timer, each would be of a later time.

    Example, with the node ``Error`` of the example of ``mujou.Run``::

        timer = mujou.time.Timer()
        node = Error()
        with mujou.run(node, timer=timer) as run:
            run.write({node.prev: torch.ones(2)})
            timer.sleep(0.1)  # 0.1 s of the run's time
            error, t = run.read(node.next)  # t is on this timer
            assert t <= timer.now()
    """

    _timer: _native.Timer

    def __init__(self) -> None:
        self._timer = _native.Timer()

    def now(self) -> Timestamp:
        """Return the time now."""
        return self._timer.now()

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds`` of this timer's time, following the scale and the
        pauses of the runs on it.

        Raises:
            ValueError: ``seconds`` is negative or not finite.
            RuntimeError: This is called from a node's hook.
        """
        host_only("Timer.sleep")
        self._timer.sleep(seconds)

    def get_time_scale(self) -> float:
        """Return how many seconds of this timer pass per real second."""
        return self._timer.time_scale()

    def pause(self) -> None:
        """Stop the time of this timer until ``resume``.

        While paused, the time does not move: the nodes of the runs on this
        timer do not update (they wait until the time moves, and the first
        ``dt`` after ``resume`` does not count the pause), ``sleep`` does not
        return until resumed, and ``VirtualTimer.advance`` raises
        ``RuntimeError``. ``Run.state_dict`` still works, so take the states of
        several runs on this timer at one time like this::

            timer.pause()
            try:
                trees = [run.state_dict() for run in runs]  # all at one time
            finally:
                timer.resume()

        This is a switch, not a count: pausing twice needs one ``resume``. A
        run that takes its state holds the time paused on its own, apart from
        this switch, so ``resume`` does not end that and its end does not end
        this pause.

        Raises:
            RuntimeError: This is called from a node's hook.
        """
        host_only("Timer.pause")
        self._timer.pause()

    def resume(self) -> None:
        """Turn off the switch of ``pause``: the time moves again from where it
        stopped, unless a run on this timer is taking its state.

        Raises:
            RuntimeError: This is called from a node's hook.
        """
        host_only("Timer.resume")
        self._timer.resume()

    def is_paused(self) -> bool:
        """Return whether the time is stopped, by ``pause`` or by a run on this
        timer that is taking its state. The virtual clock of a ``VirtualTimer``
        alone does not count as paused."""
        return self._timer.is_paused()

    def set_time_scale(self, scale: float) -> None:
        """Set how many seconds of this timer pass per real second.

        Raises:
            ValueError: ``scale`` is not finite and > 0.
            RuntimeError: This is called from a node's hook.
        """
        host_only("Timer.set_time_scale")
        self._timer.set_time_scale(scale)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(now={self.now()!r})"


@final
class VirtualTimer(Timer):
    """A timer for tests: its time moves only when ``advance`` moves it.

    It starts at 0. The scale is kept but does not apply to ``advance``.
    ``mujou.testing.run`` runs a network on a new one (``ManualRun.timer``) unless
    given a timer; step the nodes by hand, so that every ``dt`` is exactly what
    the test advanced. Make one to share it between runs.

    Example, with the node ``Error`` of the example of ``mujou.Run``::

        timer = VirtualTimer()
        with mujou.testing.run(Error(), timer=timer) as run:
            timer.advance(0.001)
            run.step()  # every node updates once with dt = 0.001
    """

    def __init__(self) -> None:
        super().__init__()
        self._timer.use_virtual_clock(True)
        # The real clock ran since the timer was made; start at exactly 0.
        self._timer.load_state_dict({"time_ns": 0, "scale": 1.0})

    def advance(self, seconds: float) -> None:
        """Move the time forward by ``seconds``.

        Raises:
            ValueError: ``seconds`` is negative or not finite.
            RuntimeError: The timer is paused (by ``pause``, or by a run on it
                that takes its state), or this is called from a node's hook.
        """
        host_only("VirtualTimer.advance")
        self._timer.advance(seconds)


@final
class ReadOnlyTimer:
    """The reading side of a ``Timer``: what a node sees of its run's timer as
    ``Node.timer``.

    ``ReadOnlyTimer(timer)`` takes the reading methods of ``timer`` (``now``
    and ``get_time_scale``) and nothing else: it does not hold ``timer``, so
    nothing reached from it sets the scale, advances the clock or waits on the
    time. A node thus cannot change the ``dt`` of the other nodes.

    Once the host has seen a run end normally, each node's ``timer`` is replaced
    by a ``ReadOnlyTimer`` that stays at the time and the scale at which the run
    stopped, the time of the state. One that a node kept from before the end
    still reads the running timer.

    Example::

        class Clock(Node):  # sends the time of its run's timer, in seconds
            def __init__(self) -> None:
                super().__init__()
                self.next = NextPort(send=TensorSpec((), torch.float64))

            @override
            def on_update(self, dt: float) -> None:
                super().on_update(dt)
                t = self.timer.now()  # a Timestamp on the run's timer
                self.next.send(torch.tensor(float(t), dtype=torch.float64))
    """

    now: Callable[[], Timestamp]
    """Return the time now on the timer."""

    get_time_scale: Callable[[], float]
    """Return how many seconds of the timer pass per real second."""

    def __init__(self, timer: Timer) -> None:
        # The native reader has no method that changes the timer, and no way to
        # reach one: the bound methods below lead only to it.
        reader = timer._timer.reader()  # pyright: ignore[reportPrivateUsage]  # this module's own
        self.now = reader.now
        self.get_time_scale = reader.time_scale

    @override
    def __repr__(self) -> str:
        return f"ReadOnlyTimer(now={self.now()!r})"


def stopped(at: Timestamp, scale: float) -> ReadOnlyTimer:
    """Return a ``ReadOnlyTimer`` whose time stays at ``at`` and scale at
    ``scale`` (a run that has ended)."""
    view = object.__new__(ReadOnlyTimer)
    # Methods of iterators that repeat the two values: nothing of the timer is
    # kept, and (unlike a Python function) no module globals hang on them.
    view.now = itertools.repeat(at).__next__
    view.get_time_scale = itertools.repeat(scale).__next__
    return view


def host_only(what: str) -> None:
    """Raise ``RuntimeError`` in a node's hook, where ``what`` must not run.

    Raises:
        RuntimeError: The calling thread is running a node's hook.
    """
    if in_hook():
        raise RuntimeError(
            f"a node's hook cannot call {what}: only the host controls the time; "
            "a node reads it with self.timer.now()"
        )


def _wrap(timer: _native.Timer) -> Timer:
    wrapped = Timer.__new__(Timer)
    wrapped._timer = timer  # pyright: ignore[reportPrivateUsage]  # Timer's own module
    return wrapped


# The global timer, which every run not given a timer of its own uses. The
# functions of `mujou.time` read the same native timer.
GLOBAL: Final = _wrap(_native.timer)


def native(timer: Timer) -> _native.Timer:
    """Return the native timer that ``timer`` wraps."""
    return timer._timer  # pyright: ignore[reportPrivateUsage]  # Timer's own module
