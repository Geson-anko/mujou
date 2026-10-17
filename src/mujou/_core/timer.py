"""Timers: the time axes runs measure their dynamics on (``specs/execution-model.md``).

``Timer`` wraps the native timer and shows only what the host needs; the framework
reaches the native one through ``native``. A node sees its run's timer only
through a ``Clock``, which reads the time and changes nothing.
"""

from typing import Final, Never, cast, final, override

from mujou._core import _native


class Timer:
    """A time axis that a run's nodes measure their ``dt`` on.

    ``Timer()`` is a new time axis that starts at 0 on the real clock, at scale 1.
    Give it to ``mujou.run(..., timer=...)`` to run a network on a time of its
    own: a pause of that run (as ``Run.state_dict`` takes) then stops only this
    timer, not the global one that every other run uses by default. The nodes of
    the run read its time through ``Node.timer``, a read-only ``Clock``.

    Times are ``Timestamp``. Use ``sleep`` instead of the standard
    ``time.sleep`` wherever a duration should follow this time, e.g. to present
    an input to a run on this timer.

    Only the host controls a timer: ``sleep`` and ``set_time_scale`` raise
    ``RuntimeError`` in a node's hook, where a node would wait for (or change the
    ``dt`` of) the other nodes.

    Example::

        timer = mujou.time.Timer()
        with mujou.run(net, timer=timer) as run:
            run.write({net.prev: x})
            timer.sleep(0.1)  # 0.1 s of the run's time
            y, t = run.read(net.next)
    """

    _timer: _native.Timer

    def __init__(self) -> None:
        self._timer = _native.Timer()

    def now(self) -> _native.Timestamp:
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

    It starts at 0. The scale is kept but does not apply to ``advance``. Run a
    network on it with ``mujou.testing.run`` and step the nodes by hand, so that
    every ``dt`` is exactly what the test advanced.

    Example::

        timer = VirtualTimer()
        with mujou.testing.run(net, timer=timer) as run:
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
            RuntimeError: A run on this timer holds it paused (e.g. while it
                takes its state), or this is called from a node's hook.
        """
        host_only("VirtualTimer.advance")
        self._timer.advance(seconds)


@final
class Clock:
    """A read-only view of a run's timer: what a node sees as ``Node.timer``.

    It reads the time and changes nothing, so a node cannot change the ``dt`` of
    the other nodes nor wait on the time. After the run has ended normally (in
    the ``state_dict`` that the host takes then), it holds the time at which the
    run stopped, the time of the state.
    """

    _timer: _native.Timer | None
    _stopped_at: _native.Timestamp | None

    def __init__(self) -> Never:
        raise TypeError("a Clock is made by mujou.run; read it as Node.timer")

    def now(self) -> _native.Timestamp:
        """Return the time now on the run's timer."""
        # A clock holds either the run's timer or the time the run stopped at.
        if self._stopped_at is not None:
            return self._stopped_at
        return cast(_native.Timer, self._timer).now()

    @override
    def __repr__(self) -> str:
        return f"Clock(now={self.now()!r})"


def clock(timer: Timer) -> Clock:
    """Return a read-only view of ``timer``."""
    view = object.__new__(Clock)
    view._timer = timer._timer  # pyright: ignore[reportPrivateUsage]  # this module's own
    view._stopped_at = None  # pyright: ignore[reportPrivateUsage]
    return view


def stopped_clock(at: _native.Timestamp) -> Clock:
    """Return a view whose time stays at ``at`` (a run that has ended)."""
    view = object.__new__(Clock)
    view._timer = None  # pyright: ignore[reportPrivateUsage]  # this module's own
    view._stopped_at = at  # pyright: ignore[reportPrivateUsage]
    return view


def host_only(what: str) -> None:
    """Raise ``RuntimeError`` in a node's hook, where ``what`` must not run.

    Raises:
        RuntimeError: The calling thread is running a node's hook.
    """
    # Imported here: node imports this module.
    from mujou._core.node import in_hook

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
