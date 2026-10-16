"""Timers: the time axes runs measure their dynamics on (``specs/execution-model.md``).

``Timer`` wraps the native timer and shows only what a user needs; the framework
reaches the native one through ``native``.
"""

from typing import Final, final, override

from mujou._core import _native


class Timer:
    """A time axis that a run's nodes measure their ``dt`` on.

    ``Timer()`` is a new time axis that starts at 0 on the real clock, at scale 1.
    Give it to ``mujou.run(..., timer=...)`` to run a network on a time of its
    own: a pause of that run (as ``Run.state_dict`` takes) then stops only this
    timer, not the global one that every other run uses by default. The nodes of
    the run read it as ``Node.timer``.

    Times are ``Timestamp``. Use ``sleep`` instead of the standard
    ``time.sleep`` wherever a duration should follow this time, e.g. to present
    an input to a run on this timer.

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
        """
        self._timer.sleep(seconds)

    def get_time_scale(self) -> float:
        """Return how many seconds of this timer pass per real second."""
        return self._timer.time_scale()

    def set_time_scale(self, scale: float) -> None:
        """Set how many seconds of this timer pass per real second.

        Raises:
            ValueError: ``scale`` is not finite and > 0.
        """
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
                takes its state).
        """
        self._timer.advance(seconds)


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
