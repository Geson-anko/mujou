import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import pytest
import torch

import mujou
import mujou.testing
from mujou import Network, NetworkError, Node, Port, Run
from mujou.testing import ManualRun
from mujou.time import Timer, VirtualTimer

# P: a dict built in a test has a precise key type, which a Mapping keyed by
# Port[Any, Any] would reject (Mapping keys are invariant).
type Inputs[P] = Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]]

H_NS = 1_000_000
H = H_NS / 1e9  # The step the tests advance a virtual clock by, in seconds.


@pytest.fixture
def virtual() -> VirtualTimer:
    """A new timer on the virtual clock at time 0, for one test."""
    return VirtualTimer()


class StartManual(Protocol):
    def __call__[P: Port[Any, Any]](
        self,
        component: Node | Network,
        *components: Node | Network,
        timer: VirtualTimer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> ManualRun: ...


class Start(Protocol):
    def __call__[P: Port[Any, Any]](
        self,
        component: Node | Network,
        *components: Node | Network,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> Run: ...


@pytest.fixture
def started() -> Iterator[list[Run]]:
    """The runs a test started, shut down at its end.

    A running network keeps running when dropped (until the interpreter exits), so
    a test must stop what it starts.
    """
    runs: list[Run] = []
    try:
        yield runs
    finally:
        for run in runs:
            with contextlib.suppress(NetworkError):
                run.shutdown()


@pytest.fixture
def start_manual(virtual: VirtualTimer, started: list[Run]) -> StartManual:
    """``mujou.testing.run`` on the test's ``virtual`` timer unless given another
    (or a state, whose run makes its own), shut down at the end of the test."""

    def start[P: Port[Any, Any]](
        component: Node | Network,
        *components: Node | Network,
        timer: VirtualTimer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> ManualRun:
        if timer is None and state is None:
            timer = virtual
        run = mujou.testing.run(
            component, *components, timer=timer, inputs=inputs, state=state
        )
        started.append(run)
        return run

    return start


@pytest.fixture
def start(started: list[Run]) -> Start:
    """``mujou.run`` (free running, on the global timer unless given another), shut
    down at the end of the test."""

    def start[P: Port[Any, Any]](
        component: Node | Network,
        *components: Node | Network,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> Run:
        run = mujou.run(
            component,
            *components,
            timer=timer,
            inputs=inputs,
            state=state,
            workers=workers,
        )
        started.append(run)
        return run

    return start


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test marked ``cuda(n)`` (``n`` defaults to 1) unless ``n`` CUDA
    devices are available; the largest ``n`` of its marks counts."""
    needed = max(
        (
            mark.args[0] if mark.args else mark.kwargs.get("n", 1)
            for mark in item.iter_markers("cuda")
        ),
        default=0,
    )
    if needed and (not torch.cuda.is_available() or torch.cuda.device_count() < needed):
        pytest.skip(f"needs {needed} CUDA device(s)")
