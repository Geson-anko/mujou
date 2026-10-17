import contextlib
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import pytest
import torch

import mujou
import mujou.testing
from mujou import Network, NetworkError, Node, Port, Run
from mujou._core._native import GlobalTimer, timer
from mujou.testing import ManualRun, VirtualTimer
from mujou.time import Timer

# P: a dict built in a test has a precise key type, which a Mapping keyed by
# Port[Any, Any, Any] would reject (Mapping keys are invariant).
type Inputs[P] = Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]]

H_NS = 1_000_000
H = H_NS / 1e9  # The step of the virtual clock the `step` fixture advances, in seconds.


@pytest.fixture
def virtual_timer() -> Iterator[GlobalTimer]:
    """Switch the global timer to the virtual clock at time 0 for one test.

    Only for the tests of the global timer itself (``mujou.time`` and the native
    controls): nothing public switches it. Runs take a ``virtual`` timer instead.
    """
    timer.use_virtual_clock(True)
    timer.load_state_dict({"time_ns": 0, "scale": 1.0})
    try:
        yield timer
    finally:
        timer.resume()
        timer.set_time_scale(1.0)
        timer.use_virtual_clock(False)


@pytest.fixture
def virtual() -> VirtualTimer:
    """A new timer on the virtual clock at time 0, for one test."""
    return VirtualTimer()


class StartManual(Protocol):
    def __call__[P: Port[Any, Any, Any]](
        self,
        component: Node | Network,
        *components: Node | Network,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> ManualRun[Any]: ...


class Start(Protocol):
    def __call__[P: Port[Any, Any, Any]](
        self,
        component: Node | Network,
        *components: Node | Network,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
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
    """``mujou.testing.run`` on the test's ``virtual`` timer unless given another,
    shut down at the end of the test."""

    def start[P: Port[Any, Any, Any]](
        component: Node | Network,
        *components: Node | Network,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> ManualRun[Any]:
        run = mujou.testing.run(
            component,
            *components,
            timer=virtual if timer is None else timer,
            inputs=inputs,
            state=state,
        )
        started.append(run)
        return run

    return start


@pytest.fixture
def start(started: list[Run]) -> Start:
    """``mujou.run`` (free running, on the global timer unless given another), shut
    down at the end of the test."""

    def start[P: Port[Any, Any, Any]](
        component: Node | Network,
        *components: Node | Network,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> Run:
        run = mujou.run(component, *components, timer=timer, inputs=inputs, state=state)
        started.append(run)
        return run

    return start


class Step(Protocol):
    def __call__(self, run: ManualRun[Any], *order: Node) -> None: ...


@pytest.fixture
def step() -> Step:
    """Advance the run's ``VirtualTimer`` by ``H`` and update the nodes once, in
    ``order`` (every node, in the order of the tree, if none is given)."""

    def step(run: ManualRun[Any], *order: Node) -> None:
        timer = run.timer
        assert isinstance(timer, VirtualTimer)
        timer.advance(H)
        run.step(*order)

    return step


@pytest.fixture
def accelerator() -> torch.device:
    """Device 0 of torch's current accelerator (CUDA, MPS, ...); skips the test
    when none is available."""
    found = torch.accelerator.current_accelerator(check_available=True)
    if found is None:
        pytest.skip("needs an accelerator")
    return torch.device(found.type, 0)


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
