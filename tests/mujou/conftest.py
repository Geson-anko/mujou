import contextlib
import gc
import os
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import pytest
import torch

import mujou
from mujou import ManualRun, Network, NetworkError, Node, Port, Run
from mujou.time import Timer, VirtualTimer

# P: a dict built in a test has a precise key type, which a Mapping keyed by
# Port[Any, Any] would reject (Mapping keys are invariant).
type Inputs[P] = Mapping[P, torch.Tensor | Mapping[str, torch.Tensor]]

# cuBLAS keeps a workspace for each pair of a thread's handle and a stream, of
# 32 MiB by default on recent GPUs, and never gives it back. Every node on CUDA
# updates on a stream of its own from worker threads, so the workspaces alone
# would fill the budget below: the tests' matrices are small, and 128 KiB each
# is plenty (read when CUDA makes the first handle, after this module loads).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "128")  # KiB

H_NS = 1_000_000
H = H_NS / 1e9  # The step the tests advance a virtual clock by, in seconds.


@pytest.fixture
def virtual() -> VirtualTimer:
    """A new timer on the virtual clock at time 0, for one test."""
    return VirtualTimer()


class StartManual(Protocol):
    def __call__[P: Port[Any, Any]](
        self,
        root: Node | Network,
        *,
        timer: VirtualTimer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> ManualRun: ...


class Start(Protocol):
    def __call__[P: Port[Any, Any]](
        self,
        root: Node | Network,
        *,
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
    """``mujou.run_manually`` on the test's ``virtual`` timer unless given another
    (or a state, whose run makes its own), shut down at the end of the test."""

    def start[P: Port[Any, Any]](
        root: Node | Network,
        *,
        timer: VirtualTimer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> ManualRun:
        if timer is None and state is None:
            timer = virtual
        run = mujou.run_manually(
            root, timer=timer, inputs=inputs, state=state, workers=workers
        )
        started.append(run)
        return run

    return start


@pytest.fixture
def start(started: list[Run]) -> Start:
    """``mujou.run`` (free running, on the global timer unless given another), shut
    down at the end of the test."""

    def start[P: Port[Any, Any]](
        root: Node | Network,
        *,
        timer: Timer | None = None,
        inputs: Inputs[P] | None = None,
        state: Mapping[str, Any] | None = None,
        workers: int | None = None,
    ) -> Run:
        run = mujou.run(root, timer=timer, inputs=inputs, state=state, workers=workers)
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
    if needed:
        # The caching allocator may hold at most CUDA_BUDGET on each device, so a
        # test that needs more fails with an out-of-memory error instead of
        # crowding the machine (with the context, the tests stay under 2 GiB).
        for index in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(index).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, CUDA_BUDGET / total), index
            )


CUDA_BUDGET = 1 << 30  # Bytes of the caching allocator on each device.


@pytest.fixture(autouse=True)
def release_cuda(request: pytest.FixtureRequest) -> Iterator[None]:
    """Give back to the device what a CUDA test cached (its tensors, the memory
    pools of its CUDA graphs, the workspaces of cuBLAS) once its runs are shut
    down, so the memory of the tests does not add up."""
    yield
    if request.node.get_closest_marker("cuda") and torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()  # pyright: ignore[reportPrivateUsage]
        torch.cuda.empty_cache()
