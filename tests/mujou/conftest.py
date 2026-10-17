import contextlib
from collections.abc import Iterator
from typing import Protocol

import pytest

import mujou
from mujou import Network, NetworkError
from mujou._core._native import GlobalTimer, timer


@pytest.fixture
def virtual_timer() -> Iterator[GlobalTimer]:
    """Switch the global timer to the virtual clock at time 0 for one test."""
    timer.use_virtual_clock(True)
    timer.load_state_dict({"time_ns": 0, "scale": 1.0})
    try:
        yield timer
    finally:
        timer.resume()
        timer.set_time_scale(1.0)
        timer.use_virtual_clock(False)


class RunNetwork(Protocol):
    def __call__[N: Network](
        self,
        network: N,
        *,
        _manual: bool = False,
    ) -> N: ...


@pytest.fixture
def run_network() -> Iterator[RunNetwork]:
    """``mujou.run`` that shuts the networks down at the end of the test.

    A running network keeps running when dropped (until the interpreter exits), so
    a test must stop what it starts. Request it after ``virtual_timer`` so that
    the networks stop before the timer is reset: ``virtual_timer`` turns off the
    host's pause switch only, and a running network may still hold the time.
    """
    started: list[Network] = []

    def run[N: Network](
        network: N,
        *,
        _manual: bool = False,
    ) -> N:
        started.append(network)
        return mujou.run(network, _manual=_manual)

    try:
        yield run
    finally:
        for network in started:
            with contextlib.suppress(NetworkError):
                network.shutdown()
