"""Tests of the node of ``leaky.py``, with ``mujou.testing``.

``mujou.testing.run`` runs on a ``VirtualTimer``, whose time moves only when the
test steps it, and ``ManualRun.step`` updates the nodes in a fixed order. Every
``dt`` is then exactly the step, and the run can be compared with the discrete
algorithm. Run them with::

    uv run pytest examples/test_leaky.py
"""

import math
from typing import Any

import torch
from leaky import SCALAR, Leaky, LeakyChain

import mujou.testing


# --8<-- [start:test]
class TestLeaky:
    def test_follows_the_exact_solution(self) -> None:
        tau, h, steps = 0.1, 0.001, 100
        node = Leaky(tau)
        one = torch.tensor(1.0, dtype=torch.float64)
        expected = 0.0
        with mujou.testing.run(node, inputs={node.prev: one}) as run:
            for _ in range(steps):
                run.step(h)  # 1 ms later, then every node once (period is ignored)
                expected += (1 - math.exp(-h / tau)) * (1.0 - expected)
            y, t = run.read_with_time(node.next)
        assert t is not None and t.ns == steps * 1_000_000  # sent at the last step
        torch.testing.assert_close(y, torch.tensor(expected, dtype=torch.float64))


# --8<-- [end:test]


# --8<-- [start:chain]
class TestLeakyChain:
    def test_follows_the_discrete_algorithm(self) -> None:
        tau, h, steps = 0.01, 0.001, 50
        u = torch.tensor(1.0, dtype=torch.float64)
        # The reference: the discrete algorithm, a then b at every step.
        a, b = SCALAR.zeros(), SCALAR.zeros()
        for _ in range(steps):
            a += (1 - math.exp(-h / tau)) * (u - a)
            b += (1 - math.exp(-h / tau)) * (a - b)  # b reads the new a

        net = LeakyChain(tau)
        with mujou.testing.run(net, inputs={net.prev: u}) as run:
            for _ in range(steps):
                run.step(h, net.a, net.b)  # a, then b, which reads a's new value
            y = run.read(net.next)
        assert torch.equal(y, b)  # the same operations: equal to the last bit


# --8<-- [end:chain]


# --8<-- [start:restore]
def run_steps(n: int, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a new ``LeakyChain`` for ``n`` steps of 1 ms and return its state."""
    net = LeakyChain(0.01)
    u = torch.tensor(1.0, dtype=torch.float64)
    with mujou.testing.run(net, inputs={net.prev: u}, state=state) as run:
        for _ in range(n):
            run.step(0.001)
    return run.state_dict()


class TestRestore:
    def test_a_restored_run_goes_on_as_one_that_did_not_stop(self) -> None:
        straight = run_steps(20)
        # Ten steps, then ten more in a new instance restored from the first half,
        # on a VirtualTimer of its own that starts at the time of the state.
        resumed = run_steps(10, state=run_steps(10))
        x = resumed["nodes"]["b"]["state"]["x"]
        assert torch.equal(x, straight["nodes"]["b"]["state"]["x"])


# --8<-- [end:restore]
