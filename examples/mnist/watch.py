"""Watch the ``dt`` of the nodes: whether they keep up with the time scale.

``watch(node)`` wraps the node's ``on_update`` (a run calls the hook it finds on
the node, so a wrapper set before the run is called instead) to count each
``dt`` in a histogram of quarter periods. The histogram has ``state_dict``, so
it is in the node's state, and the host reads it from ``Run.state_dict()``.
"""

import math
from collections.abc import Mapping
from typing import Any

import torch

import mujou

BINS = 8 * 4  # quarter periods up to 8 periods, then the rest


class Histogram:
    """Counts ``dt`` in bins of a quarter ``period``: bin ``i`` holds
    ``(i - 1) / 4 < dt / period <= i / 4``, and the last one the rest."""

    def __init__(self, period: float) -> None:
        self.period = period
        self.counts = torch.zeros(BINS + 2, dtype=torch.int64)

    def count(self, dt: float) -> None:
        # The tolerance keeps a dt of exactly one period (in whole ns) in bin 4.
        quarters = math.ceil(4 * dt / self.period - 1e-6)
        self.counts[min(quarters, BINS + 1)] += 1

    def state_dict(self) -> dict[str, Any]:
        return {"counts": self.counts}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.counts.copy_(state["counts"])


def watch(node: mujou.Node) -> None:
    """Count the ``dt`` of the updates of ``node`` in its attribute
    ``dt_histogram``; call it before the run."""
    histogram = Histogram(node.period)
    # An attribute with state_dict is part of the node's state. Node declares no
    # dt_histogram, hence setattr.
    setattr(node, "dt_histogram", histogram)
    # Wrapping on_update in the instance leaves the node's class as it is.
    update = node.on_update

    def on_update(dt: float) -> None:
        histogram.count(dt)
        update(dt)

    node.on_update = on_update


def quantile(counts: torch.Tensor, q: float) -> float:
    """Return the upper edge, in periods, of the bin of the ``q`` quantile."""
    rank = q * int(counts.sum())
    seen = 0
    for i, count in enumerate(counts.tolist()):
        seen += count
        if seen >= rank:
            return i / 4 if i <= BINS else math.inf
    return math.inf


def histograms(nodes: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return the counts of the watched nodes of ``nodes`` (the nodes of a state
    tree), by path."""
    return {
        path: node["state"]["dt_histogram"]["counts"]
        for path, node in nodes.items()
        if "dt_histogram" in node["state"]
    }


def report(nodes: Mapping[str, Any]) -> str:
    """Describe the ``dt`` of the watched nodes."""
    return "\n".join(
        f"  {path:>7}: {int(h.sum()):7d} updates, dt <= "
        + ", ".join(f"{quantile(h, q):g}" for q in (0.5, 0.9, 0.99))
        + f" periods (50/90/99%), > {BINS // 4} periods {int(h[BINS + 1])}"
        for path, h in histograms(nodes).items()
    )


def check(nodes: Mapping[str, Any]) -> None:
    """Warn when the ``dt`` of a watched node is too often longer than its period:
    a median over 1.5 periods, or a 99th percentile over 2 (``max_dt`` by default,
    beyond which the node drops the time)."""
    for path, h in histograms(nodes).items():
        if quantile(h, 0.5) > 1.5 or quantile(h, 0.99) > 2:
            print(
                f"warning: the updates of {path} come too late (see its dt): the "
                "nodes do not keep up with the time scale; lower --time-scale"
            )
