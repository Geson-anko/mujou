"""Benchmark of the M:N scheduling: nodes on the worker pool against a thread each.

Not a test: pytest does not collect it and no ``just`` recipe runs it. Run it by
hand on an otherwise idle machine and read the Markdown tables it prints::

    uv run python benchmarks/mn_scheduler.py                # every scenario
    uv run python benchmarks/mn_scheduler.py scale --nodes 8 64 256 --seconds 3
    uv run python benchmarks/mn_scheduler.py workers switch mixed

With the GIL (the build CI also tests), in an environment of its own::

    UV_PROJECT_ENVIRONMENT=.venv-gil uv run --python 3.14 \\
        python benchmarks/mn_scheduler.py

Scenarios (each run of a configuration warms up, then measures a window):

- ``scale``: M nodes on their own threads (``dedicated``), on the default pool
  (``pool``) and on a pool of ``--small-workers`` workers, for each M, period
  (0 and 0.01 s) and load (``light``: a Python counter, under 1 µs to a few µs;
  ``addmm``: a torch ``addmm_``, and ``tanh``: a torch ``tanh``, tens to
  hundreds of µs each, with one intra-op thread). Threads of the process,
  voluntary and involuntary context switches per second (``getrusage``; macOS
  counts every switch as involuntary), updates per
  second (total, and the median and minimum per node), and for a period above 0
  the rate achieved against one update per period and the percentiles of Δt.
- ``workers``: the same measures for M nodes over a range of worker counts, to
  judge the default number of workers.
- ``switch``: the cost of switching a worker between nodes, with one worker:
  one node against two that alternate, and the two with ``_on_enter``
  replaced by the least it must do.
- ``mixed``: nodes of period 0 among nodes of period 0.01 s, with a dedicated
  node of period 1 s and the host sleeping on the same timer, against the same
  nodes all at 0.01 s. Shows the wake-ups of other waiters on the timer that the
  pool causes.
"""

import argparse
import os
import platform
import resource
import statistics
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import override

import torch

import mujou
from mujou._core.hooks import enter_node


class Light(mujou.Node):
    """Counts its updates: the hook machinery and little else."""

    def __init__(self, *, period: float, dedicated: bool) -> None:
        super().__init__(period=period, dedicated_thread=dedicated)
        self.dts: list[float] = []

    @override
    def on_update(self, dt: float) -> None:
        self.dts.append(dt)


class Addmm(Light):
    """A dense layer on the CPU, ``out = x @ w``. On Apple silicon the matrix
    unit is shared by a cluster of cores, so this scales with the clusters."""

    def __init__(self, *, period: float, dedicated: bool, size: int) -> None:
        super().__init__(period=period, dedicated=dedicated)
        self.x = torch.randn(size, size)
        self.w = torch.randn(size, size)
        self.out = torch.zeros(size, size)

    @override
    def on_update(self, dt: float) -> None:
        self.out.addmm_(self.x, self.w, beta=0.0)
        self.dts.append(dt)


class Tanh(Light):
    """An elementwise op on the CPU, ``out = tanh(x)``: scales with the cores."""

    def __init__(self, *, period: float, dedicated: bool, size: int) -> None:
        super().__init__(period=period, dedicated=dedicated)
        self.x = torch.randn(size)
        self.out = torch.empty(size)

    @override
    def on_update(self, dt: float) -> None:
        torch.tanh(self.x, out=self.out)
        self.dts.append(dt)


class BareEnter(Light):
    """Its switches do only what a node without ports or CUDA needs."""

    @override
    def _on_enter(self) -> None:
        enter_node(self)


@dataclass
class Result:
    threads: int
    voluntary: float
    involuntary: float
    total: float
    median: float
    minimum: float
    dt_p50: float
    dt_p99: float
    dt_max: float
    others: float


def count_threads() -> int:
    """The threads of this process now."""
    if sys.platform == "linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    out = subprocess.run(
        ["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout
    return len(out.splitlines()) - 1


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def measure(
    nodes: Sequence[Light],
    workers: int | None,
    warmup: float,
    seconds: float,
    during: Callable[[float], None] | None = None,
    observed: int | None = None,
) -> Result:
    """Runs the nodes and measures a window of ``seconds`` after ``warmup``.

    The rates and Δt are of the first ``observed`` nodes (all by default);
    ``others`` is the updates per second of the rest.
    """
    with mujou.run(*nodes, workers=workers):
        time.sleep(warmup)
        start = [len(n.dts) for n in nodes]
        before = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        if during is None:
            time.sleep(seconds / 2)
            threads = count_threads()
            time.sleep(seconds / 2)
        else:
            threads = count_threads()
            during(seconds)
        elapsed = time.perf_counter() - t0
        after = resource.getrusage(resource.RUSAGE_SELF)
        end = [len(n.dts) for n in nodes]
    every = [(e - s) / elapsed for s, e in zip(start, end, strict=True)]
    k = len(nodes) if observed is None else observed
    rates = every[:k]
    windows = zip(nodes[:k], start, end, strict=False)
    dts = sorted(dt for n, s, e in windows for dt in n.dts[s:e])
    return Result(
        threads=threads,
        voluntary=(after.ru_nvcsw - before.ru_nvcsw) / elapsed,
        involuntary=(after.ru_nivcsw - before.ru_nivcsw) / elapsed,
        total=sum(rates),
        median=statistics.median(rates),
        minimum=min(rates),
        dt_p50=percentile(dts, 0.50),
        dt_p99=percentile(dts, 0.99),
        dt_max=dts[-1] if dts else float("nan"),
        others=sum(every[k:]),
    )


def make(
    load: str, m: int, period: float, dedicated: bool, args: argparse.Namespace
) -> list[Light]:
    if load == "light":
        return [Light(period=period, dedicated=dedicated) for _ in range(m)]
    if load == "addmm":
        return [
            Addmm(period=period, dedicated=dedicated, size=args.size) for _ in range(m)
        ]
    return [
        Tanh(period=period, dedicated=dedicated, size=args.tanh_size) for _ in range(m)
    ]


HEADER = (
    "| load | period | M | mode | threads | vcsw/s | ivcsw/s | updates/s | "
    "median/node | min/node | of 1/period (median, min) | Δt p50 ms | Δt p99 ms | Δt max ms |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
)


def row(load: str, period: float, m: int, mode: str, r: Result) -> str:
    hz = f"{r.median * period:.0%}, {r.minimum * period:.0%}" if period > 0 else "-"
    # macOS counts every switch as involuntary (ru_nvcsw stays 0).
    voluntary = "-" if sys.platform == "darwin" else f"{r.voluntary:,.0f}"
    return (
        f"| {load} | {period:g} | {m} | {mode} | {r.threads} | {voluntary} | "
        f"{r.involuntary:,.0f} | {r.total:,.0f} | {r.median:,.0f} | {r.minimum:,.0f} | "
        f"{hz} | {r.dt_p50 * 1e3:.3f} | {r.dt_p99 * 1e3:.3f} | {r.dt_max * 1e3:.2f} |"
    )


def scale(args: argparse.Namespace) -> None:
    print("\n## scale\n")
    print(HEADER)
    modes: list[tuple[str, bool, int | None]] = [
        ("dedicated", True, None),
        ("pool", False, None),
        (f"pool w={args.small_workers}", False, args.small_workers),
    ]
    for load in args.loads:
        for period in args.periods:
            for m in args.nodes:
                for mode, dedicated, workers in modes:
                    nodes = make(load, m, period, dedicated, args)
                    r = measure(nodes, workers, args.warmup, args.seconds)
                    print(row(load, period, m, mode, r), flush=True)


def workers(args: argparse.Namespace) -> None:
    print("\n## workers\n")
    print(HEADER)
    for load in args.loads:
        for period in args.periods:
            for m in args.workers_nodes:
                for w in args.worker_counts:
                    nodes = make(load, m, period, False, args)
                    r = measure(nodes, w, args.warmup, args.seconds)
                    print(row(load, period, m, f"pool w={w}", r), flush=True)


def switch(args: argparse.Namespace) -> None:
    print("\n## switch (workers=1, period 0, light)\n")
    print("| nodes | µs/update | extra µs/update against one node |")
    print("| --- | --- | --- |")
    cases: list[tuple[str, Callable[[], list[Light]]]] = [
        ("1 node", lambda: [Light(period=0.0, dedicated=False)]),
        (
            "2 nodes alternating",
            lambda: [Light(period=0.0, dedicated=False) for _ in range(2)],
        ),
        (
            "2 nodes alternating, _on_enter only binds",
            lambda: [BareEnter(period=0.0, dedicated=False) for _ in range(2)],
        ),
    ]
    base = None
    for name, build in cases:
        per = []
        for _ in range(args.repeats):
            r = measure(build(), 1, args.warmup, args.seconds)
            per.append(1e6 / r.total)
        us = statistics.median(per)
        base = us if base is None else base
        print(f"| {name} | {us:.3f} | {us - base:+.3f} |", flush=True)


def mixed(args: argparse.Namespace) -> None:
    m, z = args.mixed_nodes, args.mixed_zero
    print(f"\n## mixed ({m} pool nodes at 0.01 s and {z} at period 0, light)\n")
    print("A dedicated node at period 1 s and the host sleep on the same timer.")
    print("The measures are of the nodes at 0.01 s.\n")
    print(HEADER)
    for w in args.mixed_workers:
        for zero in (0, z):
            nodes = [Light(period=0.01, dedicated=False) for _ in range(m)]
            nodes += [Light(period=0.0, dedicated=False) for _ in range(zero)]
            nodes.append(Light(period=1.0, dedicated=True))
            # The host sleeping is one more waiter on the global timer.
            r = measure(
                nodes, w, args.warmup, args.seconds, mujou.time.sleep, observed=m
            )
            mode = f"w={w}, {zero} at 0 ({r.others:,.0f}/s with the 1 s node)"
            print(row("light", 0.01, m, mode, r), flush=True)


def describe() -> None:
    gil = "free-threaded" if not sys._is_gil_enabled() else "GIL"  # pyright: ignore[reportPrivateUsage]
    print(f"- Python {platform.python_version()} ({gil}), torch {torch.__version__}")
    print(
        f"- {platform.platform()}, {platform.machine()}, os.cpu_count()={os.cpu_count()}"
    )
    if sys.platform == "darwin":
        keys = ["machdep.cpu.brand_string", "hw.physicalcpu", "hw.logicalcpu"]
        keys += ["hw.perflevel0.physicalcpu", "hw.perflevel1.physicalcpu"]
        for key in keys:
            out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
            print(f"- {key}: {out.stdout.strip()}")
        print("- macOS: getrusage counts every context switch in ru_nivcsw (ivcsw/s)")
    print(f"- Py_GIL_DISABLED={sysconfig.get_config_var('Py_GIL_DISABLED')}")


def cost(node: Light) -> float:
    """µs of one ``on_update`` of ``node`` on this thread."""
    for _ in range(100):
        node.on_update(0.0)
    n = 2000
    t0 = time.perf_counter()
    for _ in range(n):
        node.on_update(0.0)
    return (time.perf_counter() - t0) / n * 1e6


SCENARIOS: dict[str, Callable[[argparse.Namespace], None]] = {
    "scale": scale,
    "workers": workers,
    "switch": switch,
    "mixed": mixed,
}


def main() -> None:
    cpus = os.cpu_count() or 1
    parser = argparse.ArgumentParser(description="Benchmark of the M:N scheduling.")
    parser.add_argument("scenarios", nargs="*", help=f"any of {', '.join(SCENARIOS)}")
    parser.add_argument("--nodes", type=int, nargs="+", default=[8, 64, 256, 1024])
    parser.add_argument("--periods", type=float, nargs="+", default=[0.0, 0.01])
    loads = ["light", "addmm", "tanh"]
    parser.add_argument("--loads", nargs="+", choices=loads, default=loads)
    parser.add_argument(
        "--size", type=int, default=384, help="addmm: side of the matrices"
    )
    parser.add_argument("--tanh-size", type=int, default=32768, help="tanh: elements")
    parser.add_argument("--small-workers", type=int, default=max(1, cpus // 2))
    parser.add_argument("--workers-nodes", type=int, nargs="+", default=[256])
    parser.add_argument(
        "--worker-counts",
        type=int,
        nargs="+",
        default=sorted({1, 2, 4, max(1, cpus // 2), cpus, 2 * cpus}),
    )
    parser.add_argument("--mixed-nodes", type=int, default=64)
    parser.add_argument("--mixed-zero", type=int, default=2)
    parser.add_argument("--mixed-workers", type=int, nargs="+", default=[4, cpus])
    parser.add_argument(
        "--warmup", type=float, default=0.5, help="seconds before measuring"
    )
    parser.add_argument("--seconds", type=float, default=2.0, help="seconds measured")
    parser.add_argument("--repeats", type=int, default=3, help="switch: runs per case")
    args = parser.parse_args()
    unknown = set(args.scenarios) - set(SCENARIOS)
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(sorted(unknown))}")

    # One intra-op thread: the nodes' parallelism is the threads that run them.
    torch.set_num_threads(1)
    print("# M:N scheduling benchmark\n")
    describe()
    addmm = cost(Addmm(period=0.0, dedicated=False, size=args.size))
    tanh = cost(Tanh(period=0.0, dedicated=False, size=args.tanh_size))
    print(f"- addmm: {args.size}x{args.size}, {addmm:.1f} µs per update alone")
    print(f"- tanh: {args.tanh_size} elements, {tanh:.1f} µs per update alone")
    for name in args.scenarios or SCENARIOS:
        SCENARIOS[name](args)


if __name__ == "__main__":
    main()
