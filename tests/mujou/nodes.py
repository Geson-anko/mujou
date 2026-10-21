"""The test nodes and the helpers that several test modules share."""

import threading
from collections.abc import Callable
from typing import Any, cast, override

import torch

from mujou import MultiPort, Network, Node, Port, TensorSpec
from mujou.time import Timestamp

F64 = torch.float64


def vec() -> TensorSpec:
    return TensorSpec((2,), F64)


class Leaky(Node):
    """``dx/dt = -(x - u) / tau`` by explicit Euler, the rate clipped at 1.

    ``u`` comes from the previous side and ``x`` goes to the next side. Raises in
    the ``raise_at``-th ``on_update`` (or in ``on_start`` with ``raise_at=0``).
    """

    def __init__(self, tau: float = 1.0, raise_at: int | None = None) -> None:
        super().__init__()
        self.prev = Port(recv=vec())
        self.next = Port(send=vec())
        self.tau = tau
        self.raise_at = raise_at
        # Set by a test to make the next on_update raise.
        self.trigger = threading.Event()
        self.x = torch.zeros(2, dtype=F64)
        self.updates = 0
        self.started = False
        self.error: Exception | None = None

    def fail(self) -> None:
        self.error = ValueError(f"boom after {self.updates} updates")
        raise self.error

    @override
    def on_start(self) -> None:
        self.started = True
        if self.raise_at == 0:
            self.fail()

    @override
    def on_update(self, dt: float) -> None:
        self.updates += 1
        if self.updates == self.raise_at or self.trigger.is_set():
            self.fail()
        self.x += min(dt / self.tau, 1.0) * (self.prev.recv() - self.x)
        self.next.send(self.x)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {"x": self.x, "updates": self.updates}

    @override
    def load_state_dict(self, state: object) -> None:
        loaded = cast(dict[str, Any], state)
        self.x.copy_(loaded["x"])
        self.updates = loaded["updates"]


class Chain(Network):
    """``u -> a -> b ->``: two leaky integrators in a row."""

    def __init__(
        self,
        tau: float = 1.0,
        raise_in_a: int | None = None,
        raise_in_b: int | None = None,
    ) -> None:
        super().__init__()
        self.a = Leaky(tau, raise_in_a)
        self.b = Leaky(tau, raise_in_b)
        self.a.next.connect(self.b.prev)
        self.prev = self.a.prev
        self.next = self.b.next


class Pair(Node):
    """Receives two inputs from the host and keeps them with their times."""

    def __init__(self) -> None:
        super().__init__()
        self.x = Port(recv=vec())
        self.y = Port(recv=vec())
        self.many = MultiPort(recv=vec())
        self.out = Port(send=vec())
        self.seen: list[
            tuple[
                tuple[torch.Tensor, Timestamp | None],
                tuple[torch.Tensor, Timestamp | None],
            ]
        ] = []

    @override
    def on_update(self, dt: float) -> None:
        self.seen.append((self.x.recv_with_time(), self.y.recv_with_time()))


class Calls(Node):
    """Calls ``action`` in each ``on_update``, keeps what it raised, and records
    each ``dt``."""

    def __init__(
        self,
        action: Callable[[], object] = lambda: None,
        *,
        dedicated_thread: bool = False,
    ) -> None:
        super().__init__(dedicated_thread=dedicated_thread)
        self.action = action
        self.errors: list[Exception] = []
        self.dts: list[float] = []

    @override
    def on_update(self, dt: float) -> None:
        self.dts.append(dt)
        try:
            self.action()
        except Exception as e:  # noqa: BLE001 - kept for the test to inspect
            self.errors.append(e)


class Probe(Node):
    """Holds the given ports and, in each ``on_update``, keeps what ``update``
    returns (or the exception it raised). A subclass sets ``on_device`` to
    compute on another device (a keyword would collide with ``**ports``)."""

    on_device = "cpu"

    def __init__(
        self,
        update: Callable[[], object] = lambda: None,
        **ports: Port[Any, Any] | MultiPort[Any, Any],
    ) -> None:
        super().__init__(device=self.on_device)
        for name, port in ports.items():
            setattr(self, name, port)
        self.update = update
        self.results: list[object] = []

    @override
    def on_update(self, dt: float) -> None:
        try:
            self.results.append(self.update())
        except Exception as e:  # noqa: BLE001 - kept for the test to inspect
            self.results.append(e)


class Hidden(Node):
    """Sends and receives ``(2, 3)`` on both sides."""

    def __init__(self) -> None:
        super().__init__()
        spec = TensorSpec((2, 3))
        self.prev = Port(send=spec, recv=spec)
        self.next = MultiPort(send=spec, recv=spec)


def assert_same_tree(actual: Any, expected: Any) -> None:
    """``actual`` has the structure and values of ``expected``: the same types,
    the same keys in the same order, and equal tensors (compared on the CPU)."""
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)  # pyright: ignore[reportUnknownArgumentType]
        for key in expected:  # pyright: ignore[reportUnknownVariableType]
            assert_same_tree(actual[key], expected[key])
    elif type(expected) in (list, tuple):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            assert_same_tree(a, e)
    else:
        assert actual == expected


def entries(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Every ``{"value", "time_ns"}`` of the slots of ``tree``."""
    found: list[dict[str, Any]] = []
    for node in tree["nodes"].values():
        for slots in node["ports"].values():
            for key, held in slots.items():
                found.extend(held.values() if key == "to" else [held])
    return found


def updates(tree: dict[str, Any]) -> dict[str, int]:
    """The last update time of each node of ``tree``, by path."""
    return {path: node["last_update_ns"] for path, node in tree["nodes"].items()}


def port_slots(tree: dict[str, Any], path: str, attribute: str) -> dict[str, Any]:
    """The slots of the port ``attribute`` of the node at ``path``."""
    return tree["nodes"][path]["ports"][attribute]


def close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-14) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)
