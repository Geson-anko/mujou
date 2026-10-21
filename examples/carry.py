"""Carrying data along a path: each sample takes its ID to the end.

``Stamp`` sends each input with the time the host wrote it as an ID. ``Double``
passes the ID on without reading or copying it, and ``Report`` sends each value
with the ID of the very sample it was computed from, as a plain message that the
host reads. The host waits for the report of the sample it wrote by its ID.
"""

from typing import TypedDict, override

import torch

import mujou
from mujou import DictSpec, NextPort, Node, PrevPort, TensorSpec

SAMPLE = TensorSpec((2,))
SAMPLE_ID = TensorSpec((), torch.int64)


class Reported(TypedDict):
    value: torch.Tensor
    id: torch.Tensor


REPORTED = DictSpec(Reported, value=SAMPLE, id=SAMPLE_ID)


# --8<-- [start:nodes]
class Stamp(Node):
    """Sends the input with the time it was written at as its ID."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = PrevPort(recv=SAMPLE)  # the host writes it
        self.next = NextPort(send=SAMPLE, send_carry=SAMPLE_ID)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        x, written_at = self.prev.recv_with_time()
        self.next.send(x, carry=torch.tensor(written_at.ns))


class Double(Node):
    """Sends ``2 x``, carrying on what came with ``x``."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = PrevPort(recv=SAMPLE, recv_carry=SAMPLE_ID)
        self.next = NextPort(send=SAMPLE, send_carry=self.prev)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(2 * self.prev.recv())


class Report(Node):
    """Sends each value with the ID it carries, for the host to read."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = PrevPort(recv=SAMPLE, recv_carry=SAMPLE_ID)
        self.next = NextPort(send=REPORTED)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        value = self.prev.recv()
        self.next.send({"value": value, "id": self.prev.carried()})


# --8<-- [end:nodes]


class Pipeline(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.stamp = Stamp()
        self.double = Double()
        self.report = Report()
        self.stamp.next.connect(self.double.prev)
        self.double.next.connect(self.report.prev)
        self.prev = self.stamp.prev
        self.next = self.report.next


# --8<-- [start:host]
def main() -> None:
    net = Pipeline()
    with mujou.run(net) as run:
        for k in range(3):
            written_at = run.write({net.prev: torch.full((2,), float(k))})
            report, _ = run.wait_for(
                net.next, lambda r: r["id"].item() == written_at.ns, timeout=5.0
            )
            print(k, report["value"])  # 2 k, never the value of another sample


# --8<-- [end:host]

if __name__ == "__main__":
    main()
