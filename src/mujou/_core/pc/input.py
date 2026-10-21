"""The input node of ``mujou.pc``."""

from typing import override

import torch

from mujou._core.node import Node
from mujou._core.pc.messages import Sample, activity_spec
from mujou._core.port import MultiPort, Port
from mujou._core.spec import DictSpec, TensorSpec


class Input(Node):
    """A node that takes the samples the host writes and sends them on as the
    ``Activity`` of the next node, with their targets.

    The host writes a ``Sample`` (``x``, ``y`` and ``mask``) to ``prev`` with
    ``Run.write``. In each ``on_update`` the node sends from ``next`` an
    ``Activity`` whose ``h`` is ``x``, with ``y``, ``mask`` and the ``time``
    (ns) at which the host wrote them: the nodes on the path pass them on with
    the activity computed from ``x``, so that the output node pairs each
    prediction with its own target. A new ``time`` tells them the input
    changed. The ``epoch`` counts the writes the node has seen, so that the next
    node moves its ``h`` with a new ``x`` even when it was written at the same
    time as the zeros before it (time 0). ``x`` is clamped: what the next node
    sends back on ``next`` is not read.

    Before the host first writes, the node sends zeros with ``mask`` False,
    ``epoch`` 0 and ``time`` 0. The state holds the ``epoch`` (``"_epoch"``)
    and the time (ns) of the last write seen (``"_seen_write"``, -1 before
    any). ``act`` is the spec of what the node sends, for the next node.

    Example::

        B = 64
        inp = pc.Input(
            x=TensorSpec((B, 784)), y=TensorSpec((B,), torch.int64), period=1e-3
        )
        ...
        with mujou.run(net) as run:
            run.write({inp.prev: {"x": x, "y": y, "mask": torch.ones(B, dtype=torch.bool)}})

    Args:
        x: The spec of the inputs ``x``, ``(B, *in)``.
        y: The spec of the targets ``y``, ``(B, *target)``.
        period: The period of the node in seconds, usually that of the next
            node.
        device: Where the node computes.

    Raises:
        ValueError: The first dimensions of ``x`` and ``y`` differ.
    """

    def __init__(
        self,
        *,
        x: TensorSpec,
        y: TensorSpec,
        period: float,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device, period=period)
        self.act = activity_spec(x, y)
        mask = self.act.specs["mask"]
        self.prev = Port(recv=DictSpec(Sample, x=x, y=y, mask=mask))  # the host
        self.next = MultiPort(send=self.act, recv=x)
        self._epoch = 0
        self._seen_write = -1
        for name in ("_epoch", "_seen_write"):
            self.register_state(name)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written_at = self.prev.recv_with_time()
        if written_at is not None and written_at.ns != self._seen_write:
            self._seen_write = written_at.ns
            self._epoch += 1
        written_ns = 0 if written_at is None else written_at.ns
        time = torch.tensor(written_ns, dtype=torch.int64, device=self.device)
        epoch = torch.tensor(self._epoch, dtype=torch.int64, device=self.device)
        self.next.send(
            {
                "h": sample["x"],
                "epoch": epoch,
                "y": sample["y"],
                "mask": sample["mask"],
                "time": time,
            }
        )
