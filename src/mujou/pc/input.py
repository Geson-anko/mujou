"""The input node of ``mujou.pc``."""

from typing import override

from mujou import DictSpec, MultiPort, Node, Port, TensorSpec
from mujou.pc.msg import Sample, activity_spec
from mujou.time import Timestamp
from mujou.types import Device, Size


class Input(Node):
    """A node that takes the samples the host writes and sends them on as the
    ``Activity`` of the next node, with their targets.

    The host writes a ``Sample`` (``x``, ``y`` and ``mask``) to ``prev`` with
    ``Run.write``. In each ``on_update`` the node sends from ``next`` an
    ``Activity`` whose ``h`` is ``x``, with ``y``, ``mask`` and the ``time``
    (ns) at which the host wrote them: the nodes on the path pass them on with
    the activity computed from ``x``, so that the output node pairs each
    prediction with its own target. A new ``time`` tells them the input
    changed. ``jumps`` counts the writes the node has seen, so that the next
    node moves its ``h`` with a new ``x`` even when it was written at the same
    time as the zeros before it (time 0). ``x`` is clamped: what the next node
    sends back on ``next`` is not read.

    Before the host first writes, the node sends zeros with ``mask`` False,
    ``jumps`` 0 and ``time`` 0. The state holds ``"jumps"`` and the time (ns)
    of the last write seen (``"seen_write"``, ``None`` before any).
    ``act_spec`` is the spec of what the node sends, for the next node.

    Example::

        B = 64
        inp = pc.Input(
            x=TensorSpec((B, 784)),
            y=TensorSpec((B,), torch.int64),
            period=1e-3,
            batch=B,
        )
        ...
        mask = torch.ones(B, dtype=torch.bool)
        with mujou.run(net) as run:
            run.write({inp.prev: {"x": x, "y": y, "mask": mask}})

    Args:
        x: The spec of the inputs ``x``, ``(*batch, *in)``.
        y: The spec of the targets ``y``, ``(*batch, *target)``.
        period: The period of the node in seconds, usually that of the next
            node.
        batch: The batch that ``x`` and ``y`` start with, as in
            ``activity_spec``: ``()`` for one sample, ``B`` for ``B`` samples.
        device: Where the node computes.

    Raises:
        ValueError: ``x`` or ``y`` does not start with ``batch``.
    """

    def __init__(
        self,
        *,
        x: TensorSpec,
        y: TensorSpec,
        period: float,
        batch: Size = (),
        device: Device = "cpu",
    ) -> None:
        super().__init__(device=device, period=period)
        self.act_spec = activity_spec(x, y, batch)
        mask = self.act_spec.field("mask", TensorSpec)
        self.prev = Port(recv=DictSpec(Sample, x=x, y=y, mask=mask))  # the host
        self.next = MultiPort(send=self.act_spec, recv=x)
        self.jumps = 0
        self.seen_write: int | None = None
        self.register_state("jumps")
        self.register_state("seen_write")

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written_at = self.prev.recv_with_time()
        if written_at is not None and written_at.ns != self.seen_write:
            self.seen_write = written_at.ns
            self.jumps += 1
        self.next.send(
            {
                "h": sample["x"],
                "jumps": self.jumps,
                "y": sample["y"],
                "mask": sample["mask"],
                "time": Timestamp(ns=0) if written_at is None else written_at,
            }
        )
