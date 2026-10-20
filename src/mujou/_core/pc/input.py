"""The input node of ``mujou.pc``."""

from collections.abc import Sequence
from typing import override

import torch

from mujou._core.node import Node
from mujou._core.pc.carry import (
    Carried,
    Sample,
    activity_spec,
    batch_of,
    check_batch,
)
from mujou._core.port import NextPort, PrevPort
from mujou._core.spec import DictSpec, TensorSpec


class Input(Node):
    """A node that takes the samples the host writes and sends them on, the input
    ``x`` as the activity and the target with it as the carried data.

    The host writes a ``Sample`` (``x``, ``y`` and ``mask``) to ``prev`` with
    ``Run.write``. In each ``on_update`` the node sends ``x`` from ``next`` as
    the ``h`` of an ``Activity``, carrying ``y``, ``mask`` and the ``time`` (ns)
    at which the host wrote them: the nodes on the path pass them on with the
    activity computed from ``x``, so that the output node pairs each prediction
    with its own target. A new ``time`` tells them the input changed. The
    ``epoch`` of the ``Activity`` counts the writes the node has seen, so that
    the next node moves its ``h`` with a new ``x`` even when it was written at
    the same time as the zeros before it (time 0). ``x`` is clamped: what the
    next node sends back on ``next`` is not read.

    Before the host first writes, the node sends zeros with ``mask`` False and
    ``epoch`` 0. The state holds the ``epoch`` (``"_epoch"``) and the time (ns)
    of the last write seen (``"_seen_write"``, -1 before any).

    Example::

        carry = pc.carry_spec(TensorSpec((B,), torch.int64), batch=B)
        inp = pc.Input((B, 784), carry=carry, period=1e-3)
        ...
        with mujou.run(net) as run:
            run.write({inp.prev: {"x": x, "y": y, "mask": torch.ones(B, dtype=torch.bool)}})

    Args:
        shape: The shape ``(B, *in)`` of ``x``.
        carry: The spec of the carried data, from ``carry_spec``.
        period: The period of the node in seconds, usually that of the next
            node.
        dtype: The dtype of ``x``; ``None`` is ``torch.get_default_dtype()``.
        device: Where the node computes.

    Raises:
        TypeError: ``carry`` is not a spec of ``carry_spec``.
        ValueError: ``shape[0]`` is not the batch of ``carry``.
    """

    def __init__(
        self,
        shape: Sequence[int],
        *,
        carry: DictSpec[Carried],
        period: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(device=device, period=period)
        check_batch(shape, batch_of(carry), "shape")
        x = TensorSpec(shape, dtype)
        sample = DictSpec(Sample, x=x, y=carry.specs["y"], mask=carry.specs["mask"])
        self.prev = PrevPort(recv=sample)  # the host writes it
        self.next = NextPort(send=activity_spec(x), recv=x, send_carry=carry)
        self._epoch = 0
        self._seen_write = -1
        for name in ("_epoch", "_seen_write"):
            self.register_state(name)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written_at = self.prev.recv_with_time()
        if self.prev.has_received() and written_at.ns != self._seen_write:
            self._seen_write = written_at.ns
            self._epoch += 1
        time = torch.tensor(written_at.ns, dtype=torch.int64, device=self.device)
        epoch = torch.tensor(self._epoch, dtype=torch.int64, device=self.device)
        self.next.send(
            {"h": sample["x"], "epoch": epoch},
            carry={"y": sample["y"], "mask": sample["mask"], "time": time},
        )
