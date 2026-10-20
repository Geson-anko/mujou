"""The input node of ``mujou.pc``."""

from collections.abc import Sequence
from typing import override

import torch

from mujou._core.node import Node
from mujou._core.pc.carry import Carried, Sample, batch_of, check_batch
from mujou._core.port import NextPort, PrevPort
from mujou._core.spec import DictSpec, TensorSpec


class Input(Node):
    """A node that takes the samples the host writes and sends them on, the input
    ``x`` as the activity and the target with it as the carried data.

    The host writes a ``Sample`` (``x``, ``y`` and ``mask``) to ``prev`` with
    ``Run.write``. In each ``on_update`` the node sends ``x`` from ``next``,
    carrying ``y``, ``mask`` and the ``time`` (ns) at which the host wrote them:
    the nodes on the path pass them on with the activity computed from ``x``, so
    that the output node pairs each prediction with its own target. A new
    ``time`` tells them the input changed. ``x`` is clamped: what the next node
    sends back on ``next`` is not read.

    Before the host first writes, the node sends zeros with ``mask`` False.

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
        self.next = NextPort(send=x, recv=x, send_carry=carry)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        sample, written_at = self.prev.recv_with_time()
        time = torch.tensor(written_at.ns, dtype=torch.int64, device=self.device)
        self.next.send(
            sample["x"], carry={"y": sample["y"], "mask": sample["mask"], "time": time}
        )
