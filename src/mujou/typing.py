"""Names for type annotations.

Write these in annotations only; the values that mujou works with are made by the
names at the top level (``Port``, ``TensorSpec`` and so on).

- ``Single``: the peer count of a port that connects to at most one peer, the third
  type argument of ``Port[S, R, M]`` (the default of ``Port`` and ``PrevPort``).
  ``recv()`` of such a port returns the message itself
- ``Multi``: the peer count of a port that connects to any number of peers (the
  default of ``NextPort``, or ``multi=True``). ``recv()`` of such a port returns a
  dict keyed by ``Channel``
- ``MessageSpec[T]``: a spec of messages of type ``T``. ``TensorSpec`` is a
  ``MessageSpec[torch.Tensor]`` and ``DictSpec(Sample, ...)`` a
  ``MessageSpec[Sample]``. Annotate with it a function that takes either

An omitted direction is ``Never``: ``PrevPort(recv=TensorSpec((3,)))`` is a
``PrevPort[Never, torch.Tensor, Single]``.

Example::

    from typing import Never

    import torch

    from mujou import NextPort, PrevPort, TensorSpec
    from mujou.typing import MessageSpec, Multi, Single


    def sender[T](spec: MessageSpec[T]) -> NextPort[T, Never, Multi]:
        return NextPort(send=spec)


    def mean_of(port: PrevPort[Never, torch.Tensor, Multi]) -> torch.Tensor:
        return torch.stack(list(port.recv().values())).mean(0)


    inp: PrevPort[Never, torch.Tensor, Single] = PrevPort(recv=TensorSpec((3,)))
"""

from mujou._core.port import Multi as Multi, Single as Single
from mujou._core.spec import MessageSpec as MessageSpec

__all__ = ["MessageSpec", "Multi", "Single"]
