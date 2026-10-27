"""The types of mujou that annotations use.

- ``Size`` is a shape: an int ``n`` is ``(n,)``, and ``()`` has no dimension,
  as the batch of ``mujou.pc`` (``mujou.pc.msg.activity_spec``). Unlike
  ``torch.Size``, it takes an int as well.
- ``Device`` is where a node computes or a tensor is made: a ``torch.device``
  or its name, such as ``"cpu"`` or ``"cuda:0"``.
- ``Message`` is a message the host writes: a ``torch.Tensor``, a Python
  scalar (``int``, ``float``, ``bool`` or ``Timestamp``), or a dict of them (a
  ``TypedDict`` of the port's ``DictSpec``).

Import them from here::

    from mujou.types import Device, Message, Size
"""

from collections.abc import Mapping

import torch

from mujou._native import Timestamp

__all__ = ["Device", "Message", "Size"]

type Size = int | tuple[int, ...]
type Device = torch.device | str
# A TypedDict is a Mapping[str, object], not a Mapping[str, torch.Tensor].
type Message = torch.Tensor | int | float | bool | Timestamp | Mapping[str, object]
