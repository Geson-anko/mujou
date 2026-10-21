"""The types of mujou that annotations use, such as ``Size``.

``Size`` is a shape: an int ``n`` is ``(n,)``, and ``()`` has no dimension, as
the batch of ``mujou.pc`` (``mujou.pc.activity_spec``). Unlike ``torch.Size``,
it takes an int as well.

Import it from here::

    from mujou.types import Size
"""

__all__ = ["Size"]

type Size = int | tuple[int, ...]
