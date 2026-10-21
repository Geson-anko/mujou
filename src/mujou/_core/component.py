"""The common base of nodes and networks (``specs/network.md``).

A node or a network declares its parts as instance attributes: the run reads
them from the instance's ``__dict__`` (``tree.py``), so the base keeps no
registry of its own. It only rejects, as the class is defined, what no
instance's ``__dict__`` holds: the parts declared on the class, and the
attributes of ``__slots__``.
"""

from typing import Any

from mujou._core.port import MultiPort, Port


class Component:
    """The common base of ``Node`` and ``Network``.

    Raises:
        TypeError: A subclass declares a ``Port``, a ``MultiPort``, a ``Node``
            or a ``Network`` as a class attribute: every instance would share
            it, and the run, which reads the instance's attributes, would not
            see it. Or a subclass has ``__slots__`` (a dataclass with
            ``slots=True``, say), whose attributes the run would not see either.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__slots__" in vars(cls):
            raise TypeError(
                f"{cls.__name__} has __slots__, whose attributes a run does not "
                "read; leave them out (a dataclass without slots=True)"
            )
        for name, value in vars(cls).items():
            if is_part(value):
                raise TypeError(
                    f"{cls.__name__}.{name} is a {type(value).__name__} on the "
                    "class; assign it to an attribute of the instance in __init__"
                )


def is_part(value: object) -> bool:
    """Whether ``value`` is part of a tree: a port, a node or a network."""
    return isinstance(value, Port | MultiPort | Component)
