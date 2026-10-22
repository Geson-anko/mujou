"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

import importlib.metadata

from mujou.errors import NetworkError
from mujou.execution import Run, run
from mujou.network import Group, Network
from mujou.node import Node
from mujou.port import Channel, MultiPort, Port
from mujou.spec import (
    BoolSpec,
    DictSpec,
    FieldSpec,
    FloatSpec,
    IntSpec,
    MessageSpec,
    TensorSpec,
    TimestampSpec,
)

__all__ = [
    "BoolSpec",
    "Channel",
    "DictSpec",
    "FieldSpec",
    "FloatSpec",
    "Group",
    "IntSpec",
    "MessageSpec",
    "MultiPort",
    "Network",
    "NetworkError",
    "Node",
    "Port",
    "Run",
    "TensorSpec",
    "TimestampSpec",
    "__version__",
    "run",
]

__version__ = importlib.metadata.version("mujou")
