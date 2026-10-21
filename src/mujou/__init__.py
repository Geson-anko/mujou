"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

import importlib.metadata

from mujou.errors import NetworkError
from mujou.execution import Run, run
from mujou.network import Group, Network
from mujou.node import Node
from mujou.port import Channel, MultiPort, Port
from mujou.spec import DictSpec, MessageSpec, TensorSpec

__all__ = [
    "Channel",
    "DictSpec",
    "Group",
    "MessageSpec",
    "MultiPort",
    "Network",
    "NetworkError",
    "Node",
    "Port",
    "Run",
    "TensorSpec",
    "__version__",
    "run",
]

__version__ = importlib.metadata.version("mujou")
