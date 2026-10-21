"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

import importlib.metadata

from mujou._core.errors import NetworkError
from mujou._core.network import Group, Network
from mujou._core.node import Node
from mujou._core.port import Channel, MultiPort, Port
from mujou._core.run import Run, run
from mujou._core.spec import DictSpec, MessageSpec, TensorSpec

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
