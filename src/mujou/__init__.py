"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

import importlib.metadata

# The submodules, so that `import mujou` is enough for `mujou.time.Timer` and
# `mujou.typing.Multi`. Not in __all__: a star import would shadow the standard
# `time` and `typing`.
from mujou import time as time, typing as typing
from mujou._core.errors import NetworkError
from mujou._core.graph import Graph
from mujou._core.network import Group, Network
from mujou._core.node import Node
from mujou._core.port import Channel, NextPort, Port, PrevPort
from mujou._core.run import Run, run
from mujou._core.spec import DictSpec, TensorSpec
from mujou._core.state import StateDict

__all__ = [
    "Channel",
    "DictSpec",
    "Graph",
    "Group",
    "Network",
    "NetworkError",
    "NextPort",
    "Node",
    "Port",
    "PrevPort",
    "Run",
    "StateDict",
    "TensorSpec",
    "__version__",
    "run",
]

__version__ = importlib.metadata.version("mujou")
