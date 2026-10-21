"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

import importlib.metadata

# The submodule, so that `import mujou` is enough for `mujou.time.Timer`. Not in
# __all__: a star import would shadow the standard `time`.
from mujou import time as time
from mujou._core.errors import NetworkError
from mujou._core.graph import Graph
from mujou._core.network import Group, Network
from mujou._core.node import Node
from mujou._core.port import Channel, MultiPort, Port
from mujou._core.run import Run, run
from mujou._core.spec import DictSpec, MessageSpec, TensorSpec
from mujou._core.state import StateDict

__all__ = [
    "Channel",
    "DictSpec",
    "Graph",
    "Group",
    "MessageSpec",
    "MultiPort",
    "Network",
    "NetworkError",
    "Node",
    "Port",
    "Run",
    "StateDict",
    "TensorSpec",
    "__version__",
    "run",
]

__version__ = importlib.metadata.version("mujou")
