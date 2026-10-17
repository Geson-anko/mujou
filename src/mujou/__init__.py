"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

# The submodules, so that `import mujou` is enough for `mujou.time.sleep` and
# `mujou.typing.Multi`. Not in __all__: a star import would shadow the standard
# `time` and `typing`.
from mujou import time as time, typing as typing
from mujou._core._native import NetworkError, PauseTimeoutError, __version__
from mujou._core.graph import Graph
from mujou._core.network import Group, Network
from mujou._core.node import Node
from mujou._core.port import Channel, NextPort, Port, PrevPort
from mujou._core.run import Run, run
from mujou._core.spec import DictSpec, TensorSpec

__all__ = [
    "Channel",
    "DictSpec",
    "Graph",
    "Group",
    "Network",
    "NetworkError",
    "NextPort",
    "Node",
    "PauseTimeoutError",
    "Port",
    "PrevPort",
    "Run",
    "TensorSpec",
    "__version__",
    "run",
]
