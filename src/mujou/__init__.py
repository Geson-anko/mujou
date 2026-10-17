"""Mujou: Asynchronous Distributed Deep Learning Library for Continuous Learning."""

# The submodule, so that `import mujou` is enough for `mujou.time.sleep`. Not in
# __all__: a star import would shadow the standard `time`.
from mujou import time as time
from mujou._core._native import NetworkError, PauseTimeoutError, __version__
from mujou._core.graph import Graph
from mujou._core.network import Compose, Network
from mujou._core.node import Node
from mujou._core.port import Channel, NextPort, Port, PrevPort
from mujou._core.run import Run, run
from mujou._core.spec import DictSpec, TensorSpec

__all__ = [
    "Channel",
    "Compose",
    "DictSpec",
    "Graph",
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
