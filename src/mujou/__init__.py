"""Mujou: asynchronous, continually learning deep learning trained with PC-
ALM."""

# The submodule, so that `import mujou` is enough for `mujou.time.sleep`. Not in
# __all__: a star import would shadow the standard `time`.
from mujou import time as time
from mujou._core._native import NetworkError, PauseTimeoutError, __version__
from mujou._core.network import Network, run
from mujou._core.node import Node
from mujou._core.port import Channel, NextPort, Port, PrevPort
from mujou._core.spec import DictSpec, TensorSpec

__all__ = [
    "Channel",
    "DictSpec",
    "Network",
    "NetworkError",
    "NextPort",
    "Node",
    "PauseTimeoutError",
    "Port",
    "PrevPort",
    "TensorSpec",
    "__version__",
    "run",
]
