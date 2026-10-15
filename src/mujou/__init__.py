"""Mujou: asynchronous, continually learning deep learning trained with PC-
ALM."""

# The submodule, so that `import mujou` is enough for `mujou.time.sleep`. Not in
# __all__: a star import would shadow the standard `time`.
from mujou import time as time
from mujou._core import NetworkError, PauseTimeoutError, __version__
from mujou._network import Network, run
from mujou._node import Node
from mujou._port import Channel, NextPort, Port, PrevPort
from mujou._spec import DictSpec, TensorSpec

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
