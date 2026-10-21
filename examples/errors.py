"""What happens when a node raises.

The whole run stops: every node's ``on_exception``, then its ``on_stop``, is
called, and the host gets a ``NetworkError`` whose ``__cause__`` is the node's
exception. An exception in ``on_start`` comes from ``mujou.run`` itself.
"""

import logging
import sys
from typing import override

import torch

import mujou
from mujou import NetworkError, Node, Port, TensorSpec

SCALAR = TensorSpec(())

# --8<-- [start:logging]
# mujou also logs the node's exception, with its traceback, to the "mujou" logger.
# Show the log on stdout here (without a handler, Python prints it to stderr).
logging.basicConfig(stream=sys.stdout, format="log: %(message)s")
# --8<-- [end:logging]


class Checked(Node):
    """Passes its input on, and raises once it is negative."""

    def __init__(self, fail_on_start: bool = False) -> None:
        super().__init__()
        self.fail_on_start = fail_on_start
        self.prev = Port(recv=SCALAR)
        self.next = Port(send=SCALAR)

    @override
    def on_start(self) -> None:
        super().on_start()
        if self.fail_on_start:
            raise RuntimeError("cannot start")

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        u = self.prev.recv()
        if u < 0:
            raise ValueError(f"negative input: {u.item()}")
        self.next.send(u)

    @override
    def on_exception(self) -> None:
        super().on_exception()
        print("on_exception: called on every node")

    @override
    def on_stop(self) -> None:
        super().on_stop()
        print("on_stop: called after on_exception")


class Single(mujou.Network):
    def __init__(self, fail_on_start: bool = False) -> None:
        super().__init__()
        self.checked = Checked(fail_on_start)
        self.prev = self.checked.prev


# An exception while the run goes on reaches the host at its next call.
net = Single()
run = mujou.run(net)
run.write({net.prev: torch.tensor(-1.0)})
try:
    run.join(timeout=10.0)  # waits until the run stops (here: by the exception)
except NetworkError as error:
    cause = error.__cause__
    print(f"NetworkError, caused by {type(cause).__name__}: {cause}")
    assert isinstance(cause, ValueError)

# An exception in on_start comes from mujou.run.
try:
    mujou.run(Single(fail_on_start=True))
except NetworkError as error:
    print(f"mujou.run raised NetworkError, caused by {error.__cause__!r}")
    assert isinstance(error.__cause__, RuntimeError)
