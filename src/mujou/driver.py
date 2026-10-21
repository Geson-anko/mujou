"""The entry of one node for the runtime: its ``Driver`` (``specs/nodes.md``).

The runtime calls one method of the driver for each hook of the node, on the
thread that runs the node, and ``current.running()`` returns the driver
meanwhile. Every method does the framework's work around the hook in the same
order, so that no override of a hook can skip it:

1. on CUDA, make the node's device and stream current (and, around
   ``state_dict``, set the calling thread's back afterwards);
2. enter the node's own ``contextvars.Context``;
3. call the hook, looked up on the node at each call;
4. after ``on_update`` and ``load_state_dict``, wait for the work queued on the
   node's stream.

``capture`` copies what ``state_dict`` returns on the node's stream, so that the
copy stays as taken once the node resumes. The state is data: its tensors are
detached before the copy, and no graph goes with them.
"""

import contextvars
import copy
from collections.abc import Callable, Mapping
from typing import Final, final

import torch

from mujou._native import ReadOnlyTimer
from mujou.node import Node, map_tensors
from mujou.port import AnyPort, Wire

# The wires of a node in its state_dict: none, so that taking the state is not
# seen by the other nodes.
_NO_WIRES: Final[Mapping[AnyPort, Wire]] = {}


@final
class Driver:
    """What the runtime calls for the hooks of ``node``, which runs on
    ``clock`` and whose ports read and write through ``wires``."""

    def __init__(
        self, node: Node, clock: ReadOnlyTimer, wires: Mapping[AnyPort, Wire]
    ) -> None:
        self.node = node
        self.clock = clock
        # What the ports of the running hook find (port.py).
        self.wires = wires
        # Empty, as a new thread's: the hooks do not see the host's variables,
        # and keep their own whichever worker runs them.
        self._context = contextvars.Context()
        device = node.device
        # Each wait records an event on the stream and waits for that, not for
        # the whole stream: nodes past PyTorch's pool of 32 streams per device
        # share one, and the event does not wait for work that another node
        # queues on it later.
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def start(self) -> None:
        self._run(self.node.on_start)

    def restore(self, state: object) -> None:
        # The copies queue on the node's stream; wait for them, so that the host
        # may drop the tree it passed (whose memory they read) once run returns.
        self._run(self.node.load_state_dict, state)
        if self._stream is not None:
            self._stream.record_event().synchronize()

    def update(self, dt: float) -> None:
        # The hot path: what _run does, without its call.
        stream = self._stream
        if stream is not None:
            _use(stream)
        self._context.run(self.node.on_update, dt)
        if stream is not None:
            # The CPU never runs ahead of the GPU.
            stream.record_event().synchronize()

    def pause(self) -> None:
        self._run(self.node.on_pause)

    def resume(self) -> None:
        self._run(self.node.on_resume)

    def exception(self) -> None:
        self._run(self.node.on_exception)

    def stop(self) -> None:
        self._run(self.node.on_stop)

    def capture(self) -> object:
        stream = self._stream
        if stream is None:
            # Not torch.cuda.stream(None), which initializes CUDA.
            return self._copy()
        # Unlike the other hooks, it may run on a thread the node does not own
        # (the host's, after the run ends): torch.cuda.stream sets that thread's
        # device and stream back on exit.
        with torch.cuda.stream(stream):
            # Wait for the work of every hook (not only on_update) before
            # state_dict reads it, and for the copies before the node resumes.
            stream.record_event().synchronize()
            state = self._copy()
            stream.record_event().synchronize()
        return state

    def _run[T](self, hook: Callable[..., T], *args: object) -> T:
        if self._stream is not None:
            _use(self._stream)
        return self._context.run(hook, *args)

    def _copy(self) -> object:
        wires, self.wires = self.wires, _NO_WIRES
        try:
            return self._context.run(self._detached_state)
        finally:
            self.wires = wires

    def _detached_state(self) -> object:
        """A deep copy of the node's ``state_dict``, its tensors detached."""
        return copy.deepcopy(map_tensors(self.node.state_dict(), torch.Tensor.detach))


def _use(stream: torch.cuda.Stream) -> None:
    """Make ``stream`` and its device current on this thread: per thread, so set
    again before each hook, as a worker runs other nodes in between."""
    torch.cuda.set_device(stream.device)
    torch.cuda.set_stream(stream)
