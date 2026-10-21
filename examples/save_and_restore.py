"""Save the state of a run, and restore it in a new run that goes on from there.

A node collects its state by itself: every attribute that holds a
``torch.Tensor`` or an ``nn.Module``. ``Run.state_dict`` gathers the states of all
nodes, the latest messages and the time into one tree, a ``mujou.StateDict``.
Save it with ``torch.save`` (or pickle), and read it back with ``torch.load``.
"""

import io
from typing import override

import torch
from torch import nn

import mujou
from mujou import Node, Port, TensorSpec

VECTOR1 = TensorSpec((1,))


class Integrator(Node):
    """Integrates its input over time (dx/dt = u) and sends ``readout(x)``."""

    def __init__(self) -> None:
        super().__init__()
        self.prev = Port(recv=VECTOR1)
        self.next = Port(send=VECTOR1)

    @override
    def on_start(self) -> None:
        super().on_start()
        self.x = VECTOR1.zeros()  # a tensor: part of the state
        # A module: part of the state.
        self.readout = nn.Linear(1, 1, device=self.device)

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.x += dt * self.prev.recv()  # in place, so that a restore copies into it
        self.next.send(self.readout(self.x))


class Model(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.integrator = Integrator()
        self.prev = self.integrator.prev
        self.next = self.integrator.next


one = torch.ones(1)

# Run for a while, then stop and take the state at the end.
net = Model()
with mujou.run(net, inputs={net.prev: one}) as run:
    run.timer.sleep(0.2)
tree = run.state_dict()  # after a normal end: the state at the stop
saved = tree["nodes"]["integrator"]  # this node's state_dict()
print("state of the node:", {k: type(v).__name__ for k, v in saved.items()})
file = io.BytesIO()  # or a path, such as "state.pt"
# .cpu(): every tensor on the CPU, so that a machine without the GPU reads it too.
torch.save(tree.cpu(), file)

# A new instance of the same network goes on from the saved state and time, on a
# timer of its own that starts at the time of the state.
restored = Model()
file.seek(0)
loaded = torch.load(file)  # the default weights_only=True reads a StateDict
with mujou.run(restored, inputs={restored.prev: one}, state=loaded) as run:
    run.timer.sleep(0.1)
after = run.state_dict()["nodes"]["integrator"]

x0, x1 = saved["x"].item(), after["x"].item()
print(f"x = {x0:.3f} at the save, {x1:.3f} after 0.1 s more")
# The integral went on from the saved value: from 0, it would be about 0.1. (How
# far it gets in 0.2 s and 0.1 s depends on when the last update falls.)
assert x1 > x0 > 0
# The module was restored too: the new instance had other random weights.
assert torch.equal(after["readout"]["weight"], saved["readout"]["weight"])
