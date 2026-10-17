# Nodes

A node is the unit of computation in Mujou. You write a node as a class that inherits from `mujou.Node`.

## Synchronous inside, asynchronous outside

Inside a node, ordinary PyTorch code runs synchronously. It can be matrix arithmetic, or BP closed inside the node.

Nodes talk to each other only by reading and writing values through [ports](ports.md). A node never waits for its neighbors. It reads whatever value is latest at that moment. If the neighbor has not written since, the node reads the same value as before; if the neighbor has never written, it reads the initial value.

A node keeps three rules.

- Do not join computation graphs across nodes. You cannot write a `backward()` that crosses nodes. A value is detached from the computation graph when it is sent, so gradients never pass through a port
- Talk to other nodes and to the host only through the node's own ports. Calling `mujou.run` or a method of `Run` (`write`, `read`, `state_dict`, and so on) from a hook raises `RuntimeError`
- Only read the time. A node reads the time with `self.timer.now()`. Calling an operation that moves the time (`mujou.time.sleep`, `mujou.time.set_time_scale`, and so on) from a hook raises `RuntimeError` (see [Global time](time.md))

## The size of a node

How much computation goes into one node depends on how you write the node class. You cannot split or merge nodes by changing a setting.

| Size  | Example                                              |
| ----- | ---------------------------------------------------- |
| Layer | One fully connected layer per node. The default view |
| Stack | Several layers, or a whole model, in one node        |

The asynchronous boundaries are the node boundaries. To compute several layers synchronously, waiting for each other, put them in one node class.

## Lifecycle hooks

You write a node by overriding the methods that Mujou calls (hooks). Mujou owns the loop; the node's author writes one step in `on_update(dt)`. Do not write your own `while` loop.

The hooks are called in this order.

1. `on_thread_start()`: once, when the run starts. Make settings that apply only to this node's own thread, such as torch's per-thread settings (see [PyTorch in Nodes](../user-guide/torch.md))
2. `on_start()`: once, when the run starts. Allocate and initialize the state tensors
3. `load_state_dict(state)`: only when the run restores a state, once, after `on_start` (see [State](state.md))
4. `on_update(dt)`: called repeatedly. Read the neighbors' latest values from the ports, update the state by `dt` seconds, and write the results to the ports
5. `on_exception()`: only when a hook of any node raises, once on every node. Put a safe stop of actuators and the like here
6. `on_stop()`: once, when the run ends, also when it ends with an exception. Release resources such as I/O

The following hooks are called outside that order, when needed.

- `on_pause()`: when the network pauses, after the running `on_update` has finished. A pause happens when the state is taken, for example (see [State](state.md))
- `on_resume()`: when the network resumes from a pause, before the next `on_update`
- `state_dict()`: when the state is taken

By default, `on_update` repeats as fast as it can. Pass a period, as in `super().__init__(period=0.01)`, and it is called once every that many seconds on average. Even with a period, `dt` is the time that actually passed since the last update (see [Global time](time.md)).

If any hook of any node raises, the whole network stops. Mujou never stops only the node that raised and keeps the others running.

Declare the ports in `__init__`. Create the state tensors in `on_start`, and from then on update those same tensors in place.

## Hooks that do not exist

There is no hook for the start or end of an epoch, and no hook that switches between training and inference. There is also no hook called for each received message. A node does not wait for messages; it reads the latest values at each `on_update`.

## Example

This node receives `x` from the previous node, moves its state `h` towards `x @ w` over time, and sends `h` to the next node.

```python
from typing import override

import torch

import mujou


class Leaky(mujou.Node):
    def __init__(self, n_in: int, n_out: int) -> None:
        self.prev = mujou.PrevPort(recv=mujou.TensorSpec((32, n_in)))   # receive only
        self.next = mujou.NextPort(send=mujou.TensorSpec((32, n_out)))  # send only
        self.n_in, self.n_out = n_in, n_out

    @override
    def on_start(self) -> None:
        # Create the state tensors in on_start. Tensor attributes are collected as state
        self.w = torch.randn(self.n_in, self.n_out, generator=self.generator) * 0.1
        self.h = torch.zeros(32, self.n_out)

    @override
    def on_update(self, dt: float) -> None:
        x = self.prev.recv()                  # the previous node's latest value; no waiting
        self.h += dt * (x @ self.w - self.h)  # in proportion to dt, in place
        self.next.send(self.h)
```

How to write a node in detail is in [Writing a node](../user-guide/writing-a-node.md).
