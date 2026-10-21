# 💾 6. Save and Resume

This page takes the state of a run, saves it, and resumes from it in a new run. What the state holds is in [How Mujou Works](../concepts.md#7-state).

## The steps

```python
--8<-- "examples/save_and_restore.py"
```

1. The node keeps its state in attributes: `self.x` (a tensor) and `self.readout` (a module) are collected by themselves, with the state of the node's random generator
2. It creates them in `__init__` and updates them in place in `on_update`, so that other references to them (from an optimizer, say) keep pointing at the state (see [2. Write a Node](2-node.md))
3. The host takes the tree with `run.state_dict()`, while the run runs or after it ended normally (here, after the `with` block)
4. It saves the tree with `torch.save`
5. To resume, it builds a **new instance** of the same network, not the one the tree came from, since a network runs once ([rules a node keeps](../concepts.md#rules-a-node-keeps)), and passes the tree as `mujou.run(net, state=tree)`

## The tree

The tree is a plain `dict` of copies on the CPU, keyed by the paths of the nodes: the state of each node, the latest message on each channel with its time, the time of each node's last update, and the time of the timer. Its exact shape is in the API reference of [`Run.state_dict`](../api/run.md#mujou.Run.state_dict).

Taking it pauses the run: each node stops between two updates, never in the middle of one, so every value in the tree is of the same time. The paused time does not enter any `dt` afterwards. Each node's `on_pause` and `on_resume` run around this pause, to hold its I/O meanwhile; a pause of the timer (`run.timer.pause()`) calls neither.

| When you take it       | What you get                                                                                                                     |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| While the network runs | The state at that time                                                                                                           |
| After a normal end     | The values as the nodes left them after `on_stop`, at the time `shutdown` asked the run to stop                                  |
| After an exception     | Nothing: `state_dict` raises `NetworkError`. Take the state you want to keep while the network runs, regularly during a long run |

## When you resume

- **The resumed run has a timer of its own.** It runs on a new `Timer` (`run.timer`) that starts at the time and the time scale of the tree, so `mujou.time.global_timer` and the other runs do not go back. Passing `timer=` with `state=` raises `ValueError`, and `global_timer.pause()` does not pause a resumed run: pause its `run.timer`
- **Build the same network.** The resumed network must have the node paths, ports and connections of the one the tree came from; otherwise `mujou.run` raises `ValueError`, listing what is missing and what is in excess. Renaming an attribute changes a path, so a tree saved before the rename no longer resumes. In both cases nothing has started, and the same instances run once the tree is fixed
- **Keep the state of each node.** A node state with other keys or shapes makes the default `load_state_dict` raise `ValueError` while the node starts, so `mujou.run` raises `NetworkError` (its `__cause__` is the `ValueError`), and those instances do not run again: build new ones
- **Inputs.** With both `state=` and `inputs=`, the values of `inputs` replace those of the tree

## Values of your own in the state

Tensors, objects with `state_dict` and `load_state_dict`, and the state of `self.generator` are collected; numbers, strings, and lists or dicts of tensors are not. For a plain value that changes while the node runs, assign the attribute in `__init__` and register it:

```python
self.elapsed = 0.0
self.register_state("elapsed")
```

The value goes into the state as it is, whatever you assign to the attribute later. Register only an attribute that exists (otherwise `AttributeError`). Keep it to types that `torch.load` reads with its default (`int`, `float`, `bool`, `str`, `None`, and lists, tuples and dicts of them, or tensors).

For the rest, such as a `deque`, override both `state_dict` and `load_state_dict`: extend what `super().state_dict()` returns, and pass `super().load_state_dict` the state without the keys you added.

```python
from collections import deque
from collections.abc import Mapping
from typing import Any, override

import torch

from mujou import Node


class Recent(Node):
    """Keeps the last 8 inputs in a deque."""

    def __init__(self) -> None:
        super().__init__()
        self.recent: deque[torch.Tensor] = deque(maxlen=8)

    @override
    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "recent": list(self.recent)}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.recent = deque((x.clone() for x in state["recent"]), maxlen=8)  # a copy of the tree's
        super().load_state_dict({k: v for k, v in state.items() if k != "recent"})
```

`state_dict` is called while the node is paused, and its ports refuse to receive and send there, so that taking the state is not seen by the other nodes. Mujou copies what it returns. `load_state_dict` is called once, after `on_start` and before the first `on_update`.

## Files

- `torch.save` and `torch.load` with its default `weights_only=True` read a tree made of tensors, numbers, strings, dicts, lists and tuples: what Mujou puts in it. A tree that holds objects of your own classes (from `register_state` or an override) does not load that way
- `pickle.load` and `torch.load(..., weights_only=False)` can run code from the file. Do not read files you do not trust with them
- The default `load_state_dict` copies each tensor to the node's device, so a machine without a GPU can read a tree saved from nodes on CUDA. The state of a random generator differs between the CPU and CUDA, though: a tree restores only into nodes on the same kind of device as the ones that saved it

## Several runs at one time

`run.state_dict()` takes the state of one run at one time. To take the states of several runs on one timer at one time, pause the timer first; see [Time](../topics/time.md#pausing-the-time).

## Next

- The topics go deeper into each part, starting with [Ports and Channels](../topics/ports.md)
