# 💾 Saving and Restoring State

This page shows how to take the state of a running network, save it, and later run the network again from that state.

What the state tree holds, and how time stops while the state is taken, is in [State](../concepts.md#state).

## The basic steps

The following example runs a network for 0.2 seconds, takes its state, saves it with `torch.save`, restores it into a new instance, and goes on running.

```python
--8<-- "examples/save_and_restore.py"
```

The steps are as follows.

1. The node keeps the values of its state in attributes that hold a `torch.Tensor` or an `nn.Module`. `mujou.Node` collects them into the state automatically, with the state of its random generator (`self.x` and `self.readout` in the example)
2. The node creates the state tensors in `__init__`, and updates them in place in `on_update` (`+=`, `add_`, `copy_`, ...)
3. The host takes the state tree with `run.state_dict()`. It works while the network runs, and also after the `with` statement exits (after a normal end)
4. The host saves the tree with `torch.save` or the like. How to save it is up to the host
5. To restore, create a **new instance** of the same network and pass it as `mujou.run(net, state=tree)`

Step 2 updates in place because the default `load_state_dict` restores the state by copying into the tensors the node holds. With in-place updates, any other attribute or local variable that refers to the same tensor also sees the restored values.

Step 5 needs a new instance because a network can run only once. Passing the network the tree was taken from to `mujou.run` again raises `RuntimeError`.

## When you restore

- **The restored run has a timer of its own.** It runs on a new `Timer` (`run.timer`) that starts at the time and the time scale of the tree, so the global timer and the other runs do not go back. Passing `timer=` with `state=` raises `ValueError`. `mujou.time.global_timer.pause()` does not stop a restored run: pause its `run.timer` (see [Controlling time](time.md))
- The restored network must have the same node paths, ports and connections as the network the tree was taken from. Otherwise `mujou.run` raises `ValueError`, which lists what is missing and what is in excess. The state of each node must match too: the default `load_state_dict` raises `ValueError` for other keys or shapes
- If you pass both `state=` and `inputs=`, the values in `inputs` override the input values in the tree

## Taking the states of several runs at one time

`run.state_dict()` takes the state of one run at one time. When several runs share a timer (by default, all of them run on the global timer) and you take their states one after another, each state is of a later time than the one before, because the time goes on in between.

To take them all at one time, pause the timer first, take the states, and resume it.

```python
--8<-- "examples/pause.py:pause"
```

`run.state_dict()` works while the timer is paused, and every tree then holds the time of the pause. While paused, the nodes do not update, and the paused time does not enter the `dt` after `resume` (see [Pausing the time](time.md#pausing-the-time)).

!!! tip "Put resume in finally"

    Put the `resume` in `finally`, so that an exception does not leave the time stopped.

Each restored run goes on from the time of its tree on a timer of its own, so the trees can be restored together. The restored runs no longer share a time axis, though: their timers drift apart, and pausing one does not pause the others. Runs that must stay on one time axis after a restore run as one run (`mujou.run(Group(a=a, b=b))`), whose single tree restores onto one timer.

## The state is not saved after an exception

When a run ends with an exception, `run.state_dict()` raises `NetworkError` and returns no state (see [Errors and shutdown](errors.md)).

!!! tip "Save regularly during a long run"

    If you do not want to lose the state of a long run, take `run.state_dict()` regularly while it runs and save it.

## Putting other values in the state

Only `torch.Tensor` values and objects with `state_dict` and `load_state_dict` (an `nn.Module`, say) assigned directly to attributes are collected, with the state of `self.generator` (see [How Mujou works](../concepts.md#state)). The following are not collected.

- Number attributes (such as a step counter)
- Lists and dicts of tensors
- The settings that `__init__` computes from its arguments (a time constant, say): a node restored with other arguments keeps its own

For a value that changes while the node runs, such as a number, a string, `None`, or a list or dict of them, assign the attribute and register its name with `register_state`. The value then goes into the state as it is, whatever you assign to the attribute later.

```python
self.elapsed = 0.0
self.register_state("elapsed")
```

The default `load_state_dict` assigns a copy of the saved value to the attribute. Register only an attribute that already exists (otherwise `AttributeError`), and register it in `__init__`: a restored run uses new node instances, which must have the same state. Keep the values to basic types if the tree is to be read with the default `torch.load` (see [Formats](#formats)).

For the rest, override both `state_dict` and `load_state_dict`: extend what `super().state_dict()` returns, and pass `super().load_state_dict` the state without the keys you added.

```python
from collections.abc import Mapping
from typing import Any, override

from mujou import Node


class Recent(Node):
    """Keeps the last few inputs in a list."""

    @override
    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "recent": list(self.recent)}

    @override
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.recent = [x.clone() for x in state["recent"]]  # a copy of the tree's
        super().load_state_dict({k: v for k, v in state.items() if k != "recent"})
```

When you override them, follow these rules.

| Hook                     | Rules                                                                                                                                                                                                                                |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `state_dict()`           | Return a dict that pickles. Do not call `recv` or `send` on a port in it: the tree holds the slots apart, and a send would change the run while its state is taken. You may put the live tensors in it; Mujou copies what it returns |
| `load_state_dict(state)` | `state` is what `state_dict()` returned, read from the tree: copy what you keep. It is called after `on_start` and before the first `on_update`                                                                                      |

!!! warning "Do not change the state in on_stop"

    Do not change the state in `on_stop`. After a normal end, `run.state_dict()` copies the state as the nodes left it, after `on_stop`, with the time at which `shutdown` asked the run to stop. A change made in `on_stop` would go into the tree (the reason is in [State](../concepts.md#state)).

## Formats

The tree `run.state_dict()` returns is a plain `dict` whose tensors are copies on the CPU, also those of the nodes that compute on CUDA. You index it as a dict, and you can save it with pickle or with `torch.save`. The shape of the tree is in the API reference of `Run.state_dict`.

For your own files, either will do. To share a tree with others, save it with `torch.save`, so that they can read it with the default `torch.load`.

Read a file back the same way it was saved (`pickle.loads` for `pickle.dumps`, `torch.load` for `torch.save`). If you do not know how a file was saved, read it first with the default `torch.load`. If that fails, read it with pickle only when you trust the sender.

- `torch.load` with its default (`weights_only=True`) reads a tree made only of tensors, numbers, strings, dicts, lists and tuples. If objects of your own classes go into the tree (from a registered attribute, or from an override of `state_dict`), the default `torch.load` cannot read it
- `pickle.load` and `torch.load(..., weights_only=False)` can run code from the file while reading it. Do not read files you do not trust with them
- The default `load_state_dict` copies the tensors to the node's device. The state of `generator` has a different format on the CPU and on CUDA, so restore a tree into nodes on the same kind of device as the nodes that saved it: a tree saved from nodes on CUDA restores into nodes on CUDA, on another machine with a GPU too

```python
torch.save(run.state_dict(), "state.pt")
tree = torch.load("state.pt")  # a plain dict
```
