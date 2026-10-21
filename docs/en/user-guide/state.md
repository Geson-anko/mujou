# 💾 Saving and Restoring State

This page shows how to take the state of a running network, save it, and later run the network again from that state.

What the state tree holds, and how time stops while the state is taken, is in [State](../concepts.md#state).

## The basic steps

The following example runs a network for 0.2 seconds, takes its state, saves it with `torch.save`, restores it into a new instance, and goes on running.

```python
--8<-- "examples/save_and_restore.py"
```

The steps are as follows.

1. The node keeps the values of its state in attributes that hold a `torch.Tensor` or an `nn.Module`. `mujou.Node` collects them into the state automatically (`self.x` and `self.readout` in the example)
2. The node creates the state tensors in `on_start`, and updates them in place in `on_update` (`+=`, `add_`, `copy_`, ...)
3. The host takes the state tree with `run.state_dict()`. It works while the network runs, and also after the `with` statement exits (after a normal end)
4. The host saves the tree with `torch.save` or the like. How to save it is up to the host
5. To restore, create a **new instance** of the same network and pass it as `mujou.run(net, state=tree)`

Step 2 updates in place because the default `load_state_dict` restores the state by copying into the tensors created in `on_start`. With in-place updates, any other attribute or local variable that refers to the same tensor also sees the restored values.

Step 5 needs a new instance because a network can run only once. Passing the network the tree was taken from to `mujou.run` again raises `RuntimeError`.

## When you restore

- **The restored run has a timer of its own.** It runs on a new `Timer` (`run.timer`) that starts at the time and the time scale of the tree, so the global timer and the other runs do not go back. Passing `timer=` with `state=` raises `ValueError`. `mujou.time.global_timer.pause()` does not stop a restored run: pause its `run.timer` (see [Controlling time](time.md))
- The restored network must have the same node paths and connections as the network the tree was taken from. Otherwise `mujou.run` raises `ValueError`
- If you pass both `state=` and `inputs=`, the values in `inputs` override the input values in the tree
- Data carried along a path ([Carrying data along a path](building-a-network.md#carrying-data-along-a-path)) is in the tree too: with the messages that carry it, and, for each port with a `recv_carry`, the data of the message it last received. A restored node passes on the same data as one that never stopped

## Taking the states of several runs at one time

`run.state_dict()` takes the state of one run at one time. When several runs share a timer (by default, all of them run on the global timer) and you take their states one after another, each state is of a later time than the one before, because the time goes on in between.

To take them all at one time, pause the timer first, take the states, and resume it.

```python
--8<-- "examples/pause.py:pause"
```

`run.state_dict()` works while the timer is paused, and every tree then holds the time of the pause. While paused, the nodes do not update, and the paused time does not enter the `dt` after `resume` (see [Pausing the time](time.md#pausing-the-time)).

!!! tip "Put resume in finally"

    Put the `resume` in `finally`, so that an exception does not leave the time stopped.

Each restored run goes on from the time of its tree on a timer of its own, so the trees can be restored together. The restored runs no longer share a time axis, though: their timers drift apart, and pausing one does not pause the others. Runs that must stay on one time axis after a restore run as one run of several components (`mujou.run(a, b)`), whose single tree restores onto one timer.

## The state is not saved after an exception

When a run ends with an exception, `run.state_dict()` raises `NetworkError` and returns no state (see [Errors and shutdown](errors.md)).

!!! tip "Save regularly during a long run"

    If you do not want to lose the state of a long run, take `run.state_dict()` regularly while it runs and save it.

## Putting other values in the state

Only `torch.Tensor` and `nn.Module` values assigned directly to attributes are collected (see [How Mujou works](../concepts.md#state)). The following are not collected.

- Number attributes (such as a step counter)
- Lists and dicts of tensors
- The state of the random number generator `self.generator`
- The state of an optimizer (such as moments) and the gradients accumulated in `.grad`

For a number, a string, `None`, or a list or dict of them, assign the attribute and register its name with `register_state`. The value then goes into the state as it is, and stays in it whatever you assign to the attribute later, until `del`.

```python
self.elapsed = 0.0
self.register_state("elapsed")
```

The default `load_state_dict` assigns a copy of the saved value to the attribute. Register only an attribute that already exists, and register it in `__init__` or `on_start`: a restored run uses new node instances, and a name first registered later (in `on_update`, say) is not registered there yet, so the restore rejects it. Keep the values to basic types if the tree is to be read with the default `torch.load` (see [Formats](#formats)).

For the rest, override both `state_dict` and `load_state_dict`. Call `super()`, and the automatically collected and registered part is handled as before.

The following example adds the state of the random number generator. A restored run uses new node instances, whose generators start again from `seed`. Without its state, the random numbers after a restore differ from those of a run that went on without stopping.

```python
from typing import Any, override

from mujou import Node


class Noisy(Node):
    @override
    def state_dict(self) -> Any:
        state = super().state_dict()
        state["generator"] = self.generator.get_state()
        return state

    @override
    def load_state_dict(self, state: Any) -> None:
        state = dict(state)  # copy first, so the given tree is not modified
        self.generator.set_state(state.pop("generator"))
        super().load_state_dict(state)
```

When you override them, follow these rules.

| Hook                     | Rules                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_dict()`           | Return an object that pickles. Do not call `recv` or `send` on a port in it: the tree holds the slots apart, and a send would change the run while its state is taken. You may put the live tensors in it; Mujou copies the tree |
| `load_state_dict(state)` | `state` is a copy of what `state_dict()` returned. It is called after `on_start` and before the first `on_update`                                                                                                                |

!!! warning "Do not change the state in on_stop"

    Do not change the state in `on_stop`. After a normal end, `run.state_dict()` copies the state as the nodes left it, after `on_stop`, with the time at which `shutdown` asked the run to stop. A change made in `on_stop` would go into the tree (the reason is in [State](../concepts.md#state)).

## Formats

The tree `run.state_dict()` returns is a `mujou.StateDict`. It is a subclass of `dict`: you index it as a dict, and you can save it with pickle or with `torch.save`. A tree read back is a `mujou.StateDict` again.

For your own files, either will do. To share a tree with others, save it with `torch.save`, so that they can read it with the default `torch.load`. `mujou.run(..., state=)` takes a `mujou.StateDict`, or a plain dict of the same items.

Read a file back the same way it was saved (`pickle.loads` for `pickle.dumps`, `torch.load` for `torch.save`). If you do not know how a file was saved, read it first with the default `torch.load`. If that fails, read it with pickle only when you trust the sender.

- `torch.load` with its default (`weights_only=True`) reads a tree made only of tensors, numbers, strings, dicts, lists and tuples. It reads a `mujou.StateDict` too, once `mujou` is imported. If objects of your own classes go into the tree (from a registered attribute, or from an override of `state_dict`), the default `torch.load` cannot read it
- `pickle.load` and `torch.load(..., weights_only=False)` can run code from the file while reading it. Do not read files you do not trust with them
- The state of a node that computes on CUDA goes into the tree as CUDA tensors. `tree.cpu()` returns a new tree with every tensor of the tree on the CPU (the tree itself is not changed). Move the tree to the CPU before saving it, and a machine without a GPU can read it too. To read a file saved with CUDA tensors, use `torch.load(..., map_location="cpu")`. The default `load_state_dict` copies to the node's device, so you can pass the CPU tensors to `state=` as they are

```python
torch.save(run.state_dict().cpu(), "state.pt")  # the state of GPU nodes as CPU tensors
tree = torch.load("state.pt")  # a mujou.StateDict
```

`tree.to(device)` moves every tensor of the tree to `device`, and `cpu()` is `to("cpu")`.

It looks into the dicts (including the `OrderedDict` of the `state_dict` of an `nn.Module`), lists and tuples of the tree, and makes these containers anew. Other values are shared with the original tree. As with `torch.Tensor.to`, a tensor already on `device` is not copied but shared with the original tree
