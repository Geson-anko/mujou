# Saving and Restoring State

This page shows how to take the state of a running network, save it, and later run the network again from that state. What the state tree holds, and how time stops while the state is taken, is in [State](../concepts.md#state).

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

- **The timer goes back to the time of the tree.** On the default global timer, restoring sets the global time (`mujou.time.time()`) back to the time the tree was taken. When `mujou.run(..., state=tree)` returns, the time has already gone back. Do not compare a `mujou.time.time()` read before the restore with a time read after it. Measure elapsed time as the difference between two times read after the restore
- **You cannot restore while another run is running on the same timer.** In that case `mujou.run` raises `ValueError` and starts nothing. End the other run first, or give the restored run a timer of its own, `mujou.time.Timer()` (see [Controlling time](time.md))
- The restored network must have the same node paths and connections as the network the tree was taken from. Otherwise `mujou.run` raises `ValueError`
- If you pass both `state=` and `inputs=`, the values in `inputs` override the input values in the tree

## The state is not saved after an exception

When a run ends with an exception, `run.state_dict()` raises `NetworkError` and returns no state (see [Errors and shutdown](errors.md)). If you do not want to lose the state of a long run, take `run.state_dict()` regularly while it runs and save it.

## Putting other values in the state

Only `torch.Tensor` and `nn.Module` values assigned directly to attributes are collected (see [How Mujou works](../concepts.md#state)). The following are not collected.

- Number attributes (such as a step counter)
- Lists and dicts of tensors
- The state of the random number generator `self.generator`
- The state of an optimizer (such as moments) and the gradients accumulated in `.grad`

To put these in the state, override both `state_dict` and `load_state_dict`. Call `super()`, and the automatically collected part is handled as before. For a number, you can instead keep it as a 0-dimensional tensor created in `on_start` (`self.count = torch.zeros((), device=self.device)`) and update it in place (`self.count += 1`), without overriding anything. It is then collected automatically.

The following example adds the state of the random number generator. The generator is created again from `seed` every time the run starts. Without its state, the random numbers after a restore differ from those of a run that went on without stopping.

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

| Hook                     | Rules                                                                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `state_dict()`           | Return an object that pickles. Do not call `recv` or `send` on a port in it (that raises `RuntimeError`). You may put the live tensors in it; Mujou copies the tree |
| `load_state_dict(state)` | `state` is a copy of what `state_dict()` returned. It is called on the node's thread, after `on_start` and before the first `on_update`                             |

Do not change the state in `on_stop`. After a normal end, `run.state_dict()` copies the state as the nodes left it, after `on_stop`, with the time at which `shutdown` asked the run to stop. A change made in `on_stop` would go into the tree (the reason is in [State](../concepts.md#state)).

## Formats

The tree `run.state_dict()` returns is a `mujou.StateDict`. It is a subclass of `dict`: you index it as a dict, and you can save it with pickle or with `torch.save`. A tree read back is a `mujou.StateDict` again. For your own files, either will do. To share a tree with others, save it with `torch.save`, so that they can read it with the default `torch.load`. `mujou.run(..., state=)` takes a `mujou.StateDict`, or a plain dict of the same items.

Read a file back the same way it was saved (`pickle.loads` for `pickle.dumps`, `torch.load` for `torch.save`). If you do not know how a file was saved, read it first with the default `torch.load`. If that fails, read it with pickle only when you trust the sender.

- `torch.load` with its default (`weights_only=True`) reads a tree made only of tensors, numbers, strings, dicts, lists and tuples. It reads a `mujou.StateDict` too, once `mujou` is imported. If you override `state_dict` to put objects of your own classes in the tree, the default `torch.load` cannot read it
- `pickle.load` and `torch.load(..., weights_only=False)` can run code from the file while reading it. Do not read files you do not trust with them
- The state of a node that computes on CUDA goes into the tree as CUDA tensors. `tree.cpu()` returns a new tree with every tensor of the tree on the CPU (the tree itself is not changed). Move the tree to the CPU before saving it, and a machine without a GPU can read it too. To read a file saved with CUDA tensors, use `torch.load(..., map_location="cpu")`. The default `load_state_dict` copies to the node's device, so you can pass the CPU tensors to `state=` as they are

```python
torch.save(run.state_dict().cpu(), "state.pt")  # the state of GPU nodes as CPU tensors
tree = torch.load("state.pt")  # a mujou.StateDict
```

`tree.to(device)` moves every tensor of the tree to `device`, and `cpu()` is `to("cpu")`. It looks into the dicts (including the `OrderedDict` of the `state_dict` of an `nn.Module`), lists and tuples of the tree, and makes these containers anew. Other values are shared with the original tree. As with `torch.Tensor.to`, a tensor already on `device` is not copied but shared with the original tree
