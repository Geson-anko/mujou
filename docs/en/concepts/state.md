# State

Mujou can take the state of a running network without stopping it, and later run the network again from that state. Unlike the conventional way of saving weights after training ends, it saves the state as it is, in the middle of continuous learning.

## The state tree

`run.state_dict()` returns the state of the network as one tree (nested dicts). The tree holds the following.

| Key              | Content                                                                               | Who puts it in             |
| ---------------- | ------------------------------------------------------------------------------------- | -------------------------- |
| `nodes`          | The state of each node (weights, activities, ...), by node path                       | Each node's `state_dict()` |
| `slots`          | The latest value of each value slot and its time (including the host's inputs)        | Mujou, automatically       |
| `last_update_ns` | The time of each node's last update, in ns, by node path (where the next `dt` starts) | Mujou, automatically       |
| `timer`          | The time and scale of the run's timer (`time_ns` and `scale`)                         | Mujou, automatically       |

The port values are in the tree because the state of a network is the state of its nodes together with the values flowing between them. If the port values went back to their initial values on a restart, the neighbors would read initial values and leave the equilibrium they had reached.

The keys of `nodes` are the node paths (such as `"body.l1"`; see [Networks](networks.md)). For example, take the state of `l1` in a run of `mujou.run(net)` with `tree["nodes"]["l1"]`. For a single node run with `mujou.run(node)`, use `tree["nodes"][""]`.

## How a node collects its state

Like `nn.Module`, `mujou.Node` collects its state by itself from the values assigned to its attributes.

| Value assigned                                      | In the state?                      |
| --------------------------------------------------- | ---------------------------------- |
| `torch.Tensor` (including `nn.Parameter`)           | Yes                                |
| `nn.Module`                                         | Yes (that module's `state_dict()`) |
| A number, `None`, a port, a list or dict of tensors | No                                 |

An attribute is collected whether or not its name starts with `_`. Assigning `None` or the like to a collected attribute, or deleting it with `del`, removes it from the state. The state of an optimizer or a random number generator is not collected automatically. To put anything else in the state, override `state_dict` and `load_state_dict`.

Create the state tensors in `on_start`, and update them in place afterwards.

Do not change the state in `on_stop`. After a normal end, `run.state_dict()` promises to return the state at the moment `run.shutdown()` asked the run to stop. `on_stop` is called after that, so a change made in `on_stop` would leak into the tree taken after the end. For example, `self.w = None` in `on_stop` removes `w` from the state, and `w` disappears from the tree taken after the end. Release resources such as I/O in `on_stop`, and keep the state tensors.

## When the state can be taken

| The network is          | `run.state_dict()`                                                   |
| ----------------------- | -------------------------------------------------------------------- |
| Running                 | Works. Returns the state at that moment                              |
| Stopped normally        | Works. Returns the state at the moment `shutdown()` asked it to stop |
| Stopped by an exception | Does not work. Raises `NetworkError`                                 |

The state at an exception is not saved. Take the state you want to keep while the network runs or after it stops normally.

## Time stops while the state is taken

To take the state of a running network, Mujou does the following, in order.

1. It asks every node of the run to pause, and stops the run's timer at once
2. Each node finishes the running `on_update` before it pauses, then `on_pause()` is called. A node never pauses in the middle of `on_update`
3. Once all nodes have paused, it calls `state_dict()` on every node and makes a copy of the tree
4. It resumes the timer and the nodes. Each node has `on_resume()` called before it goes on to the next `on_update`
5. It returns the copied tree to the host

The nodes of other runs on the same timer are not asked to pause, and `on_pause()` is not called on them. Their time stops too, so they start no new `on_update` until the timer resumes, and the pause does not enter their `dt`.

Because the timer is stopped while the nodes are paused, the pause does not enter `dt` after resuming (see [Global time](time.md)). To the nodes' updates, taking the state is as if it never happened. And because the tree is taken while all nodes are paused, every value in it is the state at the same time.

The tree is a copy, so it does not change in the host's hands when the nodes go on updating. Where and how to store the tree (`pickle`, `torch.save`, ...) is up to you. Neither Mujou nor the nodes handle files.

## Restoring

To run again from a saved tree, pass it as `mujou.run(net, state=tree)`. Mujou calls `on_start()` on each node, then, before the first `on_update`, restores the state with each node's `load_state_dict()`. It also restores the port values and the timer's time, and starts running from the time the tree was taken.

The network you restore must have the same node paths and the same connections as the network the tree was taken from. Otherwise `mujou.run` raises `ValueError` and starts nothing.

How to save and restore step by step is in [Saving and restoring state](../user-guide/state.md).
