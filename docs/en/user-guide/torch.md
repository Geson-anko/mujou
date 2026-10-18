# 🔥 PyTorch in Nodes

A node computes with PyTorch inside. This page covers update rules, devices and random numbers, backpropagation (BP) inside a node, CUDA, and the settings that affect the whole process.

## Update rules

The update a node writes in `on_update(dt)` is that node's update rule (its learning rule). Mujou builds no update rule into nodes, ports or the execution machinery, so each node chooses its own and can swap it.

- The default update rule is PC-ALM (Augmented Lagrangian Predictive Coding, <https://pub.sakana.ai/pc-alm/>). It is reported to perform close to BP with local updates. The choice is provisional
- BP is also an update rule, as long as it stays inside one node (see "BP inside a node" below)

!!! note "The PC-ALM nodes are not implemented yet"

    The PC-ALM nodes are planned for the module `mujou.pc`, which is not implemented yet. Until then, subclass `mujou.Node` and write the update equations yourself in `on_update(dt)` (see [Writing a node](writing-a-node.md)).

## Example

The following node learns a linear map from its input to the target online, and sends its prediction. BP stays inside the one node.

```python
--8<-- "examples/torch_node.py:node"
```

In `online_learning.py` in [Examples](../examples/index.md), a node like this learns from a stream whose target changes partway through the run, and keeps up with the change.

## Device and seed

Pass the device and the random seed in `__init__`, with `super().__init__(device=..., seed=...)`.

- `device` is the CPU (the default, `"cpu"`) or a CUDA device. Mujou supports only these two. Any other device, such as `"mps"` or `"meta"`, raises `ValueError`, and so does a CUDA device when CUDA is not available
- Received messages (the values of `recv()`) arrive on the node's device
- Allocate tensors with `device=self.device`

## Random number generator

Draw random numbers from the node's own generator, `self.generator`. The generator is made from `seed` and lives on the node's device.

```python
nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
noise = torch.randn(4, generator=self.generator, device=self.device)
```

- `super().__init__()` creates `self.generator`, so you can use it in `__init__` already
- Do not call `torch.manual_seed` in a node. The default generator is shared by the whole process, so it would change the random numbers of other nodes too
- A restored run uses new node instances, whose generators start again from `seed`. To continue the same random sequence after a restore, put the generator's state in the state tree (see [Saving and restoring state](state.md))

## BP inside a node

Mujou turns gradient recording off before every hook of every node (`torch.set_grad_enabled(False)`). The default update rule, PC-ALM, computes its update equations directly, so it will need no autograd graph.

To use autograd in a node, write it as follows.

1. Wrap both the forward pass and the gradient computation (`torch.autograd.grad` or `.backward()`) in `with torch.enable_grad():`
2. Mark that `with` line with `# mujou: local-autograd`. The mark says that BP stays inside this node
3. Update the parameters in place, outside the block (`p -= h * self.rate * g` in the example). No gradient is recorded outside the block, so the update does not enter the autograd graph

BP stays inside the node. A node cannot pass gradients to another node: `recv()` returns a copy, and `send()` detaches the value with `detach()` before sending it.

- Do not use `torch.inference_mode()`. Tensors created in it cannot be used for backward
- Gradients accumulated in `.grad` by `.backward()`, and the state of an optimizer, are not put in the state tree automatically. To continue the same way after a restore, override `state_dict` to add them (see [Saving and restoring state](state.md))
- Make the update size proportional to `dt`. The example splits `dt` into `n` steps `h = dt / n`, each no longer than `MAX_STEP`. It computes the gradient again for each step and updates `n` times. Explicit Euler then does not diverge when `dt` is long, and no elapsed time is lost

## CUDA

For a node given a CUDA `device`, Mujou sets up the following.

- It makes the node's device the current device in the node's hooks
- It creates a CUDA stream for the node alone and makes it the current stream in the node's hooks. All of the node's operations go on this stream
- It synchronizes the node's stream after each `on_update`, and around taking and restoring the state. The worker waits meanwhile, as in a hook that blocks (see [Running a network](running.md#threads-of-a-run))

Mujou sets the device and the stream whenever a worker switches to a CUDA node. A node on the CPU does not set them, so on a worker that ran a CUDA node just before, the CUDA stream of that node is still current. A node on the CPU does not use CUDA.

So the node's author does not have to write any synchronization. If you synchronize yourself, synchronize only your own stream (`torch.cuda.current_stream().synchronize()`).

!!! warning "Do not wait for the whole device"

    Do not use `torch.cuda.synchronize()`, which waits for the whole device, including the work of other nodes.

Without an index, as in `device="cuda"`, the device is the host's current device at the time the node is created.

In the tree `run.state_dict()` returns, the state of a CUDA node stays in CUDA tensors. Move every tensor of the tree to the CPU with `cpu()` before saving it, and a machine without a GPU can read it too. On a restore, the default `load_state_dict` copies the tensors back to the node's device (see [Saving and restoring state](state.md#formats)).

```python
torch.save(run.state_dict().cpu(), "state.pt")
```

## Settings of the whole process

The following settings are shared by the whole process. Do not change them in a node. If needed, the host sets them once, before `mujou.run`.

- `torch.manual_seed` and the default random number generator
- `torch.set_num_threads`, `torch.use_deterministic_algorithms`, and the flags in `torch.backends.*`
- Operations on a whole device, such as `torch.cuda.synchronize()` and `torch.cuda.empty_cache()`

A node may change only per-thread settings, and only inside a hook: the next hook of the node may run on another thread, and the thread runs other nodes in between (see [Run on a thread of its own](writing-a-node.md#run-on-a-thread-of-its-own)). Set such a setting back before the hook returns, by keeping it inside a `with` in the hook (`with torch.autocast(...):`); a setting left behind leaks into the next node on the same worker. Mujou itself sets back only gradient recording, before every hook, and on CUDA the current device and stream, whenever a worker switches to the node. A node with `dedicated_thread=True` runs every hook on one thread, and can make its per-thread settings once, in `on_thread_start`, a hook called on that thread just before `on_start`.

### Internal threads

The nodes compute on the run's worker threads (and the threads of the nodes with `dedicated_thread=True`). On the CPU, torch operations use further internal threads for each of those threads. With too many internal threads, the nodes compete for the CPU and everything slows down.

Mujou does not set the number of internal threads, so the host decides it before `mujou.run`.

```python
--8<-- "examples/torch_node.py:threads"
```

!!! tip "Rule of thumb for the number of threads"

    The number of workers (plus the nodes with `dedicated_thread=True`) times the `n` of `torch.set_num_threads(n)` should not exceed the number of physical cores. With many nodes on small matrices, `n = 1` is enough.
