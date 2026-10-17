# PyTorch in Nodes

A node computes with PyTorch inside. This page covers update rules, devices and random numbers, backpropagation (BP) inside a node, CUDA and other accelerators, and the settings that affect the whole process.

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

## Device and seed

Pass the device and the random seed in `__init__`, with `super().__init__(device=..., seed=...)`.

- `device` is the CPU (the default, `"cpu"`) or a device of PyTorch's current accelerator: CUDA, MPS, XPU, and so on. A device that is not available, and any other device (such as `"meta"`), raise `ValueError`
- Received messages (the values of `recv()`) arrive on the node's device
- Allocate tensors with `device=self.device`

## Random number generator

Draw random numbers from the node's own generator, `self.generator`. The generator is made from `seed` and lives on the node's device.

```python
nn.init.normal_(self.model.weight, std=0.1, generator=self.generator)
noise = torch.randn(4, generator=self.generator, device=self.device)
```

- `self.generator` is created on the node's thread, so you can use it from `on_start` on. Using it in `__init__` raises `RuntimeError`
- Do not call `torch.manual_seed` in a node. The default generator is shared by the whole process, so it would change the random numbers of other nodes too
- The generator is created again from `seed` every time the run starts. To continue the same random sequence after a restore, put the generator's state in the state tree (see [Saving and restoring state](state.md))

## BP inside a node

Mujou turns gradient recording off on every node's thread (`torch.set_grad_enabled(False)`). The default update rule, PC-ALM, computes its update equations directly, so it will need no autograd graph.

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

- It makes the node's device the current device of the node's thread
- It creates a CUDA stream for the node alone and makes it the current stream of that thread. All of the node's operations go on this stream
- It synchronizes the node's stream after each `on_update`, and around taking and restoring the state

So the node's author does not have to write any synchronization. If you synchronize yourself, synchronize only your own stream (`torch.cuda.current_stream().synchronize()`). Do not use `torch.cuda.synchronize()`, which waits for the whole device, including the work of other nodes.

Without an index, as in `device="cuda"`, the device is the host's current device at the time the node is created.

??? tip "A CUDA error names `_after_update`"

    A CUDA error raised in `on_update` (a device-side assert, for example) surfaces at the synchronization right after it. So the message of `NetworkError` names `_after_update`, an internal hook of Mujou. The cause is in `on_update`.

## Other accelerators (MPS, ...)

A node on another accelerator, such as `device="mps"`, gets the same setup as on CUDA, except for the stream: the node computes on the device's current stream, with no stream of its own. Mujou synchronizes the whole device where it would synchronize the node's stream on CUDA.

- Without an index, as in `device="mps"`, the device is the host's current device (`mps:0`)
- PyTorch's MPS cannot be used from two threads at once: the process crashes. So only one node of all the running runs may compute on MPS. `mujou.run` (and `mujou.testing.run`) raises `ValueError` before any node starts when a second one would; put the others on the CPU, or compute them in one node. Do not compute on MPS on the host either while a run with a node on MPS is running
- MPS has no float64. A node on MPS cannot receive a message of a float64 spec

## Settings of the whole process

The following settings are shared by the whole process. Do not change them in a node. If needed, the host sets them once, before `mujou.run`.

- `torch.manual_seed` and the default random number generator
- `torch.set_num_threads`, `torch.use_deterministic_algorithms`, and the flags in `torch.backends.*`
- Operations on a whole device, such as `torch.cuda.synchronize()` and `torch.cuda.empty_cache()`

A node may change only per-thread settings. Write them in `on_thread_start`, a hook called once on the node's thread just before `on_start`.

### Internal threads

Each node computes on its own thread. On the CPU, torch operations use further internal threads for each of those threads. With too many internal threads, the nodes compete for the CPU and everything slows down. Mujou does not set the number of internal threads, so the host decides it before `mujou.run`.

```python
--8<-- "examples/torch_node.py:threads"
```

As a rule of thumb, the number of nodes that compute on the CPU times the `n` of `torch.set_num_threads(n)` should not exceed the number of physical cores. With many nodes on small matrices, `n = 1` is enough.
