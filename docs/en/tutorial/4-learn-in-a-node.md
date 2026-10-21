# 🔥 4. Learn in a Node

This page writes a node that learns with backpropagation (BP) inside itself, from a stream whose target changes while it runs. There is no training phase and no inference phase: the node predicts and learns from each new batch as it comes.

## The program

A teacher node sends a batch from $y = a x + b$ every 10 ms, with the `a` the host writes; the host changes `a` while the network runs. The learner follows its loss: it falls, jumps when the target changes, and falls again.

```sh
uv run python examples/online_learning.py
```

```text
t = 0.48 s: loss = 0.000090  (learned a = 2.0)
t = 0.49 s: loss = 3.405589  (a changed at 0.48 s)
t = 0.97 s: loss = 0.000095  (learned a = -1.0)
the learner's weight = -0.988, bias = 0.498
```

## The learner

```python
--8<-- "examples/online_learning.py:node"
```

- **The module is state.** `self.model` has `state_dict` and `load_state_dict`, so it is in the node's state by itself, as a tensor attribute is
- **The seed.** `seed=0` seeds `self.generator`, the node's own random generator. The default initialization of `nn.Linear` draws from the global generator that every node shares, so initialize the weights again from `self.generator`
- **Plain values join the state with `register_state`.** `last` and `elapsed` change while the node runs, but they are neither tensors nor modules: registered, they are saved and restored with the rest
- **New or not.** `recv_with_time()` returns the batch with the time it was sent. Its time is `None` until the teacher first sends, and it stays the same until the teacher sends again, so the learner learns once from each batch
- **Gradients as in plain PyTorch.** The hooks record gradients as plain PyTorch does: `loss.backward()` fills `p.grad`, and the parameters are updated in place inside `torch.no_grad()`. Values computed only to send (the loss before learning) need no graph either
- **By the time, not by the batch.** The learner follows the gradient flow $\frac{dW}{dt} = -\text{rate}\,\frac{\partial L}{\partial W}$ over the time since it last learned (`elapsed`), not a fixed step per batch: how far it learns depends on time, not on how often it runs. A long `elapsed` is split into steps no longer than `MAX_STEP`, so that explicit Euler does not diverge

BP stays inside the node. A port sends a copy, detached from the graph, so no gradient crosses nodes.

## Settings of torch

```python
--8<-- "examples/online_learning.py:threads"
```

The host sets the settings of the whole process, such as the number of torch's threads, before the run. A node changes only settings of its own thread, and sets them back before its hook returns (see [Threads and Devices](../topics/threads-and-devices.md)).

## The run

```python
--8<-- "examples/online_learning.py:run"
```

`run.wait_for` follows the loss port until the condition holds. After the run, the nodes' attributes are the host's to read: `learner.model` is the module the learner learned.

## Next

- A rule that learns across nodes, with a target that travels along the path: [Update Rules](../topics/update-rules.md)
- PC-ALM, the nodes of `mujou.pc`: [5. Learn with PC-ALM](5-learn-with-pc.md)
