# 🧠 PC-ALM

The module `mujou.pc` learns a PyTorch model split into nodes with PC-ALM (Augmented Lagrangian Predictive Coding, <https://pub.sakana.ai/pc-alm/>). [5. Learn with PC-ALM](../tutorial/5-learn-with-pc.md) builds a chain with `pc.Chain` and shows it batches with `pc.show`. This page covers how to choose the time constants, the optimizer and the relaxer, how it differs from BP, the pitfalls, MNIST and CIFAR-10, and nodes of your own.

## How it differs from BP

| Topic                     | BP in plain PyTorch                                  | `mujou.pc`                                                                                                                 |
| ------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Inference and learning    | Forward, backward and `optimizer.step()`, one by one | At the same time. Every node updates on its own period, and learns while the input is shown                                |
| One training step         | One batch                                            | The host shows a batch for a while (`S` periods). The nodes relax towards the target, then update their weights            |
| The target                | Passed to the loss with the batch                    | Written with the input, and carried in the activity from node to node until the output node                                |
| Training and test         | `model.train()` / `model.eval()`                     | No modes. A sample whose `mask` is False is not learned from, so the test batches go through the same running nodes        |
| The loss                  | Usually the batch mean                               | The loss of each sample, of the shape of the batch. The output node sums it                                                |
| Where gradients are taken | `.backward()` through the whole model                | Inside each node only. A node exchanges activities and energy gradients with its neighbors as messages                     |
| Cost                      | One forward and backward per batch                   | One forward and VJP per node per period, so `S` of them per batch. On the CPU, MNIST takes about 40 times longer per batch |

## The nodes

| Node        | What it holds                                          | Ports                                                                                                                                                                           |
| ----------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pc.Input`  | Nothing                                                | `prev`: the host writes a `pc.msg.Sample` (`x`, `y`, `mask`, `reset`). `next`: sends `x` as the `h` of a `pc.msg.Activity`, with `y`, `mask`, `reset` and the time of the write |
| `pc.Layer`  | A module, its deviation `r`, the dual `lam`, its clock | `prev` and `next`. `next` sends its prediction `mu` as the `h` of a `pc.msg.Activity`, with the `y`, `mask`, `reset` and `time` that came with its input                        |
| `pc.Output` | The last module, the loss, its clock                   | `prev`. `next` sends a `pc.msg.Prediction` (`y_hat`, `y`, `mask`, `time`) for the host to read                                                                                  |

A `pc.msg.Activity` carries `h`, `y`, `mask`, `reset` and `time`. A `pc.Layer` holds the deviation `r = h - mu` of its activity from its prediction `mu = module(h_in)` and the dual `lam`, and sends `mu`: what flows forward is the forward pass, and the credit flows back only. Whatever moves `mu` (a new sample, the learning of the nodes below, its own learning), `r` and `lam` stay and relax, unless the sample starts anew ([Starting anew](#starting-anew-reset)).

`pc.Chain(*modules, ...)` connects `Input` → `Layer` → ... → `Output`, with every period `tau.period` (`tau.h / 2` with `tau.r = tau.h`). For other periods or other shapes, build the nodes one by one: give `pc.Input` the specs `x` and `y` and a `period`, give each `pc.Layer` and `pc.Output` the `act_spec` of the node before it (the spec of the `pc.msg.Activity` it receives), and connect `next` to `prev` along the chain. `pc.show` shows samples to such a network too, when it inherits `pc.Shown` as well as `mujou.Network` and exposes the `prev` of its `Input` and the `next` of its `Output` as its own `prev` and `next`.

Put in a node any `nn.Module` that treats each sample of the batch on its own: one layer, a stack of layers, or a residual block. `Dropout` and `BatchNorm` in training mode mix or drop samples, so do not use them; `LayerNorm` is fine. Initialize the weights when you make the modules: the nodes do not. A parameter with `requires_grad=False` does not learn.

## Starting anew: `reset`

`r` and `lam` hold the credit of the sample a node has seen. The host says with `reset` (a bool for each sample, the shape of `mask`) whether each sample it writes starts anew:

| Data                                           | `reset`                                          |
| ---------------------------------------------- | ------------------------------------------------ |
| Shuffled batches, independent samples          | True for every sample                            |
| A stream, each frame continuing the one before | False, and True at the first frame of an episode |
| The test batches                               | True, as for training                            |

A node acts on it once, in the update in which it first sees the write: it sets `r` and `lam` of those samples to 0, so the credit of an unrelated sample is not left over, and the credit of the new one builds up from zero as it comes back from the output. A sample written with `reset` False keeps them, so the credit carries on from frame to frame. A write that a node never sees (the host wrote again before the node updated) does not reset it.

```python
new = torch.ones_like(mask)  # shuffled: every sample starts anew
p = pc.show(run, net, pc.msg.Sample(x=x, y=y, mask=mask, reset=new), seconds=shown)
```

## Choosing the time constants

Every rate in `mujou.pc` is a time constant in seconds of the run's timer: the shorter it is, the faster the value moves. Choose them in this order.

| Value            | What it sets                                    | How to choose it                                                                                                                                                                | MNIST                    |
| ---------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `tau.h`          | How fast each deviation `r` relaxes             | First. The others follow from it                                                                                                                                                | 4 ms                     |
| `tau.r`          | How fast the deviation `r = h - mu` decays      | `tau.h`. Required, with no default. `math.inf` turns the decay off                                                                                                              | 4 ms                     |
| `tau.lam`        | How fast the dual `lam` follows                 | `tau.h / 2`                                                                                                                                                                     | 2 ms                     |
| The period       | How often each node updates                     | `tau.period` (`tau.h / 2` with `tau.r = tau.h`), the default of `Layer`, `Output` and `pc.Chain` (`period=`). With the two above, the credit moves one node back in each period | 2 ms                     |
| `relaxer`        | How `r` moves along the force on it             | The default `pc.relax.Euler()`. See [Choosing the relaxer](#choosing-the-relaxer)                                                                                               |                          |
| The showing time | How long the host shows each batch              | `S` periods, with `S` a few periods more than the round trip of the credit, `2 * hops`                                                                                          | 16 periods, 32 ms        |
| `learn_interval` | When the weights update, counted from the write | Longer than half of the showing time, shorter than the showing time, and longer than the round trip of the credit. MNIST uses 3/4                                               | 24 ms                    |
| `optimizer`      | How fast the weights learn                      | `pc.optim.SGD(tau=showing time / lr)`, with `lr` the learning rate per batch. See [Choosing the optimizer](#choosing-the-optimizer)                                             | `pc.optim.SGD(tau=0.32)` |
| `max_dt`         | The longest `dt` one update counts              | The default, `tau.max_dt`: 1 period here. See the period and `max_dt` below                                                                                                     | 2 ms                     |

- **The period and `max_dt`.** The default `tau.period` is `tau.h / (1 + tau.h / tau.r)`, and the default `tau.max_dt` is `tau.period` or `2 * tau.lam`, whichever is shorter. A `max_dt` shorter than the period raises `ValueError` when the node is made. With `tau.lam >= tau.period / 2`, the default `max_dt` is the default period
- **The learning rate.** With `pc.optim.SGD`, a weight update moves the weights by `elapsed / tau` times the mean gradient of the batch, with `elapsed` the time since the node last learned. With one update per batch, `elapsed` is the showing time, so `pc.optim.SGD(tau=showing time / lr)` gives the learning rate `lr` per batch. `elapsed` is the sum of the `dt` of the updates in between, each capped at `max_dt`; updates on time have a `dt` of one period, so it is counted in full
- **The mean gradient.** The sum of the per-sample gradients divided by `B`, the number of samples in the batch (1 without a batch). A sample whose `mask` is False adds nothing to the sum but still counts in `B`
- **Why `learn_interval` is between half and all of the showing time.** It is counted from the time the host wrote the input (the `time` of the activity), not from when a node received it, so the nodes learn at about the same time however deep they are. A node learns `learn_interval` after each write, and again every `learn_interval` while the batch is still shown: shorter than half, it learns twice or more per batch. If it is as long as the showing time or longer, the next batch is written first, and the node never learns
- **The round trip of the credit.** A node never learns in the update in which it first sees a batch, and the credit needs time to go down the path and back: about `2 * hops * period`, with `hops` the number of connections on the path (3 connections and 6 periods for MNIST, from the `Input` to the `Output`). A `learn_interval` longer than that is recommended: before the credit has come back, the nodes near the input learn from too little of it. Each period of every node costs a forward and a VJP, so fewer, larger nodes need fewer periods for a batch
- **A stream whose input changes at every frame.** Pass `learn_interval=None`: the node then learns in every update, whether or not the input changed, and `1 / tau` of `pc.optim.SGD` is the learning rate per second. Good values for streams have not been measured yet
- **The decay of `r`.** `tau.r` adds `-(tau.h / tau.r) * r` to `tau.h * dr/dt`. `lam` integrates `r`, so `r = 0` at the fixed point whatever `tau.r` is, and `lam` is still the credit of BP there. It pulls `r` `1 + tau.h / tau.r` times as hard, so the default period shrinks by as much
- **Why these values.** With `tau.r = tau.h`, `tau.lam = tau.h / 2` and the period `tau.h / 2`, one update moves `delta = r + lam` to `-g` exactly: the credit moves one node back in each period, and does not overshoot. The nodes send `mu`, so the credit flows back only, and the steps are as stable in a deep chain as in one node, whatever the Jacobians of the modules. Longer steps (`dt = tau.h = tau.lam`, without the decay) overshoot `delta` by 2 times, and the overshoot grows at every node it passes; shorter steps are stable but need more periods

## Choosing the optimizer

`pc.Layer`, `pc.Output` and `pc.Chain` require `optimizer=`. It sets how the weights move with the mean gradient `g` of the batch when a node learns. The optimizer is a setting without state, so `pc.Chain` gives the same one to every node. Each node binds it to its own weights and holds the state of its weights itself.

| Optimizer                                                                    | One learning step of `elapsed` seconds                                                                                  | State                  |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `pc.optim.SGD(tau=...)`                                                      | `p -= (elapsed / tau) * g`                                                                                              | None                   |
| `pc.optim.AdamW(tau=..., tau_m=..., tau_v=..., eps=1e-8, weight_decay=1e-2)` | AdamW with the learning rate `elapsed / tau`. The moments `m` and `v` decay with the time constants `tau_m` and `tau_v` | `m`, `v`, `c_m`, `c_v` |

- **Time, not steps.** Every `tau` is in seconds. `elapsed` is the time since the node last learned, so the `elapsed` of the steps in one batch add up to the showing time, however many steps there are. With `pc.optim.SGD`, the weights thus move as far per batch whether the nodes learn once or twice per batch
- **SGD** is what MNIST uses. Choose `tau` as in the table of the time constants: the showing time divided by the learning rate per batch
- **AdamW** moves each weight by about `elapsed / tau` per step, whatever the size of its gradient. Good values of AdamW for `mujou.pc` have not been measured yet
- **From `torch.optim.AdamW`.** Let `Δt` be the time between two learning steps: the showing time when the nodes learn once per batch. To reuse the values `lr` and `betas` of `torch.optim.AdamW`, set `tau = Δt / lr`, `tau_m = -Δt / ln(beta1)` and `tau_v = -Δt / ln(beta2)`. With a constant `Δt`, the steps are then those of `torch.optim.AdamW`. For one step every 32 ms, `beta1 = 0.9` is `tau_m` ≈ 0.30 s and `beta2 = 0.999` is `tau_v` ≈ 32 s
- **`weight_decay`** has no unit, as in `torch.optim.AdamW`: it decays the weights with the time constant `tau / weight_decay`
- **Invalid values.** Every `tau` and `eps` must be finite and > 0, and `weight_decay` finite and >= 0; otherwise the optimizer raises `ValueError`. For a node that never learns, give that `pc.Layer` or `pc.Output` `learn_interval=math.inf`, not an infinite `tau` (`pc.Chain` gives one `learn_interval` to every node)
- **Only the weights that learn.** The nodes bind the parameters with `requires_grad=True` only. A frozen parameter gets no weight decay either
- **The state** of a node's optimizer is in `run.state_dict()["nodes"][path]["state"]["optimizer"]`: a dict with the keys `"m"`, `"v"`, `"c_m"` and `"c_v"` for `pc.optim.AdamW`, empty for `pc.optim.SGD`. It is saved and restored with the rest of the run

```python
# lr=1e-3, betas=(0.9, 0.999) of torch, rounded, for one step per batch of 32 ms.
optimizer = pc.optim.AdamW(tau=0.032 / 1e-3, tau_m=0.3, tau_v=32.0)
```

??? question "Why not `torch.optim`?"

    Its optimizers count steps, not time, so the time constant that `betas` gives depends on how often a node learns. Making one also brings the GIL back in a free-threaded Python.

## Choosing the relaxer

`pc.Layer` and `pc.Chain` take `relaxer=`. It sets how the deviation `r` moves along the force `F = (1 + tau.h / tau.r) * r + lam + g` on it, as the optimizer does for the weights. It is optional, and `pc.relax.Euler()` is the default. A relaxer is a setting without state, so `pc.Chain` gives the same one to every `pc.Layer`. Each binds it to its own `r` and holds the state of its `r` itself. `pc.Output` has no `r`, so it takes none.

In one step of `dt` seconds, `pc.relax.Euler()` moves `r -= (dt / tau.h) * F`, and has no state.

- **A relaxer changes how `r` gets to the fixed point, not where it is.** At the fixed point `F = 0`, and `lam` is the credit of BP. `lam` moves after `r`, whichever relaxer moves `r`
- **The state** of a node's relaxer is in `run.state_dict()["nodes"][path]["state"]["relaxer"]`: empty for `pc.relax.Euler`, and whatever a relaxer of your own holds. It is saved and restored with the rest of the run. With each new input, the node calls `reset_(reset)`: a relaxer that remembers forces forgets them for the samples that start anew
- **A relaxer of your own** inherits `pc.relax.Relaxer` and `pc.relax.BoundRelaxer`, as an optimizer of your own does ([How mujou.pc Works](pc-concepts.md#7-relaxers)). Its fixed point must be where `F = 0`: `lam` is the credit only there

## MNIST

`examples/mnist/` learns MNIST with a 784-256-256-10 ReLU network in a `pc.Chain`, and trains the same network, from the same initial weights on the same batches, by BP. After 3 epochs, PC-ALM reached a test accuracy of 97.56% and BP 97.54% (seed 0), and the prequential accuracy was within 0.03 points of BP in every epoch.

```sh
uv run python examples/mnist/main.py            # 3 epochs, about 2 minutes
uv run python examples/mnist/main.py --epochs 1 --watch
```

```python
--8<-- "examples/mnist/main.py:chain"
```

```python
--8<-- "examples/mnist/main.py:run"
```

- **Prequential accuracy** (during training): the first prediction for each batch, which `pc.show` returns, made before any node has learned from it. It measures how well the model predicts data it has not seen yet
- **Test accuracy**: the host shows the test batches with every `mask` False, so no node learns from them, and keeps the original `mask` to score with. There is no inference mode to switch to
- **The last batch.** `data.batches` pads the last batch with zeros whose `mask` is False, so that every batch has the size of the specs. Count only the samples whose original `mask` is True
- **`--watch`** counts the `dt` of each node (`watch.py`), and warns when the updates come later than 1.5 periods (median) or 2 periods (99%). At `--time-scale 2`, 99% of the updates still came within 1.25 periods, with the same accuracy
- `data.py`, `metrics.py` and `bp.py` do not depend on `mujou.pc`: the data are plain tensors, and the BP baseline is plain PyTorch

## CIFAR-10

`examples/cifar10/` learns CIFAR-10 with a ResNet-20 on a CUDA device, with a node for each stage, and trains the same network by BP on the GPU, from the same initial weights on the same batches with the same `pc.optim.AdamW`. The modules become the nodes after the `Input` of a `pc.Chain`: the stem with the first stage, the second stage, the third stage, and the head, five nodes in all. With batches of 32 shown for 12 periods each, in the 3rd epoch the prequential accuracy was 57.11% for PC-ALM and 55.91% for BP (seed 0), and the test accuracy after it 59.47% and 61.76%; in the 10th, 73.05% and 73.20%, and 73.90% and 74.35%. An epoch took 48 s against 8 to 9 s for BP. The test accuracy moves by a few points from epoch to epoch for both, as the learning rate does not decay, and the GPU does not compute bit for bit the same in every run.

```sh
uv run python examples/cifar10/main.py            # 3 epochs, about 4 minutes on an RTX 4090
uv run python examples/cifar10/main.py --epochs 10  # about 13 minutes
```

```python
--8<-- "examples/cifar10/main.py:chain"
```

- **A node for each stage, with BP inside each.** Inside a node, autograd differentiates its module (the blocks, their skip connections and activations included); between the nodes, only the messages of PC-ALM flow
- **12 periods for each batch**, and the nodes learn at 0.9 of it. The credit moves one node back in each period, and the round trip from the `Input` to the `Output` and back is 8 periods
- **Fewer nodes, fewer periods.** On one GPU, a period of all the nodes costs about the same however the network is cut: launching the kernels from the CPU is the limit, and an update takes about 1.7 ms. With a node for each residual block (11 nodes), a batch needed 24 periods, and the nodes could not keep up with a period of 2.5 ms of real time; with batches of 128 and a period of 5 ms, they reached 53.94% (prequential) and 57.67% (test) in the 3rd epoch, at 17 times the time of BP
- **On the GPU, twice the real time.** `device="cuda"` makes every node compute on the GPU, and the messages between them stay there ([Devices](threads-and-devices.md#devices)). `tau.h` is 10 ms, so the period is 5 ms of the timer, and the timer goes twice as fast as real time (`--time-scale` 2): 2.5 ms of real time. The updates came within 1.00 periods on average
- **A batch of 32.** An update costs about the same with 32 samples as with 128, while BP gains more from a larger batch
- **AdamW**, made from the `lr` of Adam as in [Choosing the optimizer](#choosing-the-optimizer). The BP baseline binds the same `pc.optim.AdamW` to its whole model and steps it by the showing time once per batch, so both take about the same steps for the same gradients (the nodes learn for the time since they last learned, which is close to the showing time)
- **GroupNorm** instead of BatchNorm: it has no running statistics and no mode to switch, and the samples of a batch, padded ones included, do not affect each other

## The batch

The batch is the shape that every tensor of the messages starts with. Give it as `batch` to `pc.Chain`, `pc.Input` or `pc.msg.activity_spec`, and start the specs `x` and `y` with it. `pc.Layer` and `pc.Output` take it from the `act_spec` they receive.

| `batch`        | `x` (784 inputs) | `y` (one class) | `mask`   | For                                             |
| -------------- | ---------------- | --------------- | -------- | ----------------------------------------------- |
| `()` (default) | `(784,)`         | `()`            | `()`     | One sample at a time, as from a sensor          |
| `64`           | `(64, 784)`      | `(64,)`         | `(64,)`  | 64 samples                                      |
| `(8, 4)`       | `(8, 4, 784)`    | `(8, 4)`        | `(8, 4)` | 4 agents in each of 8 environments, for example |

- The samples must be independent: the module must compute each one on its own, along every dimension of the batch. A dimension that the module mixes, such as the time of a sequence, is a dimension of the sample, not of the batch: 64 sequences of `T` steps are `batch=64` with `x` of `(64, T, 784)`
- A batch of several dimensions learns as the same samples in one dimension: `(8, 4)` as `32`
- The losses take the batch too: `pc.loss.MSELoss(batch)` and `pc.loss.CrossEntropyLoss(batch)`. For `pc.loss.CrossEntropyLoss`, `y` holds the classes, int64 of the shape of `y_hat` without its class dimension, or the probabilities of the classes, of the shape of `y_hat`. The classes are the dimension of `y_hat` right after the batch, as in `nn.CrossEntropyLoss`, and the dimensions after them, such as the pixels of an image, are averaged

## Pitfalls

!!! warning "Give the batch to the chain and to the loss"

    The default `batch` is `()`, one sample, and the batch is never read from the shapes of the specs. With specs of 64 samples and no `batch` anywhere, the chain takes the 64 samples as one sample and `pc.loss.MSELoss()` averages over them: it still runs, but the credit of each sample is 1/64 of what it should be. Pass the same `batch` to `pc.Chain` (or `pc.Input`) and to the loss. If only one of them has it, `pc.Output` raises an error.

!!! warning "The loss returns the loss of each sample"

    Pass `pc.loss.MSELoss(batch)`, `pc.loss.CrossEntropyLoss(batch)`, or your own function `(y_hat, y)` that returns the shape of the batch. A loss that averages over the batch, such as `nn.MSELoss()`, makes `pc.Output` raise `ValueError`. `pc.loss.MSELoss` averages over every dimension of each sample (all but the batch), as `nn.MSELoss` does, with no `0.5`: the weights follow the gradient of `nn.MSELoss()` on the batch.

!!! warning "Read predictions from the output node"

    Predictions come only from the `next` of `pc.Output` (`net.next` of a `pc.Chain`). A `Layer`'s `next` sends its own prediction `mu` as `h` with the `y`, `mask`, `reset` and `time` it passes on, but not the prediction of the network. To watch `lam` of a layer, read it from `run.state_dict()["nodes"]["layer0"]["state"]["lam"]`; reading the node's attributes while it runs races with its updates.

!!! warning "The activity is zeros until the first sample arrives"

    Before the host's first write reaches a node, the activity it receives has `time` 0 and `mask` all False, so `run.read(net.next)` returns a prediction with `mask` all False, and the nodes do not learn. To get the prediction of a batch you wrote, match its `time` with the time `write` returned, as `pc.show` does.

!!! warning "Heavy load can make the network diverge"

    The nodes exchange messages while each one updates. If a node runs late, its neighbors keep learning from its old messages, and the network can diverge or lose accuracy. MNIST stops when `y_hat` is not finite; a run that diverges with finite predictions shows only as a falling accuracy. Two causes are known:

    - **A time scale that is too fast.** The examples speed up the run's timer with `--time-scale`. Too fast, and the slowest node takes longer than its period to compute: in CIFAR-10 with a node for each block, at a period of 2 ms of real time, the updates came 1.1 to 1.3 periods apart on average with stalls of up to 100 periods, and the accuracy fell by about 20 points. Run one epoch with `--watch` (MNIST) and take the highest time scale that gives no warning and does not diverge
    - **Other processes on the cores.** A node that waits for a core lets its neighbors run ahead. On 32 cores at time scale 1, one epoch ran through beside 26 busy processes and diverged beside 32; at 0.5 it ran through beside 32. Run on an otherwise idle machine, or lower the time scale

    `max_dt` does not prevent this. It caps the time each single update counts, and the time beyond it is dropped: from the relaxation and from the next weight update alike.

!!! warning "Set `reset` for shuffled samples"

    Written with `reset` False, a sample keeps the `r` and `lam` of the sample before it in each node until its own credit comes back from the output. With shuffled batches, that is the credit of an unrelated sample: until then the nodes near the input hold the wrong credit, and learn from it if they learn early. Set `reset` True for every sample that does not continue the one before.

## Nodes of your own

`mujou.pc.functional` holds the computations the nodes are made of: `vjp` and `loss_vjp` (the gradients of a node's energy) and `relax_` (one step of `r` and `lam`). The optimizer moves the weights, and the relaxer moves `r`. Use them to write a node that `pc.Layer` does not cover, such as one with several inputs. The names here (`mu`, `r`, `lam`, `g_in`, `Tick`) are explained in [How mujou.pc Works](pc-concepts.md).

1. In `__init__`, after the module is on its device, hold the clock, the bound optimizer, the deviation `r` and the dual `lam` (zeros of the shape of `mu`), and the bound relaxer if `r` should move as a relaxer of your own says, as attributes. All are then in the node's state by themselves
    - `self.clock = pc.clock.LearnClock.of(tau, period=self.period, learn_interval=...)`, which takes the default `max_dt` of `tau` as `pc.Layer` does
    - `self.optimizer = optimizer.bind(params)`, with `params` the parameters that learn. Bind in each node: two nodes must not hold the same bound optimizer
    - `self.relaxer = relaxer.bind(self.r)`, in the same way. Without it, `relax_` takes the plain step
2. In each update, `t = self.clock.tick(dt, act["time"], self.timer.now())`. When `t.new` is True, set `r` and `lam` of the samples whose `reset` is set to 0 (`self.r.masked_fill_(rows, 0.0)`, with `rows` the `reset` shaped to `r`), and call `self.relaxer.reset_(act["reset"])` if the node has a relaxer
3. Find the prediction `mu`, `g_in` and, to learn, `g_theta` with `pc.functional.vjp`, with `delta = r + lam` (zero for the samples whose `mask` is False) and the bound `params` in the same order, and send `g_in` to the node before
4. Move `r` and `lam` with `pc.functional.relax_(self.r, self.lam, g, t.dt, tau, self.relaxer)`, with `g` the sum of what the next nodes sent
5. Send `mu` as the `h` of a `pc.msg.Activity`, with the `y`, `mask`, `reset` and `time` of the activity you received
6. When `t.learn` is True and the `mask` of some sample is True, learn with `self.optimizer.step_([g / B for g in g_theta], t.elapsed)`, with `B` the number of samples in the batch. The node divides by `B`: the optimizer does not know the batch

Hold `r`, not `h`, and send `mu`, not `mu + r`, as `pc.Layer` does: `r` and `lam` then carry over whatever moves `mu`, and the credit flows back only, which keeps the steps stable in any chain ([How mujou.pc Works](pc-concepts.md#why-the-node-holds-r-and-sends-mu)).

To connect with the nodes of `mujou.pc`, declare the ports with the `act_spec` of the node before, or with `pc.msg.activity_spec(h, y, batch)`. The arguments are in the [API reference](../api/pc.md). A rule other than PC that learns across nodes is in [Update Rules](update-rules.md).

### Example: a node with two inputs

`Merge` in `examples/pc_merge.py` predicts from the activities of two towers (`pc.Layer`) side by side. It learns in every update (`learn_interval=None`).

```python
--8<-- "examples/pc_merge.py:merge"
```

It differs from `pc.Layer` in one thing only: it has a port for each input and sends each its own part of `g_in`.
