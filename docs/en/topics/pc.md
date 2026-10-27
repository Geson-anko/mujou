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

| Node        | What it holds                                         | Ports                                                                                                                                                         |
| ----------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pc.Input`  | Its count of writes                                   | `prev`: the host writes a `pc.msg.Sample` (`x`, `y`, `mask`). `next`: sends `x` as the `h` of a `pc.msg.Activity`, with `y`, `mask` and the time of the write |
| `pc.Layer`  | A module, its activity `h`, the dual `lam`, its clock | `prev` and `next`. `next` sends `h` as a `pc.msg.Activity`, with its `jumps` and the `y`, `mask` and `time` that came with its input                          |
| `pc.Output` | The last module, the loss, its clock                  | `prev`. `next` sends a `pc.msg.Prediction` (`y_hat`, `y`, `mask`, `time`) for the host to read                                                                |

A `pc.msg.Activity` carries `h`, `jumps`, `y`, `mask` and `time`. `jumps` counts the times the sender's `h` jumped rather than relaxed: a `pc.Input` counts the host's writes, and a `pc.Layer` its moves with a new sample or with a jump of the `h` it receives, and the jumps that its own weights make when it learns at intervals. A receiver that sees `jumps` change moves its `h` with its prediction once, so that the jump does not enter its deviation `h - mu`.

`pc.Chain(*modules, ...)` connects `Input` → `Layer` → ... → `Output`, with every period `tau.period` (`tau.h / 4`). For other periods or other shapes, build the nodes one by one: give `pc.Input` the specs `x` and `y` and a `period`, give each `pc.Layer` and `pc.Output` the `act_spec` of the node before it (the spec of the `pc.msg.Activity` it receives), and connect `next` to `prev` along the chain. `pc.show` shows samples to such a network too, when it inherits `pc.Shown` as well as `mujou.Network` and exposes the `prev` of its `Input` and the `next` of its `Output` as its own `prev` and `next`.

Put in a node any `nn.Module` that treats each sample of the batch on its own: one layer, a stack of layers, or a residual block. `Dropout` and `BatchNorm` in training mode mix or drop samples, so do not use them; `LayerNorm` is fine. Initialize the weights when you make the modules: the nodes do not. A parameter with `requires_grad=False` does not learn.

## Choosing the time constants

Every rate in `mujou.pc` is a time in seconds of the run's timer. Choose them in this order.

| Value            | What it sets                                    | How to choose it                                                                                                                                                                                   | MNIST                    |
| ---------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `tau.h`          | How fast each activity `h` relaxes              | First. The others follow from it                                                                                                                                                                   | 4 ms                     |
| The period       | How often each node updates                     | `tau.period` (`tau.h / 4`), the default of `Layer`, `Output` and `pc.Chain` (`period=`). Shorter for stiff modules: see [the pitfalls](#pitfalls)                                                  | 1 ms                     |
| `tau.lam`        | How fast the dual `lam` follows                 | `2 * tau.h`                                                                                                                                                                                        | 8 ms                     |
| `tau.r`          | How fast the deviation `r = h - mu` decays      | The default `math.inf`, no decay. For a deep chain, `tau.h / 3`: see [CIFAR-10](#cifar-10)                                                                                                         |                          |
| `relaxer`        | How `h` moves along the force on it             | The default `pc.relax.Euler()`. See [Choosing the relaxer](#choosing-the-relaxer)                                                                                                                  |                          |
| The showing time | How long the host shows each batch              | `S` periods, with `S = 32`                                                                                                                                                                         | 32 ms                    |
| `learn_interval` | When the weights update, counted from the write | Longer than half of the showing time, shorter than the showing time, and longer than the round trip of the credit. MNIST uses 3/4                                                                  | 24 ms                    |
| `optimizer`      | How fast the weights learn                      | `pc.optim.SGD(tau=showing time / lr)`, with `lr` the learning rate per batch. See [Choosing the optimizer](#choosing-the-optimizer)                                                                | `pc.optim.SGD(tau=0.32)` |
| `max_dt`         | The longest `dt` one update counts              | The default, `tau.max_dt` (`min(tau.h / 2, 3 * tau.lam)` without `tau.r`, 2 periods here). A `max_dt` shorter than the period raises `ValueError`; so does the default when `tau.lam < tau.h / 12` | 2 ms                     |

- **The learning rate.** With `pc.optim.SGD`, a weight update moves the weights by `elapsed / tau` times the mean gradient of the batch, with `elapsed` the time since the node last learned. With one update per batch, `elapsed` is the showing time, so `pc.optim.SGD(tau=showing time / lr)` gives the learning rate `lr` per batch. `elapsed` is the sum of the `dt` of the updates in between, each capped at `max_dt`; updates on time have a `dt` of one period, so it is counted in full
- **The mean gradient.** The sum of the per-sample gradients divided by `B`, the number of samples in the batch (1 without a batch). A sample whose `mask` is False adds nothing to the sum but still counts in `B`
- **Why `learn_interval` is between half and all of the showing time.** It is counted from the time the host wrote the input (the `time` of the activity), not from when a node received it, so the nodes learn at about the same time however deep they are. A node learns `learn_interval` after each write, and again every `learn_interval` while the batch is still shown: shorter than half, it learns twice or more per batch. If it is as long as the showing time or longer, the next batch is written first, and the node never learns
- **The round trip of the credit.** A node never learns in the update in which it first sees a batch, and the credit needs time to go down the path and back: about `2 * hops * period`, with `hops` the number of connections on the path (3 connections and 6 periods for MNIST, from the `Input` to the `Output`). A `learn_interval` longer than that is recommended. Shorter, the weights learn from a credit that has not settled: in the measurements behind the design, a deep chain whose round trip was 18 periods converged with 16 periods, stayed bounded but off by about 4 times with 4, and diverged with 2 and 1
- **A stream whose input changes at every frame.** Pass `learn_interval=None`: the node then learns in every update, whether or not the input changed, and `1 / tau` of `pc.optim.SGD` is the learning rate per second. Good values for streams have not been measured yet
- **The decay of `r`.** `tau.r` adds `-(tau.h / tau.r) * r` to `tau.h * dh/dt`, a pull of `h` towards `mu`. `lam` integrates `r`, so `r = 0` at the fixed point whatever `tau.r` is, and `lam` is still the credit of BP there. In a deep chain it brings `lam` near the credit sooner (see [CIFAR-10](#cifar-10)); a too short `tau.r` makes `r` so small that `lam`, which integrates it, slows down (in that ResNet, `lam` settled less well with `tau.h / 10` than with `tau.h / 3`). It pulls `h` `1 + tau.h / tau.r` times as hard, so `tau.max_dt` shrinks by as much
- **Stability.** A shorter period relative to `tau.h` is more stable. In the measurements behind the design, `period / tau.h` up to 0.5 was stable for ReLU, and `period / tau.lam` between 0.1 and 0.3 worked best. `S = 32` gave the accuracy of BP; shorter showing times have not been verified

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

`pc.Layer` and `pc.Chain` take `relaxer=`. It sets how the activity `h` moves along the force `F = r + lam + g` on it, as the optimizer does for the weights. It is optional, and `pc.relax.Euler()` is the default. A relaxer is a setting without state, so `pc.Chain` gives the same one to every `pc.Layer`. Each binds it to its own `h` and holds the state of its `h` itself. `pc.Output` has no `h`, so it takes none.

| Relaxer                       | One step of `dt` seconds                                   | State      |
| ----------------------------- | ---------------------------------------------------------- | ---------- |
| `pc.relax.Euler()`            | `h -= (dt / tau.h) * F`                                    | None       |
| `pc.relax.HeavyBall(tau=...)` | `v = exp(-dt / tau) * v + F`, then `h -= (dt / tau.h) * v` | `velocity` |

- **A relaxer changes how `h` gets to the fixed point, not where it is.** At the fixed point `F = 0`, and `lam` is the credit of BP. `lam` moves after `h`, as before, whichever relaxer moves `h`
- **HeavyBall for a deep chain.** A velocity remembers the force of the updates before (a heavy ball). A force that lasts moves `h` faster, so the credit crosses a chain of many nodes in fewer periods. It speeds up the front of the credit, not the slow tail of the relaxation: with the ResNet of [CIFAR-10](#cifar-10) at the period `tau.h / 4`, 64 periods with it matched 128 without it in the 3rd epoch, but fell behind them by the 10th (62.77% against 68.73%). The example now relaxes with a shorter period, a fast dual and a decay of `r` instead, which did better than both; `pc.relax.HeavyBall` has not been measured with them. A shallow chain does not need it: MNIST uses the default
- **Choosing `tau`.** The force weighs one update, not `dt`, so the gain depends on the coefficient `exp(-dt / tau)` per update, and `tau` must follow the period. The tested value is `tau = tau.h / 4` with the period `tau.period` (coefficient 0.37, `tau` equal to the period): with 11 nodes of a ResNet, 64 periods did about what 128 did with `pc.relax.Euler()`, in half the real time. In the ResNet, a coefficient of 0.47 (`tau.h / 3`) collapsed in one run of three, and 0.8 diverged. The limit of about 0.78 is for a linear chain of identities, and the real limit is lower, lower still for late updates. For another period, keep the coefficient near 0.37
- **The velocity is set to zero with each new sample** and each jump of the `h` a node reads: `r` and `lam` are kept, but the velocity holds the forces of the input before. A jump of the node's own weights does not drop it
- **Invalid values.** `tau` of `pc.relax.HeavyBall` must be finite and > 0; otherwise it raises `ValueError`
- **The state** of a node's relaxer is in `run.state_dict()["nodes"][path]["state"]["relaxer"]`: a dict with the key `"velocity"` for `pc.relax.HeavyBall`, empty for `pc.relax.Euler`. It is saved and restored with the rest of the run
- **A relaxer of your own** inherits `pc.relax.Relaxer` and `pc.relax.BoundRelaxer`, as an optimizer of your own does ([How mujou.pc Works](pc-concepts.md#7-relaxers)). Its fixed point must be where `F = 0`: `lam` is the credit only there

## MNIST

`examples/mnist/` learns MNIST with a 784-256-256-10 ReLU network in a `pc.Chain`, and trains the same network, from the same initial weights on the same batches, by BP. After 3 epochs, PC-ALM reached a test accuracy of 97.60% and BP 97.54% (seed 0).

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
- **`--watch`** counts the `dt` of each node (`watch.py`), and warns when the updates come later than 1.5 periods (median) or 2 periods (99%)
- `data.py`, `metrics.py` and `bp.py` do not depend on `mujou.pc`: the data are plain tensors, and the BP baseline is plain PyTorch

## CIFAR-10

`examples/cifar10/` learns CIFAR-10 with a ResNet-20 on a CUDA device, with a node for each residual block, and trains the same network by BP on the GPU, from the same initial weights on the same batches with the same `pc.optim.AdamW`. The modules become the nodes after the `Input` of a `pc.Chain`: the stem with the first block, each of the other 8 blocks, and the head, eleven nodes in all. In the 3rd epoch, the prequential accuracy was 54.83% for PC-ALM and 54.45% for BP (seed 0), and the test accuracy after it 59.05% and 59.02%; in the 10th, the prequential accuracy was 73.35% and 73.27%, and the test accuracy 73.79% and 72.61%. Another run of the same settings reached 71.56% and 71.41% in the 10th epoch: PC-ALM comes within about 2 points of BP, or matches it, for about 750 s of real time per epoch against 2.8 s for BP. The test accuracy moves by a few points from epoch to epoch for both, as the learning rate does not decay, and the GPU does not compute bit for bit the same in every run.

```sh
uv run python examples/cifar10/main.py            # 3 epochs, about 40 minutes on an RTX 4090
uv run python examples/cifar10/main.py --epochs 10  # about 2 hours
```

```python
--8<-- "examples/cifar10/main.py:chain"
```

- **A node for each residual block, with BP inside each.** Inside a node, autograd differentiates its module (the skip connection and the activations of the block included); between the nodes, only the messages of PC-ALM flow
- **A period of `tau.h / 16`.** The blocks make the relaxation stiff, and with the period `tau.h / 4` of `pc.Chain` the explicit steps of `h` did not find the credit of BP ([the pitfalls](#pitfalls)). With `tau.h / 16` they found it as with `tau.h / 64`, so a shorter period would add updates and nothing else
- **A fast dual and a decay of `r`.** What remains is how long the credit takes to settle across 11 nodes. `tau.lam = tau.h / 4` lets `lam` follow `r` sooner, and `tau.r = tau.h / 3` damps the deviation. Both keep the fixed point. With the weights of 10 epochs, the cosine of `lam` with the credit in the first node after `24 * tau.h` was 0.84 with the fast dual alone and 0.94 with both; with `tau.r = tau.h`, 3 epochs reached 54.24% (prequential) against 51.75% without the decay
- **`24 * tau.h` for each batch** (`S = 384` periods), and the nodes learn at 0.9 of it. With `16 * tau.h`, learning at 0.75 and `tau.r = tau.h`, the first epochs matched BP as well, but the 10th was 67.28% (prequential): the relaxation slows down as the network learns
- **On the GPU, a quarter of the real time.** `device="cuda"` makes every node compute on the GPU, and the messages between them stay there ([Devices](threads-and-devices.md#devices)). An update of a node takes about 1.2 ms on an RTX 4090, with 1 sample as with 128 (the CPU launching the kernels is the limit), so the batch is 128, and with 11 workers the nodes keep up with a period of 5 ms of real time. The period is 1.25 ms of the timer, so the timer goes at a quarter of the real time (`--time-scale` 0.25), and 99% of the updates of the first node came within 1.01 periods of the one before
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

    Predictions come only from the `next` of `pc.Output` (`net.next` of a `pc.Chain`). A `Layer`'s `next` sends its activity `h` with the `y`, `mask` and `time` it passes on, but no prediction. To watch `lam` of a layer, read it from `run.state_dict()["nodes"]["layer0"]["state"]["lam"]`; reading the node's attributes while it runs races with its updates.

!!! warning "The activity is zeros until the first sample arrives"

    Before the host's first write reaches a node, the activity it receives has `time` 0 and `mask` all False, so `run.read(net.next)` returns a prediction with `mask` all False, and the nodes do not learn. To get the prediction of a batch you wrote, match its `time` with the time `write` returned, as `pc.show` does.

!!! warning "Heavy load can make the network diverge"

    The nodes exchange messages while each one updates. If a node runs late, its neighbors keep learning from its old messages, and the network can diverge or lose accuracy. MNIST stops when `y_hat` is not finite; a run that diverges with finite predictions shows only as a falling accuracy. Two causes are known:

    - **A time scale that is too fast.** MNIST speeds up the run's timer with `--time-scale`. At 2 instead of 1, the slowest node took most of its period to compute, and 1% of its updates came later than 2 periods. Run one epoch with `--watch` and take the highest time scale that gives no warning and does not diverge
    - **Other processes on the cores.** A node that waits for a core lets its neighbors run ahead. On 32 cores at time scale 1, one epoch ran through beside 26 busy processes and diverged beside 32; at 0.5 it ran through beside 32. Run on an otherwise idle machine, or lower the time scale

    `max_dt` does not prevent this. It caps the time each single update counts, and the time beyond it is dropped: from the relaxation and from the next weight update alike.

!!! warning "A module with a large Jacobian needs a shorter period"

    A node relaxes its activity `h` with what the next node sends back, which goes through the Jacobian `J` of the next node's module. The explicit step of `h` is stable only while `(1 + |J|^2) * period / tau.h < 2`, with `|J|` the largest singular value of `J`: below about 2.6 with the period `tau.h / 4` of `pc.Chain`. That is the bound when every update comes on time; a late update takes a longer step, up to `max_dt`, and a node may read what the next node sent a period before (in a linear chain, the bound then fell to about 1.2 to 1.8). When the next node is the `Output`, the curvature of the loss (at most 1/2 for the cross entropy) multiplies `|J|^2`, which loosens the bound.

    The blocks of a ResNet go past it: in `examples/cifar10/` their singular values were 2 to 9 after a few epochs, whether BP or PC-ALM learned them. Past the bound, `lam` does not settle on the credit of backpropagation. With the weights BP learned in 3 epochs, the cosine of `lam` with the credit stayed at 0.06 to 0.3 with the period `tau.h / 4`, and reached 0.95 to 0.99 with `tau.h / 16`, as with `tau.h / 64`. While it learns with `tau.h / 4`, PC-ALM keeps the blocks near the identity (singular values 1.0 to 1.7, against 2 to 6 with `tau.h / 16`), and learns less.

    Pass a shorter `period` to `pc.Chain`, with a `max_dt` of about 2 periods (the default `tau.max_dt` is 8 periods of `tau.h / 16`). Check it on the weights you have: relax a batch with the weights frozen, and compare `lam` with the gradient autograd gives. A period that gives the same `lam` as half of it is short enough. The rest is how long the credit takes to settle across many nodes, which a fast `tau.lam` and a decay of `r` (`tau.r`) shorten ([CIFAR-10](#cifar-10)).

    Two other things were tried and did not help. Starting the residual branches smaller (0.1 or 0.25 times the branch) did not keep the singular values down, as the scales of GroupNorm grew under AdamW, and made BP worse (68.5% to 69.2% in the 10th epoch, against 72.6%). Shortening the step of a node where its next node is stiff (an estimate of the largest curvature by the power iteration) made `lam` right on fixed weights, but slowed learning.

## Nodes of your own

`mujou.pc.functional` holds the computations the nodes are made of: `vjp` and `loss_vjp` (the gradients of a node's energy) and `relax_` (one step of `h` and `lam`). The optimizer moves the weights, and the relaxer moves `h`. Use them to write a node that `pc.Layer` does not cover, such as one with several inputs. The names here (`mu`, `lam`, `g_in`, `jumps`, `Tick`) are explained in [How mujou.pc Works](pc-concepts.md).

1. In `__init__`, after the module is on its device, hold the clock and the bound optimizer (and the bound relaxer, if `h` should move as `pc.relax.HeavyBall` says) as attributes. All are then in the node's state by themselves
    - `self.clock = pc.clock.LearnClock.of(tau, period=self.period, learn_interval=...)`, which takes the default `max_dt` of `tau` as `pc.Layer` does
    - `self.optimizer = optimizer.bind(params)`, with `params` the parameters that learn. Bind in each node: two nodes must not hold the same bound optimizer
    - `self.relaxer = relaxer.bind(h)`, with the `h` the node holds, in the same way. Without it, `relax_` takes the plain step
2. In each update, `t = self.clock.tick(dt, act["time"], self.timer.now())`
3. Find the prediction `mu`, `g_in` and, to learn, `g_theta` with `pc.functional.vjp`, given the bound `params` in the same order, and send `g_in` to the node before
4. When `t.new` is True or the `jumps` of the activity differ from those read last, move `h` with the prediction once: `h <- mu + (h - mu_last)`, with the `mu_last` of the previous update, and add 1 to the node's own `jumps`. Keep `mu`, the `jumps` read last and your own `jumps` in the state (`register_state` for the plain numbers)
5. Move the activity with `pc.functional.relax_(..., t.dt, tau, self.relaxer)`. When step 4 re-anchored, call `self.relaxer.reset_()` first: what it remembers are the forces of the input before
6. Send `h` as a `pc.msg.Activity`, with your own `jumps` and the `y`, `mask` and `time` of the activity you received
7. When `t.learn` is True and the `mask` of some sample is True, learn with `self.optimizer.step_([g / B for g in g_theta], t.elapsed)`, with `B` the number of samples in the batch. The node divides by `B`: the optimizer does not know the batch

Without step 4, the node follows other dynamics than `pc.Layer`: a new sample or a jump below enters its deviation `h - mu` and the credit it sends back. A node that learns at intervals also moves `h` by the jump of `mu` that its update of the weights makes, and adds 1 to its `jumps`; `pc.Layer` does both one update later, as its docstring describes.

To connect with the nodes of `mujou.pc`, declare the ports with the `act_spec` of the node before, or with `pc.msg.activity_spec(h, y, batch)`. The arguments are in the [API reference](../api/pc.md). A rule other than PC that learns across nodes is in [Update Rules](update-rules.md).

### Example: a node with two inputs

`Merge` in `examples/pc_merge.py` predicts from the activities of two towers (`pc.Layer`) side by side. It learns in every update (`learn_interval=None`), so it needs no handling of the jumps of its weights.

```python
--8<-- "examples/pc_merge.py:merge"
```

It differs from `pc.Layer` in two things only: it has a port for each input and sends each its own part of `g_in`, and it moves `h` when the `jumps` of either input change (step 4).
