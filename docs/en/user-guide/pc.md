# 🧠 Learning with PC-ALM

The module `mujou.pc` trains a PyTorch model with PC-ALM (Augmented Lagrangian Predictive Coding, <https://pub.sakana.ai/pc-alm/>), split into nodes. This page shows how to build the nodes, stream a dataset through them, measure the accuracy, and choose the time constants. The running example is `examples/mnist/`: a 784-256-256-10 ReLU network that learns MNIST and is compared with BP.

The page assumes you have read [Building a network](building-a-network.md) (in particular [Carrying data along a path](building-a-network.md#carrying-data-along-a-path)) and [Running a network](running.md).

## How it differs from BP

| Topic                     | BP in plain PyTorch                                  | `mujou.pc`                                                                                                                       |
| ------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Inference and learning    | Forward, backward and `optimizer.step()`, one by one | Run at the same time. Every node updates on its own period, and learns while the input is shown                                  |
| One training step         | One batch                                            | The host shows a batch for a while (`S` periods). The nodes relax towards the target, then update their weights                  |
| The target                | Passed to the loss with the batch                    | Written together with the input, and carried with the activity from node to node until the output node                           |
| Training and test         | `model.train()` / `model.eval()`                     | No modes. A sample whose `mask` is False is not learned from, so the test batches go through the same running nodes              |
| The loss                  | Usually the batch mean                               | The loss of each sample, shape `(B,)`. The output node sums it                                                                   |
| Where gradients are taken | `.backward()` through the whole model                | Inside each node only. A node exchanges activities and energy gradients with its neighbors as messages                           |
| Cost                      | One forward and backward per batch                   | One forward and VJP per node per period, so `S` of them per batch. On the CPU, the example takes about 40 times longer per batch |

The accuracy is close to BP. After the 3 epochs of the example, PC-ALM reached 97.82% and 97.59% test accuracy (seeds 0 and 1), and BP 97.54% and 97.45%, with the same initial weights and the same batches.

## The nodes

| Node        | What it holds                                     | Ports                                                                                                                                                                  |
| ----------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pc.Input`  | Nothing                                           | `prev`: the host writes a `pc.Sample`, `{"x": ..., "y": ..., "mask": ...}`, here. `next`: sends `x` as a `pc.Activity`, carrying `y`, `mask` and the time of the write |
| `pc.Layer`  | A module `f`, its activity `h` and the dual `lam` | `prev` and `next`. `next` sends `h` as a `pc.Activity` and passes on the carried data that came with its input                                                         |
| `pc.Output` | The last module and the loss                      | `prev`. `next` sends a `pc.Prediction` (`y_hat`, `y`, `mask`, `time`) for the host to read                                                                             |

A model is a chain `Input` → `Layer` → ... → `Output`. Put in a `Layer` any `nn.Module` that treats each sample of the batch on its own: one layer, a stack of layers, or a residual block. `Dropout` and `BatchNorm` in PyTorch's training mode mix or drop samples, so do not use them; `LayerNorm` is fine. Inside a node the module is differentiated by autograd, as in BP.

## Building the network

```python
--8<-- "examples/mnist/main.py:build"
```

1. Make the spec of the carried data once with `pc.carry_spec(target_spec, batch=B)`, and pass it to every node as `carry`. The target spec of class labels is `TensorSpec((B,), torch.int64)`; that of `D` real values (for `pc.mse`) is `TensorSpec((B, D))`
2. Give each node the shape of what it receives, batch first: `pc.Input((B, 784), ...)`, `pc.Layer(module, input_shape=(B, 784), ...)`. A node finds the shape of what it sends by calling its module once on zeros
3. Connect `next` to `prev` along the chain
4. Put the nodes in a network to run them. The example uses `mujou.Group`

`WatchedLayer` and `WatchedOutput` are `pc.Layer` and `pc.Output` with a counter of `dt` added; use `pc.Layer` and `pc.Output` in your own code.

Initialize the weights when you create the modules: the nodes do not initialize them. A parameter with `requires_grad=False` does not learn.

## Choosing the time constants

Every rate in `mujou.pc` is a time in seconds of the run's timer. Choose them in this order.

```python
--8<-- "examples/mnist/main.py:tau"
```

| Value                   | What it sets                                                        | How to choose it                                                                                                                                                          | Example |
| ----------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `tau.h`                 | How fast each activity `h` relaxes                                  | Choose it first. The others follow from it                                                                                                                                | 4 ms    |
| The period of the nodes | How often each node updates                                         | `Layer` and `Output` default to `tau.h / 4`. `Input` needs `period=`; give it the same                                                                                    | 1 ms    |
| `tau.lam`               | How fast the dual `lam` follows                                     | `2 * tau.h`                                                                                                                                                               | 8 ms    |
| The showing time        | How long the host shows each batch                                  | `S` periods, with `S = 32`                                                                                                                                                | 32 ms   |
| `learn_interval`        | When the weights update, counted from the write                     | Longer than half of the showing time and than `2 * depth * period`, and shorter than the showing time. The example uses 3/4                                               | 24 ms   |
| `tau.theta`             | The time constant of the weights (the learning rate is its inverse) | The showing time divided by the learning rate per batch                                                                                                                   | 0.32 s  |
| `max_dt`                | The longest `dt` one update counts                                  | Keep the default, `min(tau.h / 2, 3 * tau.lam)` (2 periods here). A `max_dt` shorter than the period raises `ValueError`; so does the default when `tau.lam < tau.h / 12` | 2 ms    |

- **The learning rate.** A weight update moves the weights by (the time since the node's last weight update) / `tau.theta` times the mean gradient over the batch. With one update per batch, that time is the showing time, so `tau.theta = showing time / lr` gives the learning rate `lr` per batch
- **The time since the last weight update.** The sum of the `dt` of the updates in between, each capped at `max_dt`. Updates on time have a `dt` of one period, shorter than `max_dt`, so the 32 ms showing time of the example is counted in full
- **The mean gradient.** The sum of the per-sample gradients divided by the batch size `B`. A sample whose `mask` is False adds nothing to the sum but still counts in `B`
- **Why `learn_interval` is between half and all of the showing time.** `learn_interval` is counted from the time the host wrote the input (the carried `time`), not from when the node received it, so the start is the same however deep the node is. A node learns `learn_interval` after each write, and again every `learn_interval` while the batch is still shown. If `learn_interval` is shorter than half, the node learns twice or more per batch. A node never learns in the update in which it first sees a batch, and the credit needs time to go down the path and back, so give it at least about `2 * depth * period` (`depth`: the number of nodes on the path; 8 ms in the example). If it is as long as the showing time or longer, the next batch is written first: the node never learns, and drops the time it counted
- **A stream whose input changes at every frame.** Pass `learn_interval=None`: the node then learns in every update, whether or not the input changed, and `1 / tau.theta` is the learning rate per second. Good values for streams have not been measured yet
- **Stability.** A shorter period relative to `tau.h` is more stable. In the measurements behind the design, `period / tau.h` up to 0.5 was stable for ReLU, and `period / tau.lam` between 0.1 and 0.3 worked best. `S = 32` gave the accuracy of BP; shorter showing times have not been verified

## Training and measuring

The host writes each batch to `inp.prev` and reads the predictions from `out.next`.

```python
--8<-- "examples/mnist/main.py:show"
```

`run.write` returns the time of the write, a `Timestamp` `t` (`t.ns` in ns). Reading `out.next` right after the write returns a prediction of an earlier batch (zeros before the first one), because the new input has not passed through the nodes yet. The carried `time` of a prediction is the time of the write it was computed from (in ns), so `wait_for` with `int(p["time"]) == t.ns` returns the first prediction made from this batch. Then the host waits on the run's timer until the batch has been shown for the showing time.

```python
--8<-- "examples/mnist/main.py:run"
```

- **Prequential accuracy** (during training): the first prediction for each batch, made before any node has learned from it. It measures how well the model predicts data it has not seen yet
- **Test accuracy**: the host writes the test batches with `mask` all False, so no node learns from them (keep `y`: the host scores with it), and reads the latest prediction with `run.read(out.next)` after the showing time. There is no inference mode to switch to
- `data.batches` pads the last batch with zeros whose `mask` is False. Count only the samples whose original `mask` is True. The test writes `mask` all False, so keep the original `mask` for scoring

The example sets `torch.set_num_threads(1)`, since the nodes already compute in parallel on the run's workers (see [PyTorch in nodes](torch.md#internal-threads)).

## Pitfalls

!!! warning "The loss returns the loss of each sample"

    Pass `pc.cross_entropy`, `pc.mse`, or your own function `(y_hat, y) -> (B,)`. A loss that averages, such as `nn.MSELoss()`, makes `pc.Output` raise `ValueError`. `pc.mse` is `0.5 * ||y - y_hat||^2` summed over the features, not the mean.

!!! warning "Read predictions from `Output.next`"

    The host can read only `out.next`. `run.read(layer.next)` raises `ValueError`, since a `Layer`'s `next` passes on carried data, which the host cannot read. To watch `h` and `lam` of a `Layer`, read them from `run.state_dict()["nodes"][path]["h"]` and `["lam"]`, where `path` is the attribute name of the node in the network (`"l1"` in the example); reading `layer.h` from the host races with the node's updates.

!!! warning "The carried data is zeros until the first sample arrives"

    Before the host's first write reaches a node, the carried data is zeros: `mask` all False and `time` 0. So `run.read(out.next)` then returns a prediction with `mask` all False, and the nodes do not learn. To get the prediction of a batch you wrote, match its `time` with `wait_for`, as `show` does.

!!! warning "Heavy load can make the network diverge"

    The nodes exchange messages while each one updates. If a node runs late, its neighbors keep learning from its old messages, and the network can diverge or lose accuracy. `show` stops the example when `y_hat` is not finite (`torch.isfinite`); a run that diverges with finite predictions shows only as a falling accuracy. How fast a machine can run the nodes safely depends on the machine. Two causes are known:

    - **A time scale that is too fast.** The example speeds up the run's timer with `set_time_scale`. At 2 instead of 1, the slowest node took most of its period to compute, and 1% of its updates came later than 2 periods: one epoch ran through, but with a warning. The example prints the `dt` of the nodes and warns when updates come later than 1.5 periods (median) or 2 periods (99%). Run one epoch and take the highest time scale that gives no warning and does not diverge
    - **Other processes on the cores.** A node that waits for a core lets its neighbors run ahead. On 32 cores at time scale 1, one epoch ran through beside 26 busy processes and diverged beside 32; at 0.5 it ran through beside 32. Run on an otherwise idle machine, or lower the time scale

    `max_dt` does not prevent this. It caps the time each single update counts, and the time beyond it is dropped: from the relaxation and from the next weight update alike.

## Writing a node of your own

`mujou.pc.functional` holds the computations the nodes are made of: `vjp` and `loss_vjp` (the gradients of a node's energy), `relax_` (one step of `h` and `lam`) and `learn_` (the weight update). Use them to write a node that `pc.Layer` does not cover, such as one with several inputs. Such a node receives and sends `pc.Activity` (`DictSpec(pc.Activity, h=..., epoch=TensorSpec((), torch.int64))`) to connect with the nodes of `mujou.pc`. See the [API reference](../api/pc.md).
