# 🧠 5. Learn with PC-ALM

The module `mujou.pc` learns a PyTorch model split into nodes with PC-ALM (Augmented Lagrangian Predictive Coding, <https://pub.sakana.ai/pc-alm/>). Each node holds a part of the model and learns from its neighbors' messages only; at equilibrium, its gradients are those of backpropagation. This page learns a regression with it. What you choose are the time constants, the optimizer of the weights, and how long each batch is shown.

## The program

```sh
uv run python examples/pc_regression.py
```

```text
batch   1: loss 0.4157 (the mean of the last 1)
batch  50: loss 0.0489 (the mean of the last 50)
...
batch 300: loss 0.0080 (the mean of the last 50)
```

## The chain

```python
--8<-- "examples/pc_regression.py:chain"
```

`pc.Chain` makes one node for each module and connects them in order, as `nn.Sequential` computes them: a `pc.Input` that takes the samples, a `pc.Layer` for each module but the last (`layer0`, `layer1`), and a `pc.Output` for the last, with the loss. The children of the chain are `input`, `layer0`, `layer1` and `output`, which are also their paths in the state; `net.layers` gives the layers in order, `net.layers[0]` being `layer0`.

- `x` and `y` are the specs of the inputs and targets, and `batch` is the shape they start with: `B` samples here. Without `batch`, the chain takes one sample at a time, and `x` and `y` would be `TensorSpec((2,))` and `TensorSpec((1,))`
- `loss` returns the loss of each sample, of the shape of the batch: `pc.loss.MSELoss(B)` (the mean squared error of each sample, averaged over its dimensions as `nn.MSELoss` does) or `pc.loss.CrossEntropyLoss(B)`, or a function of your own that returns that shape. Give the loss the same batch as the chain ([the batch](../topics/pc.md#the-batch))
- `tau` holds the time constants of the activities and the duals, in seconds of the run's timer: `h` for the activities (every node of the chain updates once every `tau.period`, which is `tau.h / 4`) and `lam` for the duals. The time constant of the weights is in the optimizer
- `optimizer` moves the weights when a node learns. `pc.optim.SGD(tau=...)` is gradient descent with the time constant `tau` of the weights: a learning step moves the weights by (the time since the node last learned) / `tau` times the mean gradient of the batch. Here the nodes learn once per batch, so the learning rate per batch is `SHOWN / optimizer.tau` = 0.3, where `SHOWN` is the time each batch is shown. `pc.optim.AdamW` is the other choice ([Choosing the optimizer](../topics/pc.md#choosing-the-optimizer)). Every node binds the optimizer to its own weights, so one `pc.optim.SGD` serves the whole chain. It has no default
- `learn_interval` is when the nodes learn, counted from each write: here once per batch, three quarters of the way through it. It has no default: `None` learns in every update, for a stream whose input changes at every frame, and good values for that mode have not been measured yet

Inside a node, the module is differentiated by autograd, as in BP. Initialize the weights when you make the modules (here from `torch.manual_seed` in the host, before the run); the nodes do not.

## Show and learn

```python
--8<-- "examples/pc_regression.py:run"
```

`pc.show(run, net, sample, seconds=SHOWN)` writes the batch, waits for the first prediction the chain makes for it, and returns that prediction once the batch has been shown for `SHOWN` seconds of the run's timer. Meanwhile the nodes relax towards the target and learn from it. The prediction comes before any node has learned from the batch, so its loss measures how well the model predicts data it has not seen yet (the prequential loss).

- `mask` says which samples have a target. A sample whose `mask` is False adds nothing to the loss: with every `mask` False, no weight moves (the activities still do), which is how to evaluate the same running chain. There is no inference mode to switch to
- A `pc.msg.Prediction` holds `y_hat` with the `y`, `mask` and `time` of the sample it was made from, so a prediction always pairs with its own target
- `run.state_dict()["nodes"]["layer0"]["state"]` is the state of the first layer: the deviation `r` of its activity from its prediction, its dual `lam` (the credit of backpropagation at equilibrium), its module, its clock, its optimizer and its relaxer. The activity `h = mu + r` it sends is in the state of its port `next` ([How mujou.pc Works](../topics/pc-concepts.md#what-a-node-holds))

## Next

- How the parts connect, and what a node does in one update: [How mujou.pc Works](../topics/pc-concepts.md)
- A 784-256-256-10 network that learns MNIST with `pc.Chain` and compares itself with BP: `examples/mnist/` (`uv run python examples/mnist/main.py`), explained in [PC-ALM](../topics/pc.md#mnist)
- A ResNet-20 that learns CIFAR-10 on the GPU with `pc.Chain` and compares itself with BP: `examples/cifar10/` (`uv run python examples/cifar10/main.py`), explained in [PC-ALM](../topics/pc.md#cifar-10)
- How to choose the time constants, the optimizer and `learn_interval`, the pitfalls, and nodes of your own: [PC-ALM](../topics/pc.md)
