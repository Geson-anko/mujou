# 🧠 5. Learn with PC-ALM

The module `mujou.pc` learns a PyTorch model split into nodes with PC-ALM (Augmented Lagrangian Predictive Coding, <https://pub.sakana.ai/pc-alm/>). Each node holds a part of the model and learns from its neighbors' messages only; at equilibrium, its gradients are those of backpropagation. This page learns a regression with it. What you choose are the time constants and how long each batch is shown.

## The program

```sh
uv run python examples/pc_regression.py
```

```text
batch   1: loss 0.2079 (the mean of the last 1)
batch  50: loss 0.0192 (the mean of the last 50)
...
batch 300: loss 0.0067 (the mean of the last 50)
```

## The chain

```python
--8<-- "examples/pc_regression.py:chain"
```

`pc.Chain` makes one node for each module and connects them in order, as `nn.Sequential` computes them: a `pc.Input` that takes the samples, a `pc.Layer` for each module but the last (`layer0`, `layer1`), and a `pc.Output` for the last, with the loss. The children of the chain are `input`, `layer0`, `layer1` and `output`, which are also their paths in the state.

- `x` and `y` are the specs of the inputs and targets, batch first. The batch size `B` is written only here
- `loss` returns the loss of each sample, shape `(B,)`: `pc.mse` or `pc.cross_entropy`, or a function of your own of that shape
- `tau` holds the time constants, in seconds of the run's timer: `h` for the activities (every node of the chain updates once every `tau.h / 4`), `lam` for the duals and `theta` for the weights
- `learn_interval` is when the nodes learn, counted from each write: here once per batch, three quarters of the way through it. A learning step weighs the time since the node last learned, so with one step per batch, the learning rate per batch is the showing time divided by `tau.theta`

Inside a node, the module is differentiated by autograd, as in BP. Initialize the weights when you make the modules (here from `torch.manual_seed` in the host, before the run); the nodes do not.

## Show and learn

```python
--8<-- "examples/pc_regression.py:run"
```

`pc.show(run, net, sample, seconds=SHOWN)` writes the batch, waits for the first prediction the chain makes for it, and returns that prediction once the batch has been shown for `SHOWN` seconds of the run's timer. Meanwhile the nodes relax towards the target and learn from it. The prediction comes before any node has learned from the batch, so its loss measures how well the model predicts data it has not seen yet (the prequential loss).

- `mask` says which samples have a target. A sample whose `mask` is False adds nothing to the loss: with every `mask` False, no weight moves (the activities still do), which is how to evaluate the same running chain. There is no inference mode to switch to
- A `pc.Prediction` holds `y_hat` with the `y`, `mask` and `time` of the sample it was made from, so a prediction always pairs with its own target
- `run.state_dict()["nodes"]["layer0"]["state"]` is the state of the first layer: its activity `h`, its dual `lam` (the credit of backpropagation at equilibrium), its module and its clock

## Next

- A 784-256-256-10 network that learns MNIST with `pc.Chain` and compares itself with BP: `examples/mnist/` (`uv run python examples/mnist/main.py`), explained in [PC-ALM](../topics/pc.md#mnist)
- How to choose the time constants and `learn_interval`, the pitfalls, and nodes of your own: [PC-ALM](../topics/pc.md)
