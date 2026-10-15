# Mujou

**Asynchronous Distributed Deep Learning Library for Continuous Learning.**

Mujou (無常, "impermanence") is a deep learning framework where inference and learning happen at the same time. Computation is split into nodes: each node computes synchronously inside, and nodes exchange messages asynchronously. A node can be a layer (the default) or a stack of layers up to the whole model; single-neuron nodes are future work.

The default update rule is [PC-ALM](https://pub.sakana.ai/pc-alm/) (Augmented Lagrangian Predictive Coding), whose updates are local to each layer and neuron. The update rule is pluggable: other rules, including backpropagation confined to a single node, can be used instead.

Most machine learning systems are trained once and then frozen for inference. Mujou aims for machines that keep learning from a stream and never stop changing: real-time systems, robotics, lifelong learning.

> Status: early development. The API does not exist yet.

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/) (>= 1.58).

```sh
just setup   # create .venv, install deps, install pre-commit hooks
just run     # format -> test -> type
```
