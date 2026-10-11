# Mujou

**Asynchronous Distributed Deep Learning Library for Continuous Learning.**

Mujou (無常, "impermanence") is a deep learning framework that learns without backpropagation. Weights are updated with [PC-ALM](https://pub.sakana.ai/pc-alm/) (Augmented Lagrangian Predictive Coding), whose updates are local to each layer. This lets layers and neurons run inference and learning asynchronously.

Most machine learning systems are trained once and then frozen for inference. Mujou aims for machines that keep learning from a stream and never stop changing: real-time systems, robotics, lifelong learning.

> Status: early development. The API does not exist yet.

## Development

Requires [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/) (>= 1.58).

```sh
just setup   # create .venv, install deps, install pre-commit hooks
just run     # format -> test -> type
```
