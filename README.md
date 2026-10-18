# Mujou

**Asynchronous Distributed Deep Learning Library for Continuous Learning.**

[![Test](https://github.com/Geson-anko/mujou/actions/workflows/test.yml/badge.svg)](https://github.com/Geson-anko/mujou/actions/workflows/test.yml)
[![Type Check](https://github.com/Geson-anko/mujou/actions/workflows/type-check.yml/badge.svg)](https://github.com/Geson-anko/mujou/actions/workflows/type-check.yml)
[![Format & Lint](https://github.com/Geson-anko/mujou/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Geson-anko/mujou/actions/workflows/pre-commit.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/)

Mujou (無常, "impermanence") is a deep learning framework where inference and learning happen at the same time. You write nodes in PyTorch code and connect them into a network that runs in the background. Most models are trained once and then frozen; Mujou is for machines that keep learning from a stream, such as real-time systems and robots.

> **Status: early development.** The API works but may still change.

## Key ideas

- **No epochs, no train/inference modes.** A running network takes input, produces output and learns, all at once.
- **Synchronous inside, asynchronous outside.** Each node computes synchronously in ordinary PyTorch code. Nodes never wait for each other; a node reads the latest value its neighbour sent.
- **One global time.** A node updates by an amount proportional to `dt`, the time since its last update, so nodes that update at different rates share one flow of time.
- **Pluggable update rule.** The default will be [PC-ALM](https://pub.sakana.ai/pc-alm/) (Augmented Lagrangian Predictive Coding; not implemented yet), which learns with updates local to each node. Other rules work too, including backpropagation kept inside one node.

## Install

Mujou is not on PyPI yet; build it from source. You need [uv](https://docs.astral.sh/uv/), a Rust toolchain ([rustup](https://rustup.rs/)) and Python 3.14 (uv can install it). The free-threaded build 3.14t is recommended, because each node runs on a thread of its own.

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
uv sync   # creates .venv/, installs PyTorch and builds the Rust core
```

## Example

Two nodes compute `y = 2x + 1` on their own threads; the program writes `x` and reads `y`.

```python
from typing import override

import torch

import mujou
from mujou import NextPort, Node, PrevPort, TensorSpec

VECTOR = TensorSpec((3,))  # every message is a float32 vector of 3


class Affine(Node):
    """Sends a * x + b, where x is the latest input."""

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a, self.b = a, b
        self.prev = PrevPort(recv=VECTOR)  # receives from the previous node
        self.next = NextPort(send=VECTOR)  # sends to the next node

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.a * self.prev.recv() + self.b)  # recv() never waits


double, shift = Affine(2.0, 0.0), Affine(1.0, 1.0)
double.next.connect(shift.prev)
x = torch.tensor([1.0, 2.0, 3.0])
# Each node updates on a thread of its own until the `with` block ends.
with mujou.run(double, shift, inputs={double.prev: x}) as run:
    # y arrives after a few updates: wait for it (and the time it was sent).
    y, t = run.wait_for(shift.next, lambda y: torch.equal(y, 2 * x + 1), timeout=5.0)

print(f"y = {y} at t = {float(t):.3f} s")  # y = tensor([3., 5., 7.]) at t = ...
```

## Documentation

The documentation is at <https://geson-anko.github.io/mujou/> (English and Japanese). To serve it locally, run `just docs` (needs [just](https://just.systems/)) and open <http://127.0.0.1:8000>. You can also read the Markdown in [`docs/en/`](docs/en/) (English) and [`docs/ja/`](docs/ja/) (Japanese). Working programs are in [`examples/`](examples/), and the tests run every one of them.

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the dev container, the commands and the rules.

## License

MIT. See [LICENSE](LICENSE).
