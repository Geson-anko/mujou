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
- **One time axis per run.** A node updates by an amount proportional to `dt`, the time since its last update, so nodes that update at different rates share one flow of time.
- **Any update rule.** The rule is written inside the nodes: [PC-ALM](https://pub.sakana.ai/pc-alm/) (Augmented Lagrangian Predictive Coding, the nodes of `mujou.pc`), which learns with updates local to each node, backpropagation kept inside one node, or a rule of your own.

## Install

Mujou is not on PyPI yet; build it from source. You need [uv](https://docs.astral.sh/uv/), a Rust toolchain ([rustup](https://rustup.rs/)) and Python 3.14 (uv can install it). The free-threaded build 3.14t is recommended, because the nodes run on several threads at once.

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
uv sync   # creates .venv/, installs PyTorch and builds the Rust core
```

## Example

Two nodes compute `y = 2x + 1` in the background; the program writes `x` and waits for `y` (`examples/quickstart.py`).

```python
from typing import override

import torch

import mujou
from mujou import Network, Node, Port, TensorSpec

VECTOR = TensorSpec((3,))  # every message here is a float32 vector of 3


class Affine(Node):
    """Sends ``a * x + b``, where ``x`` is the latest input."""

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        self.a, self.b = a, b  # settings, not state
        self.prev = Port(recv=VECTOR)  # receives from the previous node
        self.next = Port(send=VECTOR)  # sends to the next node

    @override
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        self.next.send(self.a * self.prev.recv() + self.b)  # recv() never waits


class Pipeline(Network):
    """``x -> scale -> shift ->``: y = 2 x + 1."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Affine(2.0, 0.0)
        self.shift = Affine(1.0, 1.0)
        self.scale.next.connect(self.shift.prev)
        self.prev, self.next = self.scale.prev, self.shift.next  # exposed as its own


net = Pipeline()
x = torch.tensor([1.0, 2.0, 3.0])
# The nodes run on the run's worker threads until the end of the `with` block.
with mujou.run(net) as run:
    run.write({net.prev: x})
    # x takes a few updates to pass through the nodes: wait until y arrives, with
    # the time it was sent (TimeoutError after 5 s of real time).
    y, t = run.wait_for(net.next, lambda y: torch.equal(y, 2 * x + 1), timeout=5.0)
    assert t is not None  # None only for the zeros before the first send

print(f"y = {y} at t = {float(t):.3f} s")
```

More programs, from writing a node to learning MNIST with PC-ALM, are in [`examples/`](examples/).

## Documentation

The documentation is at <https://geson-anko.github.io/mujou/> (English and Japanese). To serve it locally, run `just docs` (needs [just](https://just.systems/)) and open <http://127.0.0.1:8000>. You can also read the Markdown in [`docs/en/`](docs/en/) (English) and [`docs/ja/`](docs/ja/) (Japanese).

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the dev container, the commands and the rules.

## License

MIT. See [LICENSE](LICENSE).
