# Installation

Mujou is not on PyPI yet. For now, build it from source.

## Requirements

| Requirement | Details                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| Python      | 3.14 or later. The free-threaded build 3.14t is recommended. The regular 3.14 build with the GIL also works |
| PyTorch     | Installed automatically as a dependency of Mujou                                                            |
| Rust        | 1.85 or later, installed with [rustup](https://rustup.rs/). Needed only to build from source                |
| uv          | [uv](https://docs.astral.sh/uv/). Needed only to build from source                                          |

The free-threaded build is recommended because each node runs on a thread of its own. With the GIL, nodes written in Python compete for the GIL and rarely run in parallel.

## Build from source

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
uv sync
```

`uv sync` sets up the free-threaded Python 3.14t, creates `.venv/`, installs the dependencies, and builds the core, which is written in Rust. For the regular build with the GIL, run `uv sync --python 3.14` instead.

On Linux, `uv sync` installs PyTorch from the CPU-only index, because the Mujou core does not use the GPU. To use a GPU inside your nodes, replace PyTorch with a CUDA build by following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

!!! note "After the release on PyPI"

    These commands will work once Mujou is on PyPI: `uv add mujou` in a uv project, or `pip install mujou` in a virtual environment with Python 3.14 or later.

## Check the installation

```sh
uv run python -c "import mujou; print(mujou.__version__)"  # prints a version such as 0.0.1
uv run python examples/quickstart.py  # prints a line such as: y = tensor([3., 5., 7.]) at t = 0.017 s
```

The [Quickstart](quickstart.md) explains the second program and its output.
