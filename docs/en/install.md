# 📦 Installation

## Requirements

| Requirement | Details                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| Python      | 3.14 or later. The free-threaded build 3.14t is recommended. The regular 3.14 build with the GIL also works |
| PyTorch     | Installed automatically as a dependency of Mujou                                                            |
| Rust        | 1.85 or later, installed with [rustup](https://rustup.rs/). Needed only to build from source                |
| uv          | [uv](https://docs.astral.sh/uv/). Needed only to build from source                                          |

The free-threaded build is recommended because the nodes run on several worker threads at once. With the GIL, nodes written in Python compete for the GIL and rarely run in parallel.

!!! warning "The API still changes"

    Mujou is in its 0.x releases: a minor release (0.x.0) may break the API, while a patch release (0.x.y) only fixes bugs. Pin the minor version, such as `mujou~=0.1.0`, and read the [Changelog](changelog.md) before you upgrade.

## Install from PyPI

In a uv project:

```sh
uv add mujou
```

Or with pip, in a virtual environment with Python 3.14 or later:

```sh
pip install mujou
```

Wheels are published for Linux (x86_64 and aarch64), macOS (arm64) and Windows (x86_64), each for 3.14 and 3.14t. On other platforms, pip builds Mujou from the sdist, which needs Rust.

PyTorch is installed from PyPI, like the other dependencies. To use another build, such as a CPU-only one or another CUDA version, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Build from source

To run the example programs or to work on Mujou itself, build it from a clone of the repository:

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
uv sync
```

`uv sync` sets up the free-threaded Python 3.14t, creates `.venv/`, installs the dependencies (with what the example programs need, such as numpy), and builds the core, which is written in Rust. For the regular build with the GIL, run `uv sync --python 3.14` instead.

## Check the installation

In the clone of the repository:

```sh
uv run python -c "import mujou; print(mujou.__version__)"  # prints a version such as 0.1.0
uv run python examples/quickstart.py  # prints a line such as: y = tensor([3., 5., 7.]) at t = 0.017 s
```

After `pip install mujou`, run the first line with `python` instead of `uv run python`, and copy [`examples/quickstart.py`](https://github.com/Geson-anko/mujou/blob/main/examples/quickstart.py) to run it.

The [first page of the tutorial](tutorial/1-run.md) explains the second program and its output.
