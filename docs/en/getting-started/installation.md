# Installation

There are two ways to install Mujou. For now, only building from source works: Mujou is not on PyPI yet.

## Requirements

| Requirement | Details                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------- |
| Python      | 3.14 or later. The free-threaded build 3.14t is recommended. The regular 3.14 build with the GIL also works |
| PyTorch     | Installed automatically as a dependency of Mujou. You do not need to install it yourself                    |
| Rust        | Needed only to build from source. Install 1.85 or later with [rustup](https://rustup.rs/)                   |

The free-threaded build is recommended because each node runs on a thread of its own. With the GIL, nodes written in Python compete for the GIL and rarely run in parallel.

## Build from source

Install the Rust toolchain and [uv](https://docs.astral.sh/uv/) first.

1. Get the repository.

    ```sh
    git clone https://github.com/Geson-anko/mujou.git
    cd mujou
    ```

2. Create the environment. `uv sync` sets up Python, creates `.venv/`, installs the dependencies, and builds the core, which is written in Rust.

    | Python                    | Command                 |
    | ------------------------- | ----------------------- |
    | Free-threaded build 3.14t | `uv sync`               |
    | Regular build 3.14 (GIL)  | `uv sync --python 3.14` |

On Linux, `uv sync` installs PyTorch from the CPU-only index. The Mujou core does not use the GPU, so it avoids the CUDA build, which takes several GB. To use a GPU inside your nodes, replace PyTorch with a CUDA build by following the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Install from PyPI (after release)

!!! warning "Not on PyPI yet"

    The commands in this section will work once Mujou is published on PyPI. For now, build from source.

In a [uv](https://docs.astral.sh/uv/) project, use `uv add mujou`. With pip, run `pip install mujou` in a virtual environment with Python 3.14 or later. To check a pip install, activate the virtual environment and run `python -c "import mujou; print(mujou.__version__)"`. As in "Check the installation" below, a printed version such as `0.0.1` means that Mujou is installed.

## Check the installation

Print the version.

```sh
uv run python -c "import mujou; print(mujou.__version__)"
```

If it prints a version such as `0.0.1`, Mujou is installed.

If you built from source, you can also run the quickstart program.

```sh
uv run python examples/quickstart.py
```

If it prints a line such as `y = tensor([3., 5., 7.]) at t = 0.017 s`, it works. The value of `t` varies from run to run. The [Quickstart](quickstart.md) explains how to read this output.
