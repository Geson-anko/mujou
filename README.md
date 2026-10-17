# Mujou

**Asynchronous Distributed Deep Learning Library for Continuous Learning.**

Mujou (無常, "impermanence") is a deep learning framework where inference and learning happen at the same time. Computation is split into nodes: each node computes synchronously inside, and nodes exchange messages asynchronously. A node can be a layer (the default) or a stack of layers up to the whole model; single-neuron nodes are future work.

The default update rule will be [PC-ALM](https://pub.sakana.ai/pc-alm/) (Augmented Lagrangian Predictive Coding, not implemented yet), whose updates are local to each layer and neuron. The update rule is pluggable: other rules, including backpropagation confined to a single node, can be used instead.

Most machine learning systems are trained once and then frozen for inference. Mujou aims for machines that keep learning from a stream and never stop changing: real-time systems, robotics, lifelong learning.

> Status: early development. The API works but may still change. Read the documentation (English and Japanese) with `just docs`; the programs in `examples/` are run by the tests.

## Development

Requires Python 3.14 or later. Development uses the free-threaded build 3.14t, set in `.python-version`; the regular build with the GIL is supported too.

Also requires [uv](https://docs.astral.sh/uv/), [just](https://just.systems/) (>= 1.58) and a Rust toolchain ([rustup](https://rustup.rs/)) for the native core `mujou._core._native`, which `uv sync` builds with maturin.

```sh
just setup   # create .venv, install deps, install pre-commit hooks
just run     # format -> test -> type -> docs-build
```

### Documentation

The user documentation (English and Japanese) is in `docs/` and built with MkDocs; the design specifications for developers (Japanese) are in `specs/`.

```sh
just docs         # serve the site at http://127.0.0.1:8000 with live reload
just docs-build   # build it strictly into site/
```

### Dev container

`.devcontainer/` provides Ubuntu with uv, just and the Rust toolchain. Open the main checkout in VS Code and choose "Reopen in Container", or use the [Dev Container CLI](https://github.com/devcontainers/cli) (`brew install devcontainer` or `npm install -g @devcontainers/cli`) through just, from the main checkout or any worktree under `.claude/worktrees/`:

```sh
just container just run   # start the container if needed, then run `just run` in this checkout
```

- The checkout is mounted at the same absolute path as on the host, so git worktrees work in the container: a worktree's `.git` file holds an absolute host path. Worktrees must live inside the main checkout (e.g. `.claude/worktrees/`) to be visible.
- The container keeps its own Linux virtualenv in `.venv-devcontainer/` of each checkout (`UV_PROJECT_ENVIRONMENT`), next to the host's `.venv/`. Cargo builds share each checkout's `target/` with the host: the host triple is part of cargo's artifact hashes, so Linux and macOS artifacts live side by side. Do not point several checkouts at one `CARGO_TARGET_DIR`; cargo does not tell path dependencies of different checkouts apart.
- Files are written as the `vscode` user. On Linux hosts the Dev Container CLI changes its UID/GID to yours (`updateRemoteUserUID`); Docker Desktop on macOS maps ownership to the host user by itself. No `sudo` is needed.
- The container runs `uv sync` instead of `just setup`, and installs the pre-commit hook only if none exists, because the hook records the absolute path of the Python that installed it. The main checkout's `.venv-devcontainer/bin` is on `PATH`, so a hook installed on the host still finds `pre-commit` in the container. If you open a fresh clone in the container first, run `just setup` on the host before committing there.
- `core.fileMode` is `false` inside the container only, because Docker Desktop bind mounts report every file as executable. To commit a new executable from the container, use `git update-index --chmod=+x <file>`.

## License

MIT. See [LICENSE](LICENSE).
