# Contributing to Mujou

Issues and pull requests are welcome: bug reports, questions, documentation fixes and code. Keep each pull request small and about one thing. The issue forms ask for what you need to say: for a bug, a minimal program and your environment. You may write issues and pull requests in English or Japanese.

For a large change, open an issue first so that we can agree on the design before you write the code. A change is large when it changes or adds public API, adds a dependency, changes how nodes run or exchange messages (the execution model), or touches many modules. Bug fixes, documentation fixes and small additions can go straight to a pull request.

Mujou is in early development, and the API may still change.

## Prerequisites

| Tool                                | Version and notes                                                                                                                                                           |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [uv](https://docs.astral.sh/uv/)    | Manages Python, the virtualenv and the dependencies                                                                                                                         |
| [just](https://just.systems/)       | 1.58 or later. On Windows, the recipes run through Git Bash (`C:/Program Files/Git/bin/bash.exe`), so install [Git for Windows](https://gitforwindows.org/) there           |
| Rust ([rustup](https://rustup.rs/)) | The version pinned in `rust-toolchain.toml`; rustup installs it. The code also builds with 1.85 (`rust-version` in `Cargo.toml`)                                            |
| Python                              | 3.14 or later. Development uses the free-threaded build 3.14t, set in `.python-version`; uv installs it. The regular build with the GIL is supported too, and CI tests both |

## Setup

```sh
git clone https://github.com/Geson-anko/mujou.git
cd mujou
just setup   # uv venv --clear, uv sync, pre-commit install
```

`uv sync` builds the native module `mujou._core._native` from the Rust crates with maturin. `uv sync` and `uv run` rebuild it when `pyproject.toml`, `Cargo.toml`, `Cargo.lock`, or any `Cargo.toml` or `.rs` file under `crates/` changes (`cache-keys` in `[tool.uv]` of `pyproject.toml`).

## Dev container

`.devcontainer/` provides Ubuntu with uv, just and the Rust toolchain. Open the main checkout in VS Code and choose "Reopen in Container", or use the [Dev Container CLI](https://github.com/devcontainers/cli) (`brew install devcontainer` or `npm install -g @devcontainers/cli`) through just, from the main checkout or any of its worktrees:

```sh
just container just run   # start the container if needed, then run `just run` in this checkout
```

- The checkout is mounted at the same absolute path as on the host, so git worktrees work in the container: a worktree's `.git` file holds an absolute host path. Worktrees must live inside the main checkout to be visible in the container.
- The container keeps its own Linux virtualenv in `.venv-devcontainer/` of each checkout (`UV_PROJECT_ENVIRONMENT`), next to the host's `.venv/`. Cargo builds share each checkout's `target/` with the host: the host triple is part of cargo's artifact hashes, so Linux and macOS artifacts live side by side. Do not point several checkouts at one `CARGO_TARGET_DIR`; cargo does not tell path dependencies of different checkouts apart.
- Files are written as the `vscode` user. On Linux hosts the Dev Container CLI changes its UID/GID to yours (`updateRemoteUserUID`); Docker Desktop on macOS maps ownership to the host user by itself. No `sudo` is needed.
- The container runs `uv sync` instead of `just setup`, and installs the pre-commit hook only if none exists, because the hook records the absolute path of the Python that installed it. The main checkout's `.venv-devcontainer/bin` is on `PATH`, so a hook installed on the host still finds `pre-commit` in the container. If you open a fresh clone in the container first, run `just setup` on the host before committing there.
- `core.fileMode` is `false` inside the container only, because Docker Desktop bind mounts report every file as executable. To commit a new executable from the container, use `git update-index --chmod=+x <file>`.

## Commands

| Recipe                 | What it does                                                                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `just setup`           | Recreate `.venv/`, install the dependencies and the pre-commit hooks                                |
| `just format`          | Run every pre-commit hook on all files (ruff, mdformat, codespell, cargo fmt, cargo clippy, ...)    |
| `just test`            | `just test-rs`, then `just test-py`                                                                 |
| `just test-rs`         | `cargo test --workspace`                                                                            |
| `just test-py`         | pytest on all tests, including slow ones, with coverage                                             |
| `just test-fast`       | `cargo test --workspace`, then pytest without the slow tests. Use it while you iterate              |
| `just type`            | pyright, strict on `src/`                                                                           |
| `just run`             | `format`, `test`, `type` and `docs-build`, in this order                                            |
| `just docs`            | Serve the documentation at <http://127.0.0.1:8000> with live reload. Arguments go to `mkdocs serve` |
| `just docs-build`      | Build the documentation into `site/`; any warning (a broken link, a missing API object) fails       |
| `just stub`            | Generate `src/mujou/_core/_native.pyi` from the Rust signatures and doc comments                    |
| `just clean`           | Remove `dist/`, `build/`, `site/`, `htmlcov/`, `.coverage`, `__pycache__/` and `.pytest_cache/`     |
| `just container <cmd>` | Run `<cmd>` in the dev container, in this checkout                                                  |

One test file: `uv run pytest tests/mujou/test_time.py -v`. By keyword: `uv run pytest -k "<expr>"`.

**`just run` must pass before you commit or open a pull request.**

## Project layout

| Path                   | Contents                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `src/mujou/`           | The Python package. Public names are in `mujou.*`; the implementation is in the private package `mujou._core`       |
| `crates/mujou-core`    | The core in pure Rust                                                                                               |
| `crates/mujou-python`  | The PyO3 bindings, built as `mujou._core._native`                                                                   |
| `crates/mujou-stubgen` | Writes the type stub `_native.pyi` from the built extension (used by `just stub`)                                   |
| `tests/`               | pytest tests; `tests/mujou/` mirrors `src/mujou/`                                                                   |
| `docs/`                | The user documentation (MkDocs), in English (`docs/en/`) and Japanese (`docs/ja/`)                                  |
| `specs/`               | Design specifications for developers, in Japanese. They record design decisions, including ones not implemented yet |
| `examples/`            | Programs for users. The tests run every one of them                                                                 |

`src/mujou/_core/_native.pyi` is generated and committed. After changing the bindings, run `just stub` and commit the result. A pre-commit hook runs `just stub` when the Rust sources, `pyproject.toml` or the stub change, and fails when the stub was stale; the Test workflow in CI checks it too.

## Design rules

- Keep nodes local and asynchronous. A node updates from its own state and the messages of its neighbours only. Do not build an autograd graph across nodes (no `.backward()` across nodes), and do not add global barriers, epochs, or train/inference modes.
- Autograd inside one node is fine, written as in plain PyTorch: the hooks record gradients, and a port sends a detached copy. In `src/mujou/`, mark such a line with `# mujou: local-autograd`.
- Do not build a learning rule into nodes, messages or the scheduler. The update rule must stay replaceable.
- Prefix internals, including attributes set in `__init__`, with `_`. Make public only what users need.
- Subclasses of `Node` and `Network` call `super().__init__()` first in `__init__`, and the same hook of `super()` first in every hook they override (`on_start`, `on_update(dt)`, `on_stop`, ...). Write examples and docs this way too.
- pyright is strict on `src/`; ruff checks and formats with a line length of 88.
- Write docstrings and comments in English. Docstrings use the Google style.
- Add dependencies with `uv add` (or `uv add --dev`) and commit `uv.lock`. Do not use `pip install`.

## Tests

- `tests/mujou/` mirrors `src/mujou/`. Group the tests of one thing in a `class TestXxx`.
- Prefer oracles derived from theory, such as finite differences and known equilibria, to checks that "the loss went down".
- Keep tests deterministic. Run on a `VirtualTimer` and step the nodes by hand with `mujou.testing.run` and `ManualRun.step`, and fix the seeds (a node's `seed`).
- Mark long-running convergence or training checks with `@pytest.mark.slow`. Markers are strict: register new ones under `markers` in `pyproject.toml`.
- `tests/test_examples.py` runs every program in `examples/`. The rules for examples are in [examples/README.md](examples/README.md).

## Documentation

- Change `docs/en/` and `docs/ja/` together, with the same page structure.
- The API reference (`docs/en/api/`) is generated from the docstrings with mkdocstrings and exists in English only; the Japanese site falls back to it. When you change the public API, update the `:::` entries there and the guide pages.
- Include code from `examples/` with the snippets syntax (`--8<-- "examples/<file>.py"`) instead of copying it.
- `just docs-build` is strict: a broken link or a missing API object fails the build.

## Git and pull requests

- Branch from `main` as `<type>/<YYYY-MM-DD>/<topic>`, for example `feature/2026-10-11/pc-alm-layer`. Types: `feature`, `fix`, `refactor`, `docs`, `chore`.
- Write commit messages in English as `<type>(<scope>): <summary>`, one concern per commit.
- Do not commit to `main` directly, force-push, or skip hooks with `--no-verify`.
- Before opening a pull request, merge the latest `main` into your branch and run `just run`.

CI runs these workflows on pull requests to `main`, except GPU. Test, Type Check and Docs run only when files they depend on change.

| Workflow      | Checks                                                                                                                                                                                                                                                            |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Test          | `just test-py` on Ubuntu and macOS, with Python 3.14t and 3.14, and that the committed `_native.pyi` matches `just stub`; `just test-rs` and `cargo doc` (warnings are errors) on Ubuntu and macOS; `cargo check` with the oldest supported Rust (`rust-version`) |
| Type Check    | `just type` (pyright)                                                                                                                                                                                                                                             |
| Format & Lint | All pre-commit hooks except `mujou-stub`                                                                                                                                                                                                                          |
| Docs          | `just docs-build`                                                                                                                                                                                                                                                 |
| GPU           | The tests marked `cuda`, on a self-hosted runner with a GPU. Started by hand only ([.github/runner/README.md](.github/runner/README.md))                                                                                                                          |

`CLAUDE.md`, `.claude/` and `memory/` configure the AI coding agents used in this repository; human contributors can ignore them.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
