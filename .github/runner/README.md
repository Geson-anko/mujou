# GPU runner

The tests marked `cuda` run only on a machine with CUDA devices, so the regular CI skips them. The GPU workflow (`.github/workflows/gpu.yml`) runs them on a self-hosted runner with the label `gpu`. With no such runner online, the workflow waits in the queue.

## Set up a runner

On a Linux machine (x86_64 or aarch64) with an NVIDIA GPU:

1. Install the NVIDIA driver (`nvidia-smi` must work), `git`, `curl`, a C toolchain (`build-essential`), and Rust with [rustup](https://rustup.rs/) for the user who runs the runner. The workflow installs uv, Python and just itself.
2. In the repository's Settings > Actions > Runners, choose "New self-hosted runner" and copy the registration token (valid for one hour).
3. Run the script from a checkout of this repository:

```sh
RUNNER_TOKEN=<token> .github/runner/setup-gpu-runner.sh            # configure; start with run.sh
RUNNER_TOKEN=<token> .github/runner/setup-gpu-runner.sh --service  # also run as a systemd service
```

The script downloads the latest [actions/runner](https://github.com/actions/runner) into `~/actions-runner-mujou` (`RUNNER_DIR`) and registers it with the label `gpu` under the name `<hostname>-gpu` (`RUNNER_NAME`). Set `RUNNER_VERSION` to pin a version.

## Run the workflow

The GPU workflow runs on pushes to `main` and on pull requests to `main` that change code (the same paths as the Test workflow). It skips a pull request from a fork, whose code would otherwise run on the self-hosted machine. With one runner, a newer push to the same branch cancels the run before it. A maintainer can also start it by hand: in the Actions tab, choose "GPU" and "Run workflow", or run `gh workflow run gpu.yml --ref <branch>`.

The workflow installs torch from PyPI with `uv sync`; its Linux wheels are built with CUDA.
