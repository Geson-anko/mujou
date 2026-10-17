#!/usr/bin/env bash
# Register this Linux machine with an NVIDIA GPU as a self-hosted GitHub Actions
# runner of Geson-anko/mujou, with the label `gpu`, for the GPU workflow
# (.github/workflows/gpu.yml). See .github/runner/README.md.
#
# Usage: RUNNER_TOKEN=<registration token> .github/runner/setup-gpu-runner.sh [--service]
#
#   RUNNER_TOKEN    Required. From the repository's Settings > Actions > Runners >
#                   New self-hosted runner (valid for one hour).
#   RUNNER_DIR      Where to install the runner (default: ~/actions-runner-mujou).
#   RUNNER_NAME     The runner's name (default: <hostname>-gpu).
#   RUNNER_VERSION  The version of actions/runner (default: the latest release).
#   --service       Also install and start the runner as a systemd service (sudo).
set -euo pipefail

repo_url="https://github.com/Geson-anko/mujou"
dir="${RUNNER_DIR:-$HOME/actions-runner-mujou}"
name="${RUNNER_NAME:-$(hostname)-gpu}"
service=false
for arg in "$@"; do
    case "$arg" in
        --service) service=true ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ -z "${RUNNER_TOKEN:-}" ]; then
    echo "set RUNNER_TOKEN to a registration token (Settings > Actions > Runners)" >&2
    exit 2
fi
if [ "$(uname -s)" != "Linux" ]; then
    echo "the GPU runner must be Linux, got $(uname -s)" >&2
    exit 1
fi
case "$(uname -m)" in
    x86_64) arch=x64 ;;
    aarch64) arch=arm64 ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null; then
    echo "nvidia-smi does not work: install the NVIDIA driver first" >&2
    exit 1
fi
for tool in curl tar git; do
    command -v "$tool" >/dev/null || { echo "$tool is missing" >&2; exit 1; }
done

version="${RUNNER_VERSION:-$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest | sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p')}"
if [ -z "$version" ]; then
    echo "could not find the latest runner version; set RUNNER_VERSION" >&2
    exit 1
fi

mkdir -p "$dir"
cd "$dir"
if [ ! -x ./config.sh ]; then
    archive="actions-runner-linux-${arch}-${version}.tar.gz"
    curl -fL -o "$archive" "https://github.com/actions/runner/releases/download/v${version}/${archive}"
    tar xzf "$archive"
    rm "$archive"
fi

./config.sh --unattended --replace \
    --url "$repo_url" --token "$RUNNER_TOKEN" \
    --name "$name" --labels gpu --work _work

if "$service"; then
    sudo ./svc.sh install
    sudo ./svc.sh start
    echo "the runner $name runs as a service"
else
    echo "the runner $name is configured; start it with: $dir/run.sh"
fi
