FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# uv manages Python (3.14t from .python-version) and the virtualenv.
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/

# just >= 1.58 is required for the `[windows] set shell` syntax in the justfile.
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
    | bash -s -- --tag 1.58.0 --to /usr/local/bin

# Rust toolchain for the native core (mujou._core).
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --component rustfmt,clippy
ENV PATH="/home/vscode/.cargo/bin:${PATH}"
