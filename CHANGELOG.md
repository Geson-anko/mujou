# Changelog

All notable changes to Mujou are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Mujou follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Until 1.0.0, a minor release (0.x.0) may break the API; a patch release (0.x.y) only fixes bugs.

<!-- --8<-- [start:releases] -->

## [Unreleased](https://github.com/Geson-anko/mujou/compare/v0.1.0...HEAD)

## [0.1.0](https://github.com/Geson-anko/mujou/releases/tag/v0.1.0) - 2026-10-23

The first release. The API works but will still change in breaking ways in the 0.x releases.

### Added

- Nodes, written in PyTorch: `mujou.Node` with the lifecycle hooks `on_start`, `on_update(dt)`, `on_pause`, `on_resume`, `on_exception` and `on_stop`, and a per-node period (`Node(period=...)`).
- Messages between nodes: `Port` and `MultiPort`, connected by `Channel`, with the message specs `TensorSpec`, `DictSpec`, `IntSpec`, `FloatSpec`, `BoolSpec` and `TimestampSpec`. A node reads the latest value its neighbour sent and never waits for it.
- Networks: `Network` to connect nodes and `Group` to put them side by side.
- Runs: `mujou.run(root)` runs a network in the background on a pool of worker threads, written in Rust. The handle `Run` writes inputs, reads outputs, waits for a condition and subscribes to a port from the host program.
- Time: one timer per run (`mujou.time`), which can be paused, and a virtual timer for tests.
- State: `Stateful`, and `state_dict` / `load_state_dict` of a run to save it and resume it later.
- Errors: an exception in any hook stops the whole run and is raised in the host program as `NetworkError`.
- Testing: `mujou.testing.run` and `ManualRun.step` to step nodes by hand, deterministically.
- PC-ALM (Augmented Lagrangian Predictive Coding): the nodes of `mujou.pc` (`Input`, `Layer`, `Output`, `Chain`), the losses `MSELoss` and `CrossEntropyLoss`, and the optimizers `SGD` and `AdamW` that advance with time.
- Visualization: `mujou.viz.to_html` draws a network as a self-contained HTML page.
- Wheels for Python 3.14 and the free-threaded 3.14t on Linux (x86_64, aarch64), macOS (arm64) and Windows (x86_64).

<!-- --8<-- [end:releases] -->
