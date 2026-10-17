# Quickstart

Run the smallest program and read its output. The program connects two nodes to compute $y = 2x + 1$.

```mermaid
flowchart LR
    H1["Host"] -->|"writes x"| S["scale (×2)"]
    S --> F["shift (+1)"]
    F -->|"reads y"| H2["Host"]
```

## The program

```python title="examples/quickstart.py"
--8<-- "examples/quickstart.py"
```

Run it from the root of the repository.

```sh
uv run python examples/quickstart.py
```

It prints one line like this.

```text
y = tensor([3., 5., 7.]) at t = 0.017 s
```

`y` is $2x + 1$ for `x = [1, 2, 3]`. `t` is the time, in seconds of Mujou's global time, at which `shift` sent `y`. It changes from run to run.

## What happened

- `Scale` and `Shift` are nodes: classes that inherit from `mujou.Node`. Each declares its ports in `__init__` and writes one step of work in `on_update(dt)`. As in any subclass, they call `super().__init__()` and `super().on_update(dt)` first
- `Affine` is a network: it holds the nodes as attributes, connects their ports with `connect`, and exposes the unconnected port `prev` of `scale` as its input and `next` of `shift` as its output
- `mujou.run(net)` starts the nodes, each repeating `on_update` on a thread of its own, and returns once they are running. In `on_update`, `recv()` reads the latest value without waiting, so `shift` keeps sending values computed from older inputs (zeros at first) until `x` has passed through `scale`
- So the output read right after `run.write` may still be old. `run.write` writes `x`, and `run.wait_for` waits until the output meets the condition, giving up with `TimeoutError` after 5 seconds of real time. Leaving the `with` block stops the network

## Next steps

- The ideas behind nodes, ports, networks, time, and state: [How Mujou works](../concepts.md)
- Your own nodes: [Writing a Node](../user-guide/writing-a-node.md). Running them, and the other methods of `Run`: [Running a Network](../user-guide/running.md)
- More programs: [Examples](../examples/index.md)
