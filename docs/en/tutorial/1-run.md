# 🚀 1. Run a Network

This page runs the smallest program, two nodes that compute $y = 2x + 1$, and explains it line by line. Then it branches and merges a network.

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

`y` is $2x + 1$ for `x = [1, 2, 3]`. `t` is the time at which `shift` sent `y`, in seconds of the run's timer. It changes from run to run.

## A node

```python
--8<-- "examples/quickstart.py:node"
```

- A node is a class that inherits from `mujou.Node`. As in any subclass, `__init__` calls `super().__init__()` first, and each hook calls the same hook of `super()` first
- `__init__` declares the node's ports as attributes. `Port(recv=VECTOR)` receives float32 vectors of 3, and `Port(send=VECTOR)` sends them. By convention, the port on the side of the previous node is `prev` and the one on the side of the next node is `next`
- `a` and `b` are settings: numbers set from the arguments, not state
- `on_update(dt)` is one step of the node's work. Mujou calls it again and again, about every 10 ms. `recv()` returns the latest value the peer sent, without waiting, and `send()` sends a copy to every peer

## A network

```python
--8<-- "examples/quickstart.py:network"
```

A network holds nodes as attributes, connects their ports with `connect`, and exposes the ports that the outside uses as attributes of its own (`self.prev`, `self.next`). `connect` checks that the spec each port sends matches the spec the other receives, and raises `TypeError` where it does not.

## The run

```python
--8<-- "examples/quickstart.py:run"
```

- `mujou.run(net)` checks the network, starts every node, and returns a `Run` once they are running. Each node repeats `on_update` on the run's worker threads; the host's thread goes on
- `run.write` writes `x` to the unconnected port `net.prev`. The value stays there until the next write
- `recv()` never waits, so `shift` keeps sending values computed from older inputs (zeros at first) until `x` has passed through `scale`. `run.wait_for` returns the first output that meets the condition, with its time, and raises `TimeoutError` after 5 seconds of real time
- Leaving the `with` block shuts the run down, and an exception of a node reaches the host there at the latest

## Branching and merging

A `Port` that only sends takes any number of peers: connect it to several ports, and each gets every value it sends (fan-out). A `MultiPort` merges: its `recv()` returns one value per channel, as a `dict[Channel, value]` in the order of `connect`, and the node decides how to combine them.

```python
--8<-- "examples/branching.py:fan-out"
```

`Diamond` computes $y = 2(x + 1) + (x + 1)^2$: `shift` sends to both branches, and `sum` adds what the two send. A `Port` that receives takes one peer only, because its `recv()` returns one value; receive from several with a `MultiPort`. Run it with `uv run python examples/branching.py`. The same program also sends a different value to each peer, which [Ports and Channels](../topics/ports.md#sending-to-one-channel) explains.

## Next

- Write a node whose state changes over time: [2. Write a Node](2-node.md)
- The host's other ways to write and read: [Host I/O](../topics/host-io.md)
