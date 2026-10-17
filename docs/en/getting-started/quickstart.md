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

If you built from source, run it from the root of the repository.

```sh
uv run python examples/quickstart.py
```

It prints one line like this. The value of `t` changes from run to run.

```text
y = tensor([3., 5., 7.]) at t = 0.017 s
```

## Define the nodes

`Scale` and `Shift` are nodes. You write a node as a class that inherits from `mujou.Node`.

| Code                                | Meaning                                                                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VECTOR = TensorSpec((3,))`         | Declares the shape and dtype of the messages: here, a float32 vector of length 3                                                                      |
| `self.prev = PrevPort(recv=VECTOR)` | The port on the previous side. It receives `VECTOR` messages                                                                                          |
| `self.next = NextPort(send=VECTOR)` | The port on the next side. It sends `VECTOR` messages                                                                                                 |
| `def on_update(self, dt)`           | The method Mujou calls over and over. Write one step of work in it. `dt` is the time in seconds since the previous call; this program does not use it |
| `x = self.prev.recv()`              | Reads the latest value from the previous side. It does not wait for a new value. If nothing has arrived yet, it returns a tensor of zeros             |
| `self.next.send(self.factor * x)`   | Sends the result to the next side                                                                                                                     |

Declare ports in `__init__`. Mujou runs the loop, so do not write a `while` loop inside a node. Instead of waiting for a new value, read the latest value and compute on every `on_update`. To tell whether a value is new, read the value and the time it was sent with `recv_with_time()`, and compare that time with the one you read last.

## Connect the nodes

`Affine` is a network. A network inherits from `mujou.Network` and holds its nodes as attributes.

| Code                             | Meaning                                                |
| -------------------------------- | ------------------------------------------------------ |
| `self.scale.connect(self.shift)` | Connects `next` of `scale` to `prev` of `shift`        |
| `self.prev = self.scale.prev`    | Exposes `prev` of `scale` as the input of the network  |
| `self.next = self.shift.next`    | Exposes `next` of `shift` as the output of the network |

## Run it, write inputs, read outputs

The host part runs from `with mujou.run(net) as run:` to the final `assert`.

1. `mujou.run(net)` runs the network. Each node repeats `on_update` on a thread of its own. Once the nodes are running, `mujou.run` returns and the host program moves on
2. `run.write({net.prev: x})` writes `x` to `prev` of `scale`, an input port that is not connected to anything. The written value stays on that port until you write again
3. `run.read(net.next)` returns the pair `(y, t)`: the last value that `next` of `shift` sent, and the time it was sent
4. Until `y` is $2x + 1$, the host waits 0.01 seconds of global time with `mujou.time.sleep(0.01)` and reads again. If `y` is still not $2x + 1$ after 5 seconds of global time, the host stops reading again, and the final `assert` fails
5. Leaving the `with` block stops the network

To wait, use `mujou.time.sleep`, not the standard `time.sleep`. `mujou.time.sleep` waits on Mujou's global time, so it still waits correctly when you change the speed of time or when the run is paused (see [Global Time](../concepts/time.md)).

## Read the output

```text
y = tensor([3., 5., 7.]) at t = 0.017 s
```

| Part                       | Meaning                                                                                                                                                                                                               |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `y = tensor([3., 5., 7.])` | $2x + 1$ for `x = [1, 2, 3]`. The final `assert` checks it                                                                                                                                                            |
| `t = 0.017 s`              | The global time, in seconds, at which `shift` sent `y`. It is not the time the host read `y`. Global time is 0 at the moment the process first uses it. You can take that moment to be about when `import mujou` runs |

Step 4 reads again because `y` may not be `[3., 5., 7.]` right after `write`. It takes a few updates for `x` to pass through `scale` and `shift`. Until it arrives, `shift` still sends a value computed from the previous input (zeros at first). How many updates fit in a given time depends on how the threads get scheduled. So the host does not wait for a fixed time. It reads again until the value arrives.

`t` varies from run to run. The nodes run without waiting for each other, so the update at which the value arrives differs between runs.

## When a node raises an exception

If any node raises an exception, the whole network stops. Any later call to `write` or `read` raises `mujou.NetworkError` in the host. The `__cause__` of the `NetworkError` is the original exception that the node raised (see [Errors and Shutdown](../user-guide/errors.md)).

## Update period

The nodes in this program do not call `super().__init__()`, so they have no update period. A node without a period repeats `on_update` without waiting and uses up a whole CPU core.

To update at a regular interval, call `super().__init__(period=seconds)` at the start of the node's `__init__`.

```python
class Scale(Node):
    def __init__(self, factor: float) -> None:
        super().__init__(period=0.01)  # on_update runs once every 0.01 s on average
        ...
```

Even with a period, `dt` is the time that actually passed since the previous call. It is close to the period, but longer when the node falls behind (see [Writing a Node](../user-guide/writing-a-node.md)).

## Next steps

- How to think about nodes, ports, networks, and global time: [Concepts](../concepts/index.md)
- The other methods of the `Run` that `mujou.run` returns: [Running a Network](../user-guide/running.md)
- More programs: [Examples](../examples/index.md)
