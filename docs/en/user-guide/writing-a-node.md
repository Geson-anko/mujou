# 🧩 Writing a Node

You write a node as a class that inherits from `mujou.Node`. This page shows how, step by step. For what a node is, see [How Mujou works](../concepts.md#nodes).

## Example: a leaky integrator

This node follows its input $u$ with the time constant $\tau$. Its state $x$ follows $\frac{dx}{dt} = \frac{u - x}{\tau}$.

```python
--8<-- "examples/leaky_chain/leaky.py"
```

The node writes three things.

1. In `__init__`, it declares the ports (`prev` and `next`)
2. In `on_start`, it sets up the state `x`
3. In `on_update(dt)`, it reads the input, updates the state in proportion to `dt`, and sends the result

The sections below follow this order.

!!! tip "Always call `super()` first"

    As in any subclass, `__init__` first calls `super().__init__()`, and each overridden hook first calls the same hook of `super()`. The hooks of `mujou.Node` do nothing, except `state_dict` and `load_state_dict` (see [State](state.md)).

    Call them anyway: then your node keeps working when it later inherits from a node class of your own whose hooks do something.

## 1. Declare the ports in `__init__`

Assign ports as instance attributes inside `__init__`, after `super().__init__()`.

- **Before `super().__init__()`**: the node registers each port as it is assigned, and assigning one before `super().__init__()` raises `AttributeError`
- **As a class attribute**: a port declared as a class attribute makes building the network (`build()`) raise `ValueError`

Choose the kind of port by its role in the network. By convention, name the port on the side of the previous node `prev` and the one on the side of the next node `next`, so that a chain reads `a.next.connect(b.prev)`.

| Role                                                                   | Declaration | Name by convention |
| ---------------------------------------------------------------------- | ----------- | ------------------ |
| The side that receives from the previous node                          | `PrevPort`  | `prev`             |
| The side that sends to the next node                                   | `NextPort`  | `next`             |
| Anything else (a teacher signal input, an output for observation, ...) | `Port`      | any                |

The kinds differ only in the number of peers they take by default (below). A node can have any number of ports of each kind.

Give each port the type of the values it sends as `send=`, and the type of the values it receives as `recv=`. Write a type as `TensorSpec(shape, dtype)`.

If you leave out the dtype, it is `torch.get_default_dtype()` at the time the spec is created (usually `float32`). On a one-way port, leave out the argument you do not use (do not pass `None`).

```python
self.prev = PrevPort(recv=TensorSpec((32, 784)))                       # receive only
self.next = NextPort(send=TensorSpec((32, 256)), recv=TensorSpec((32, 256)))  # both ways
```

`multi=` decides the number of peers. By default, `PrevPort` and `Port` take one peer and `NextPort` takes many. On a port with many peers, `recv()` returns a dict, not the value itself (see [Branching and merging](building-a-network.md#branching-and-merging) in Building a network).

A value you send must match the spec in both shape and dtype. The dtype is not converted automatically. The `Leaky` above uses a `torch.float64` spec, so it also creates the values it sends as `float64`.

Until a port first sends, its peers receive zeros, and so does the host that reads it. To start from another value, give the port an initial value with `initial=`. It must match `send` in shape and dtype, and it is copied when the port is made.

A restored run (see [State](state.md)) starts from the values in the state instead.

```python
self.next = NextPort(send=TensorSpec((32, 256)), initial=torch.full((32, 256), 0.5))
```

Values on two ports travel independently: a reader may see a new value on one port with an old value on the other.

Send values that belong together (an input and its target, say) as one message: describe it with a `TypedDict` and a `DictSpec`, which gives a `TensorSpec` for each key. The whole dict arrives at once, and type checkers know the type of each key.

```python
class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((32, 4)), y=TensorSpec((32, 2)))

self.next = NextPort(send=SAMPLE)  # in the sender: self.next.send({"x": x, "y": y})
self.prev = PrevPort(recv=SAMPLE)  # in the receiver: sample = self.prev.recv(); sample["x"]
```

Type checkers infer the type of a port from its specs, so you seldom write it. To annotate one (an argument of a helper, say), write `Port[S, R, M]`.

- `S` is the type it sends
- `R` the type it receives
- `M` its number of peers, `Single` or `Multi` from `mujou.typing`

An omitted direction is `Never`. A function that takes any spec takes a `MessageSpec[T]` from `mujou.typing`.

```python
from typing import Never

from mujou.typing import MessageSpec, Multi


def mean_of(port: PrevPort[Never, torch.Tensor, Multi]) -> torch.Tensor:
    return torch.stack(list(port.recv().values())).mean(0)


def sender[T](spec: MessageSpec[T]) -> NextPort[T, Never, Multi]:
    return NextPort(send=spec)
```

## 2. Set up the state in `on_start`

Create the state tensors in `on_start`.

!!! warning "Update the state in place"

    In `on_update`, do not assign a new tensor to the attribute; update the tensor in place with `+=`, `copy_` and the like.

    Assigning a new tensor raises no error, and the new tensor goes into the state under the same name. What breaks is any other reference to the tensor created in `on_start` (from an optimizer, another attribute, ...): it still points at the old tensor, and later updates do not reach it.

    Updating in place keeps every reference on the one tensor. Restoring a state also writes in place into that tensor.

`torch.Tensor` and `nn.Module` values assigned to attributes are collected into the state automatically. What is collected, and how to save it, is in [Saving and restoring state](state.md).

When the node computes on a GPU, create the tensors with `device=self.device`.

```python
@override
def on_start(self) -> None:
    self.w = torch.zeros(256, 256, device=self.device)
```

You may keep settings (a time constant, the size of a layer, ...) as numbers set in `__init__`. Numbers are not collected into the state.

An `nn.Module` may be built in `__init__` or in `on_start`. The default initialization of a module such as `nn.Linear` draws from the global generator that all nodes share, so `seed` does not reproduce it: initialize it again from `self.generator`, which can be used in `__init__` already.

## 3. Update in proportion to `dt` in `on_update(dt)`

Mujou calls `on_update(dt)` again and again. The node's author does not write a loop; write the work of one call only. One call does the following, in this order.

1. Read the neighbors' latest values with `recv()`. It never waits for a new value to arrive
2. Update the state by an amount proportional to `dt`
3. Send the result with `send()`

`dt` is the time in seconds since this node's previous update. The first `dt` counts from when the run's timer starts, after every node's `on_start` (in a restored run, from the node's last update before the state was taken).

For a differential equation $\frac{dx}{dt} = f(x)$, write `x += dt * f(x)` (explicit Euler). If the update size is not proportional to `dt`, a node that loops faster changes faster (see [Global time](../concepts.md#global-time)).

`dt` is not constant. It gets longer when the machine is busy. Write `on_update` so that it does not diverge even with the longest `dt` that can come.

!!! warning "Do not cut `dt`"

    Do not cut `dt` itself. The time you cut off is lost, and the result drifts from the continuous-time solution.

There are two ways to update stably with a long `dt`.

- **Exact solution**: if the system has an exact solution, advance the state by `dt` with that solution. The `Leaky` above does not use explicit Euler. It moves $x$ by the fraction `1 - exp(-dt / tau)` of its distance to $u$. This fraction is exact while the input stays constant. It is below 1 for any `dt`, so $x$ never overshoots $u$
- **Split the step**: if the system has no exact solution, split the step inside one call of `on_update`. Choose `n` so that one step `dt / n` is no longer than a length that updates stably, and update `n` times in a `for` loop (see the example in [PyTorch in nodes](torch.md))

`recv()` always returns a value, and `on_update` usually runs more often than the peer sends, so the same message is read many times. That is right for a value that holds over time (a sensor reading, an activity): update toward it by `dt` each time.

For noise in a continuous-time update, scale its standard deviation by $\sqrt{dt}$, not by `dt`, so that the result does not depend on how often the node updates.

When a message is an event to handle once (a training sample, say), read its time with `recv_with_time()` and skip the update while the time has not changed.

Until the peer first sends, `recv()` returns the port's initial value (zeros unless the sender declares `initial=`), and `has_received()` returns `False`: skip that value too. For a port the host writes, the `inputs=` value and each `run.write` count as sent.

In a restored run, `has_received()` returns `True` for a value that had been sent when the state was taken. `self.last` starts again from `None`, so the node handles that value once more (usually it was handled before the state was taken).

```python
@override
def on_start(self) -> None:
    super().on_start()
    self.last: mujou.time.Timestamp | None = None  # the time of the last one handled


@override
def on_update(self, dt: float) -> None:
    super().on_update(dt)
    if not self.prev.has_received():  # the initial value: the peer has not sent yet
        return
    sample, t = self.prev.recv_with_time()
    if t == self.last:  # nothing new since the last update
        return
    self.last = t
    ...  # learn from the sample once
```

## Set the period

A node updates once every `period` seconds on average, 0.01 s (100 times a second) by default. To match the period of a sensor, or to respond faster or slower, pass a period `period` in seconds.

```python
class Sensor(mujou.Node):
    def __init__(self) -> None:
        super().__init__(period=0.05)  # update once every 50 ms on average
        self.next = mujou.NextPort(send=mujou.TensorSpec((3,)))
```

The period is the average interval between updates. The time `on_update` takes counts as part of the period. A node that falls behind does not update several times in a row to catch up.

The wait is measured on the run's timer. If you change the time scale, the period changes by the same factor against real time.

Even with a period, `dt` is not exactly `period`. `dt` is the time that actually passed: after a delay it is longer than `period`, and the next one is shorter. Even with a period, update in proportion to `dt`.

In a pipeline fed by a node with a period (a sensor, say), the nodes after it read the same input until the next one arrives. Give them a period no longer than the period of their input.

!!! warning "`period=0.0` uses up a CPU core"

    `period=0.0` calls the next `on_update` as soon as any time has passed. Such a node uses up one CPU core, takes time from the other nodes and makes their `dt` irregular. Pass it only to a node that really must not wait.

Pass `period` only as an argument of `super().__init__`. Setting a default with a class attribute `period = 0.01` makes the start of the run raise `TypeError`.

A manual run (see [Testing](testing.md)) does not use the period.

## Set the seed and the device

`super().__init__` takes the following arguments. All of them are keyword arguments and can be left out.

| Argument           | Meaning                                                                                                     | Default                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `device`           | The device the node computes on. `"cpu"` or a CUDA device                                                   | `"cpu"`                       |
| `seed`             | The seed of the node's own random number generator `self.generator`                                         | `None` (different every time) |
| `period`           | The period of updates, in seconds                                                                           | `0.01`                        |
| `dedicated_thread` | Run the node on a thread of its own instead of the run's workers (see [below](#run-on-a-thread-of-its-own)) | `False`                       |

Draw random numbers from `self.generator` (`torch.randn(..., generator=self.generator)`). Do not call `torch.manual_seed` in a node, because it changes the random number generator that all nodes share.

`self.generator` can be used right after `super().__init__()`, in `__init__` too. How it works with PyTorch is in [PyTorch in nodes](torch.md).

`device`, `generator`, `period`, `dedicated_thread` and `timer` are reserved names. Do not assign attributes with these names.

## Run on a thread of its own

By default the node runs on the run's worker threads, which it shares with the other nodes (see [Running a network](running.md#threads-of-a-run)). Each hook may then run on a different thread, and the thread runs other nodes in between. So whether the next hook sees what a hook set per thread depends on the schedule, and the next node on the same worker may see it too:

- `threading.local`, `threading.current_thread()` and `threading.get_ident()` may differ from hook to hook
- Set a per-thread setting of torch, such as `torch.autocast`, back before the hook returns, by keeping it inside a `with` in the hook. A setting left behind leaks into the next node that runs on the same worker. Mujou sets back only gradient recording (before every hook) and, on CUDA, the current device and stream (whenever a worker switches to the node; see [PyTorch in nodes](torch.md))
- `contextvars` do carry over: each node runs its hooks in a `contextvars.Context` of its own. It starts empty, without the variables of the host's context, and `state_dict` runs outside it

Pass `dedicated_thread=True` to `super().__init__` when the node needs a thread of its own:

- Its hooks block, waiting for a device, a socket or a lock. On a worker, the wait would hold the worker, and the other nodes' `dt` would grow
- It relies on per-thread state set once, in `on_thread_start`

```python
class Camera(Node):
    def __init__(self) -> None:
        super().__init__(period=1 / 30, dedicated_thread=True)
```

`on_thread_start` is called only for such a node. A node on the workers that overrides it makes `mujou.run` raise `TypeError` before anything starts.

## The hooks and their order

Mujou calls the hooks of a node in this order. Override only the ones you need.

1. `on_thread_start()`: once, before `on_start`, only for a node with `dedicated_thread=True`. Per-thread settings (see [Run on a thread of its own](#run-on-a-thread-of-its-own))
2. `on_start()`: once. Create the state
3. `load_state_dict(state)`: once, only when the run is restored (see [Saving and restoring state](state.md))
4. `on_update(dt)`: again and again until the run ends
5. `on_pause()` and `on_resume()`: before and after the run's state is taken. `state_dict()` is called between them
6. `on_exception()`: once on every node, when any hook of any node raises (see [Errors and shutdown](errors.md))
7. `on_stop()`: once at the end, normal or not

## What not to do in a hook

| Do not                                                                                | Why, and what happens                                                                                                    |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Wait for another node's value to change (repeat `recv()` in a loop, ...)              | Nodes do not wait for each other. If a value has not changed, the node goes on with the same value                       |
| Call `mujou.run` or a method of `Run` (`write`, `read`, `state_dict`, ...)            | It raises `RuntimeError`. A node talks only through its own ports                                                        |
| Call an operation that moves the time, such as `mujou.time.sleep` or `set_time_scale` | It raises `RuntimeError`. Only the host moves the time                                                                   |
| Call `backward()` across nodes                                                        | A sent value is detached from the computation graph, so gradients do not pass through a port                             |
| Write a loop that never returns from `on_update`                                      | Pausing, ending and taking the state stop working. Mujou owns the loop. A finite `for` loop that splits the step is fine |

To know the time, read the time of the run's timer with `self.timer.now()`. Reading it is allowed in a hook.

If any hook raises, the whole network stops (see [Errors and shutdown](errors.md)).
