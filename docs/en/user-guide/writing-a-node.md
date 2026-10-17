# Writing a Node

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

As in any subclass, `__init__` first calls `super().__init__()`, and each overridden hook first calls the same hook of `super()`. The hooks of `mujou.Node` do nothing, except `state_dict` and `load_state_dict` (see [State](state.md)). Call them anyway: then your node keeps working when it later inherits from a node class of your own whose hooks do something.

## 1. Declare the ports in `__init__`

Assign ports as instance attributes inside `__init__`. A port declared as a class attribute makes building the network (`build()`) raise `ValueError`.

Choose the kind of port by its role in the network.

| Role                                                                   | Declaration |
| ---------------------------------------------------------------------- | ----------- |
| The side that receives from the previous node (`b` in `a.connect(b)`)  | `PrevPort`  |
| The side that sends to the next node (`a` in `a.connect(b)`)           | `NextPort`  |
| Anything else (a teacher signal input, an output for observation, ...) | `Port`      |

A node can have at most one `PrevPort` and at most one `NextPort`. It can have any number of `Port`s.

Give each port the type of the values it sends as `send=`, and the type of the values it receives as `recv=`. Write a type as `TensorSpec(shape, dtype)`. If you leave out the dtype, it is `torch.get_default_dtype()` at the time the spec is created (usually `float32`). On a one-way port, leave out the argument you do not use (do not pass `None`).

```python
self.prev = PrevPort(recv=TensorSpec((32, 784)))                       # receive only
self.next = NextPort(send=TensorSpec((32, 256)), recv=TensorSpec((32, 256)))  # both ways
```

`multi=` decides the number of peers. By default, `PrevPort` and `Port` take one peer and `NextPort` takes many. On a port with many peers, `recv()` returns a dict, not the value itself (see [Branching and merging](building-a-network.md#branching-and-merging) in Building a network).

A value you send must match the spec in both shape and dtype. The dtype is not converted automatically. The `Leaky` above uses a `torch.float64` spec, so it also creates the values it sends as `float64`.

Until a port first sends, its peers receive zeros, and so does the host that reads it. To start from another value, give the port an initial value with `initial=`. It must match `send` in shape and dtype, and it is copied when the port is made. A restored run (see [State](state.md)) starts from the values in the state instead.

```python
self.next = NextPort(send=TensorSpec((32, 256)), initial=torch.full((32, 256), 0.5))
```

Values on two ports travel independently: a reader may see a new value on one port with an old value on the other. Send values that belong together (an input and its target, say) as one message: describe it with a `TypedDict` and a `DictSpec`, which gives a `TensorSpec` for each key. The whole dict arrives at once, and type checkers know the type of each key.

```python
class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((32, 4)), y=TensorSpec((32, 2)))

self.next = NextPort(send=SAMPLE)  # in the sender: self.next.send({"x": x, "y": y})
self.prev = PrevPort(recv=SAMPLE)  # in the receiver: sample = self.prev.recv(); sample["x"]
```

Type checkers infer the type of a port from its specs, so you seldom write it. To annotate one (an argument of a helper, say), write `Port[S, R, M]`: `S` is the type it sends, `R` the type it receives, and `M` its number of peers, `Single` or `Multi` from `mujou.typing`. An omitted direction is `Never`. A function that takes any spec takes a `MessageSpec[T]` from `mujou.typing`.

```python
from typing import Never

from mujou.typing import MessageSpec, Multi


def mean_of(port: PrevPort[Never, torch.Tensor, Multi]) -> torch.Tensor:
    return torch.stack(list(port.recv().values())).mean(0)


def sender[T](spec: MessageSpec[T]) -> NextPort[T, Never, Multi]:
    return NextPort(send=spec)
```

## 2. Set up the state in `on_start`

Create the state tensors in `on_start`. In `on_update`, do not assign a new tensor to the attribute; update the tensor in place with `+=`, `copy_` and the like.

Assigning a new tensor raises no error, and the new tensor goes into the state under the same name. What breaks is any other reference to the tensor created in `on_start` (from an optimizer, another attribute, ...): it still points at the old tensor, and later updates do not reach it. Updating in place keeps every reference on the one tensor. Restoring a state also writes in place into that tensor.

`torch.Tensor` and `nn.Module` values assigned to attributes are collected into the state automatically. What is collected, and how to save it, is in [Saving and restoring state](state.md).

When the node computes on a GPU, create the tensors with `device=self.device`.

```python
@override
def on_start(self) -> None:
    self.w = torch.zeros(256, 256, device=self.device)
```

You may keep settings (a time constant, the size of a layer, ...) as numbers set in `__init__`. Numbers are not collected into the state.

An `nn.Module` may be built in `__init__` or in `on_start`; it is collected either way. Initialize anything random from `self.generator`, which exists from `on_start` on, so a module initialized that way is built in `on_start`.

## 3. Update in proportion to `dt` in `on_update(dt)`

Mujou calls `on_update(dt)` again and again. The node's author does not write a loop; write the work of one call only. One call does the following, in this order.

1. Read the neighbors' latest values with `recv()`. It never waits for a new value to arrive
2. Update the state by an amount proportional to `dt`
3. Send the result with `send()`

`dt` is the time in seconds since this node's previous update. The first `dt` counts from when the run's timer starts, after every node's `on_start` (in a restored run, from the node's last update before the state was taken). For a differential equation $\frac{dx}{dt} = f(x)$, write `x += dt * f(x)` (explicit Euler). If the update size is not proportional to `dt`, a node that loops faster changes faster (see [Global time](../concepts.md#global-time)).

`dt` is not constant. It gets longer when the machine is busy. Write `on_update` so that it does not diverge even with the longest `dt` that can come. Do not cut `dt` itself. The time you cut off is lost, and the result drifts from the continuous-time solution. There are two ways to update stably with a long `dt`.

- If the system has an exact solution, advance the state by `dt` with that solution. The `Leaky` above does not use explicit Euler. It moves $x$ by the fraction `1 - exp(-dt / tau)` of its distance to $u$. This fraction is exact while the input stays constant. It is below 1 for any `dt`, so $x$ never overshoots $u$
- If the system has no exact solution, split the step inside one call of `on_update`. Choose `n` so that one step `dt / n` is no longer than a length that updates stably, and update `n` times in a `for` loop (see the example in [PyTorch in nodes](torch.md))

`recv()` always returns a value, and `on_update` usually runs more often than the peer sends, so the same message is read many times. That is right for a value that holds over time (a sensor reading, an activity): update toward it by `dt` each time. For noise in a continuous-time update, scale its standard deviation by $\sqrt{dt}$, not by `dt`, so that the result does not depend on how often the node updates.

When a message is an event to handle once (a training sample, say), read its time with `recv_with_time()` and skip the update while the time has not changed. Reading the time in `on_start` skips the value that is there before the peer's first send: the port's initial value (zeros unless the sender declares `initial=`), or the value of a restored state. A restored value was usually handled before the state was taken, but one the peer sent after the node's last update was not: a restored run may skip that one message. (A port the host writes gets its `inputs=` value at that same time, so this skips it too.)

```python
@override
def on_start(self) -> None:
    super().on_start()
    _, self.last = self.prev.recv_with_time()  # the time of the value there now


@override
def on_update(self, dt: float) -> None:
    super().on_update(dt)
    sample, t = self.prev.recv_with_time()
    if t == self.last:  # nothing new since the last update
        return
    self.last = t
    ...  # learn from the sample once
```

## Set the period

By default, a node's next `on_update` is called as soon as the previous one returns, so the node loops as fast as it can. One such node uses up one CPU core. To match the period of a sensor, or to leave the CPU free, pass a period `period` in seconds.

```python
class Sensor(mujou.Node):
    def __init__(self) -> None:
        super().__init__(period=0.01)  # update once every 10 ms on average
        self.next = mujou.NextPort(send=mujou.TensorSpec((3,)))
```

The period is the average interval between updates. The time `on_update` takes counts as part of the period. A node that falls behind does not update several times in a row to catch up.

The wait is measured on the run's timer. If you change the time scale, the period changes by the same factor against real time.

Even with a period, `dt` is not exactly `period`. `dt` is the time that actually passed: after a delay it is longer than `period`, and the next one is shorter. Even with a period, update in proportion to `dt`.

In a pipeline fed by a node with a period (a sensor, say), the nodes after it read the same input until the next one arrives. Without a period of their own they repeat the same work and each uses up a core; give them a period too, no longer than the period of their input.

Pass `period` only as an argument of `super().__init__`. Setting a default with a class attribute `period = 0.01` makes the start of the run raise `TypeError`. A manual run (see [Testing](testing.md)) does not use the period.

## Set the seed and the device

`super().__init__` takes the following arguments. All of them are keyword arguments and can be left out.

| Argument | Meaning                                                                                         | Default                       |
| -------- | ----------------------------------------------------------------------------------------------- | ----------------------------- |
| `device` | The device the node computes on. `"cpu"`, a CUDA device, or another accelerator such as `"mps"` | `"cpu"`                       |
| `seed`   | The seed of the node's own random number generator `self.generator`                             | `None` (different every time) |
| `period` | The period of updates, in seconds                                                               | `None` (as fast as it can)    |

Draw random numbers from `self.generator` (`torch.randn(..., generator=self.generator)`). Do not call `torch.manual_seed` in a node, because it changes the random number generator that all nodes share. `self.generator` can be used in the hooks from `on_start` on. How it works with PyTorch is in [PyTorch in nodes](torch.md).

`device`, `generator`, `period` and `timer` are reserved names. Do not assign attributes with these names.

## The hooks and their order

Mujou calls the hooks of a node in this order. Override only the ones you need.

1. `on_thread_start()`: once, before `on_start`. Per-thread settings (see [PyTorch in nodes](torch.md))
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
