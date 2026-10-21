# 🔌 Ports and Channels

A port is a connection point of a node, and `connect` joins two ports with one channel. This page covers the kinds of ports, their specs, connecting, branching and merging, and what a port holds before anyone sends.

## The kinds of ports

| Kind        | Peers                                                   | `recv()` returns                             |
| ----------- | ------------------------------------------------------- | -------------------------------------------- |
| `Port`      | At most one if it receives; any number if it only sends | The value                                    |
| `MultiPort` | Any number, of one kind                                 | A `dict[Channel, value]`, in `connect` order |

Declare ports as attributes of the node in `__init__`, after `super().__init__()`; a run reads the attributes of the instance when it starts. A run raises `TypeError` for a port kept in a list, a tuple or a dict, and the class statement for a port declared as a class attribute: assign each port to an attribute of its own, or use a `MultiPort`.

Inputs of different roles, such as an input and its teacher signal, or a prediction and an error, go to separate `Port`s, each with a name that says its role. A `MultiPort` is for peers of one kind that the node combines the same way.

## Specs

A port takes the spec of what it sends as `send=` and of what it receives as `recv=`, as keyword arguments; leave out the one a one-way port does not use.

```python
self.prev = Port(recv=TensorSpec((32, 784)))  # receives only
self.next = MultiPort(send=TensorSpec((32, 256)), recv=TensorSpec((32, 256)))  # both ways
```

`TensorSpec(shape, dtype)` is one tensor; its dtype is `torch.float32` unless given, whatever `torch.get_default_dtype()` is. A value sent must match the spec in shape and dtype: no dtype is converted. The dtypes a spec takes are in `TensorSpec.dtypes`.

Values on two ports travel independently: a reader may see a new value on one port with an old one on the other. Send values that belong together, such as an input and its target, as one message: a `DictSpec` gives a `TensorSpec` for each key of a `TypedDict`, the whole dict arrives at once, and type checkers know the type of each key.

```python
class Sample(TypedDict):
    x: torch.Tensor
    y: torch.Tensor


SAMPLE = DictSpec(Sample, x=TensorSpec((32, 4)), y=TensorSpec((32, 2)))

self.next = Port(send=SAMPLE)  # in the sender: self.next.send({"x": x, "y": y})
self.prev = Port(recv=SAMPLE)  # in the receiver: sample = self.prev.recv(); sample["x"]
```

Type checkers infer the type of a port from its specs. To annotate one (an argument of a helper, say), write `Port[S, R]` or `MultiPort[S, R]`, where `S` is the type it sends and `R` the type it receives; a direction left out is `Never`, and a type checker rejects a `send` on a port that sends nothing. A function that takes any spec takes a `MessageSpec[T]`.

```python
from typing import Never

from mujou import MessageSpec, MultiPort, Port


def mean_of(port: MultiPort[Never, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(port.recv().values())).mean(0)


def sender[T](spec: MessageSpec[T]) -> Port[T]:
    return Port(send=spec)
```

## Connecting

`a.next.connect(b.prev)` joins two ports and returns the `Channel`. Nodes and networks have no `connect` of their own: a node may have several inputs and outputs, and naming the two ports says which path you mean.

- The spec that each port sends must match the spec the other receives, in both directions; otherwise `connect` raises `TypeError`
- A `Port` that receives takes one peer: connecting a second raises `ValueError`. Receive from several with a `MultiPort`
- Connecting the same two ports again returns the channel they have. A port may connect to a port of its own node
- Connections may form cycles: a node may receive from a node it sends to, as the nodes of `mujou.pc` do on two-way ports
- A port connects only to ports of nodes in the same run; `mujou.run` raises `ValueError` for a peer outside the tree

## Fan-out and merging

A `Port` that only sends takes any number of peers, and sends every value to each of them. A `MultiPort` merges: its `recv()` returns the latest value of each channel. The value of a peer that has never sent is the initial value of the peer's port.

```python
--8<-- "examples/branching.py:fan-out"
```

The receiving node decides how to combine the values: sum them, average them, weight each peer. To treat the peers differently, keep the `Channel` that `connect` returned for each, as an attribute of the receiving node (`self.sum.from_double = self.double.next.connect(self.sum.prev)`), and index with it: `self.prev.recv()[self.from_double]`.

## Sending to one channel

A `MultiPort` has two ways to send:

| How                       | Where it goes                            | What `Run.read` returns |
| ------------------------- | ---------------------------------------- | ----------------------- |
| `send(value)`             | The same value to every peer             | This value              |
| `send_to(channel, value)` | Only to the peer at the end of `channel` | Unchanged               |

```python
--8<-- "examples/branching.py:send-to"
```

The network hands the node the channels `connect` returned (`self.split.plus = ...`). Do not pick channels by their order in `self.next.channels` instead: swapping two `connect` calls would swap the destinations without any error. An attribute that holds a `Channel` is not part of the state.

The host reads with `Run.read` only what a port sent with `send` (its initial value until then); it cannot read the values of `send_to`, nor a port that only receives.

## Before anyone sends

Until a port first sends, its peers, and the host that reads it, get its initial value: zeros, or the value of `initial=`, which must match `send` and is copied when the port is made. Its time is `None`.

```python
self.next = Port(send=TensorSpec((32, 256)), initial=torch.full((32, 256), 0.5))
```

`recv_with_time()` returns the value with the time it was sent (a `mujou.time.Timestamp`), or `None` before the first send. Compare it with the time of the value read last to tell a new value from the same one read again. A port that the host writes counts as sent from the `inputs=` of the run, or from the host's first `write`: see [Host I/O](host-io.md#initial-values).

## Exposed ports

A network exposes a child's port by assigning it to an attribute of its own (`self.prev = self.l1.prev`). The attribute is only a reference: the port still belongs to the node, and the outside connects to it as to the node's (`x.next.connect(net.prev)`). By convention, a network exposes the port of its first node as `prev` and that of its last as `next`, so that it connects like a node.
