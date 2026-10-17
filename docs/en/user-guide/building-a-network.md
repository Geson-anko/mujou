# 🕸️ Building a Network

You write a network as a class that inherits from `mujou.Network`. This page shows how to:

- connect nodes
- expose ports to the outside
- nest networks
- branch and merge
- check the built structure

How to think about networks is in [Networks](../concepts.md#networks) and [Ports and channels](../concepts.md#ports-and-channels).

## The basic form

In `__init__`, write these three things in order.

1. Assign nodes (or other networks) to attributes
2. Connect their ports with `connect`
3. Expose the ports that should connect from outside, or that the host reads and writes, by assigning them to attributes of the network

```python
--8<-- "examples/quickstart.py:network"
```

Call `super().__init__()` first: the network registers each node, network and port as it is assigned to an attribute, as `nn.Module` registers submodules, and assigning one before `super().__init__()` raises `AttributeError`.

!!! warning "Assign each node to its own attribute"

    Nodes put in a list or a dict are not registered and do not become part of the network: assign each node to an attribute of its own.

## Connecting

Connect two ports with `Port.connect`, as in `a.next.connect(b.prev)`. Nodes and networks have no `connect` of their own: a node can have several inputs and outputs, so naming the two ports always says which path you mean.

By convention, the port on the side of the previous node is a `PrevPort` named `prev`, and the port on the side of the next node is a `NextPort` named `next`, so a chain reads `a.next.connect(b.prev)`.

The names and the kinds only make the direction readable. `PrevPort` and `NextPort` differ from `Port` only in how many peers they take by default (one for `PrevPort`, many for `NextPort`), and a node can have any number of each.

Two ports connect only when the `send` spec of one matches the `recv` spec of the other, in both directions. If they do not match, the call to `connect` raises `TypeError`. Connecting a second channel to a port that takes one peer raises `ValueError`.

`connect` returns the connection it made (a `Channel`). Use it to send a different value to each peer when branching (see [Branching and merging](#branching-and-merging) below).

## Exposing ports

Assigning a child's port to an attribute of the network exposes that port. The kind of the port decides what the exposed port does.

The outside then names the port by a short name. By convention, a network exposes the port of its first node as `prev` and that of its last node as `next` (`self.prev = self.l1.prev`), so that it connects like a node: `x.next.connect(net.prev)` and `net.next.connect(y.prev)`. Expose other ports under names of their own, such as `net.teacher`.

A network can expose only ports held by nodes inside it.

The host can write with `run.write` to any port that receives and is not connected, whether it is exposed or not. It names the port by following the attributes of the children, as in `run.write({net.head.teacher: y})`. Exposing a port only gives it a shorter name.

## Nesting

A network can hold another network as an attribute. If the child network exposes `prev` and `next`, connect them just like the ports of a node.

```python
class Stack(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden(784, 256)
        self.l2 = Hidden(256, 256)
        self.l1.next.connect(self.l2.prev)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()
        self.head = Head()
        self.body.next.connect(self.head.prev)  # body.next is l2.next itself
        self.prev = self.body.prev     # exposes again a port that the child exposed
        self.teacher = self.head.teacher
```

`Hidden` and `Head` are nodes that each have a `PrevPort` and a `NextPort`. `Head` also has a `Port` (`teacher`) that receives a teacher signal.

Nodes are told apart by their path: the attribute names from the root, joined with `.`. The paths of the nodes in this example are `"body.l1"`, `"body.l2"` and `"head"`.

!!! warning "Renaming an attribute changes the path"

    Renaming an attribute changes the path, and a state saved before the rename can no longer be restored (see [Saving and restoring state](state.md)).

A node cannot hold other nodes or networks as attributes: assigning one raises `ValueError`. To line up nodes, put them in attributes of a network.

## Branching and merging

Connecting several channels to one port makes a branch or a merge. The following example splits the output of `split` into two and joins them in `sum`.

```python
--8<-- "examples/branching.py:network"
```

### Branching: the sending side

A `NextPort` takes many peers by default, so writing `a.next.connect(b.prev)` and then `a.next.connect(c.prev)` makes a branch. There are two ways to send.

| How to send               | Where it goes                            | Value read by `Run.read` |
| ------------------------- | ---------------------------------------- | ------------------------ |
| `send(value)`             | The same value to every peer             | Updated                  |
| `send_to(channel, value)` | Only to the peer at the end of `channel` | Not updated              |

If the host reads with `Run.read` a port that only uses `send_to`, it gets the port's initial value (`initial`, or zeros). The host has no way to read the values sent with `send_to` themselves.

`Run.read` reads only ports that have a send type, so it cannot read the receive-only ports of the branches (each branch's `prev`) either. What the host can read is the result that a branch's node computed from the received value and sent with `send` (in the example above, `2 x` at `net.double.next`).

Keep the `Channel` passed to `send_to` apart for each destination. As in the example above, the network assigns the `Channel` that `connect` returns to an attribute of the sending node (`self.split.plus = self.split.next.connect(self.double.prev)`).

!!! warning "Do not pick channels by their order"

    If the node takes them from `self.next.channels` (in the order of connection) instead, swapping the order of `connect` swaps the destinations without any error.

Attributes that hold a `Channel` are not collected into the state.

### Merging: the receiving side

`PrevPort(multi=True)` receives from many peers. On a port with many peers, `recv()` returns a dict keyed by `Channel`, in the order of connection. The value for the channel of a peer that has never sent is the initial value of the peer's port (zeros unless it declares `initial`).

The receiving node decides how to combine the values (sum them, average them, weight each peer, ...). Mujou does not combine them. `Sum` in the example above sums them.

To treat the peers differently, keep the `Channel` of each on the receiving node, as for `send_to`: `self.sum.from_double = self.double.next.connect(self.sum.prev)`, then `self.prev.recv()[self.from_double]` in the node.

A `NextPort` also takes many peers by default, so its `recv()` returns a dict. If the node is sure to have only one next node, declare `NextPort(send=..., multi=False)`, and `recv()` returns the value itself.

## Checking the built structure

`net.build()` checks the structure and the types, freezes them, and returns the built structure (a `Graph`). `mujou.run` calls `build()` itself, so you need not call it just to run. Call it to check the structure before running.

```python
--8<-- "examples/branching.py:build"
```

!!! warning "`build()` freezes the network"

    After `build()`:

    - no port in the tree can `connect` any more
    - no node, network or port can be assigned to or removed from the nodes and networks of the tree (both `RuntimeError`)
    - the network cannot become part of a bigger network

    So call it only on the network you will run.

If the structure is invalid, `build()` raises `ValueError` with the reason. The API reference lists `Graph` and the causes.
