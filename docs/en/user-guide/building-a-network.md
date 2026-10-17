# Building a Network

You write a network as a class that inherits from `mujou.Network`. This page shows how to connect nodes, expose ports to the outside, nest networks, branch and merge, and check the built structure. How to think about networks is in [Networks](../concepts.md#networks) and [Ports and channels](../concepts.md#ports-and-channels).

## The basic form

In `__init__`, write these three things in order.

1. Assign nodes (or other networks) to attributes
2. Connect them with `connect`
3. Expose the ports that should connect from outside, or that the host reads and writes, by assigning them to attributes of the network

```python
--8<-- "examples/quickstart.py:network"
```

A network follows only its direct attributes. Nodes put in a list or a dict do not become part of the network.

## Connecting

| Form                   | What it connects                                                 |
| ---------------------- | ---------------------------------------------------------------- |
| `a.connect(b)`         | The `NextPort` of `a` and the `PrevPort` of `b`                  |
| `a.out.connect(b.inp)` | The two ports named. Connect ports declared with `Port` this way |

`a.connect(b)` and `b.connect(a)` are different connections. In `a.connect(b)`, `a` is the previous node and `b` the next.

Two ports connect only when the `send` spec of one matches the `recv` spec of the other, in both directions. If they do not match, the call to `connect` raises `TypeError`. Connecting a second channel to a port that takes one peer raises `ValueError`.

`connect` returns the connection it made (a `Channel`). Use it to send a different value to each peer when branching (see [Branching and merging](#branching-and-merging) below).

## Exposing ports

Assigning a child's port to an attribute of the network exposes that port. The kind of the port decides what the exposed port does.

| Exposed port | What it makes possible                                         |
| ------------ | -------------------------------------------------------------- |
| `PrevPort`   | The network can be the `net` side of `x.connect(net)`          |
| `NextPort`   | The network can be the `net` side of `net.connect(x)`          |
| `Port`       | The outside can name it by a short name, such as `net.teacher` |

Like a node, a network can expose at most one `PrevPort` and at most one `NextPort`. A network can expose only ports held by nodes inside it.

The host writes to unconnected ports. It can also write to ports that are not exposed. As in `run.write({net.head.teacher: y})`, the host can name any port in the tree by following the attributes of the children. Exposing a port also gives it a shorter name.

## Nesting

A network can hold another network as an attribute. If the child network exposes a `PrevPort` and a `NextPort`, connect it with `connect` just like a node.

```python
class Stack(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden(784, 256)
        self.l2 = Hidden(256, 256)
        self.l1.connect(self.l2)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()
        self.head = Head()
        self.body.connect(self.head)   # connects body.next (in fact l2.next) and head.prev
        self.prev = self.body.prev     # exposes again a port that the child exposed
        self.teacher = self.head.teacher
```

`Hidden` and `Head` are nodes that each have a `PrevPort` and a `NextPort`. `Head` also has a `Port` (`teacher`) that receives a teacher signal.

Nodes are told apart by their path: the attribute names from the root, joined with `.`. The paths of the nodes in this example are `"body.l1"`, `"body.l2"` and `"head"`. Renaming an attribute changes the path, and a state saved before the rename can no longer be restored (see [Saving and restoring state](state.md)).

A node cannot hold other nodes or networks as attributes. If it does, building raises `ValueError`. To line up nodes, put them in attributes of a network.

## Branching and merging

Connecting several channels to one port makes a branch or a merge. The following example splits the output of `split` into two and joins them in `sum`.

```python
--8<-- "examples/branching.py:network"
```

### Branching: the sending side

A `NextPort` takes many peers by default, so writing `a.connect(b)` and then `a.connect(c)` makes a branch. There are two ways to send.

| How to send               | Where it goes                            | Value read by `Run.read` |
| ------------------------- | ---------------------------------------- | ------------------------ |
| `send(value)`             | The same value to every peer             | Updated                  |
| `send_to(channel, value)` | Only to the peer at the end of `channel` | Not updated              |

If the host reads with `Run.read` a port that only uses `send_to`, it gets the port's initial value (`initial`, or zeros). The host has no way to read the values sent with `send_to` themselves. `Run.read` reads only ports that have a send type, so it cannot read the receive-only ports of the branches (each branch's `prev`) either. What the host can read is the result that a branch's node computed from the received value and sent with `send` (in the example above, `2 x` at `net.double.next`).

Keep the `Channel` passed to `send_to` apart for each destination. As in the example above, the network assigns the `Channel` that `connect` returns to an attribute of the sending node. If the node takes them from `self.next.channels` (in the order of connection) instead, swapping the order of `connect` swaps the destinations without any error. Attributes that hold a `Channel` are not collected into the state.

### Merging: the receiving side

`PrevPort(multi=True)` receives from many peers. On a port with many peers, `recv()` returns a dict keyed by `Channel`, in the order of connection. The value for the channel of a peer that has never sent is the initial value of the peer's port (zeros unless it declares `initial`).

The receiving node decides how to combine the values (sum them, average them, weight each peer, ...). Mujou does not combine them. `Sum` in the example above sums them.

A `NextPort` also takes many peers by default, so its `recv()` returns a dict. If the node is sure to have only one next node, declare `NextPort(send=..., multi=False)`, and `recv()` returns the value itself.

## Checking the built structure

`net.build()` checks the structure and the types, freezes them, and returns the built structure (a `Graph`). `mujou.run` calls `build()` itself, so you need not call it just to run. Call it to check the structure before running.

```python
--8<-- "examples/branching.py:build"
```

`build()` freezes the network: afterwards no port in the tree can `connect` any more (`RuntimeError`), and the network cannot become part of a bigger network. So call it only on the network you will run. If the structure is invalid, `build()` raises `ValueError` with the reason. The API reference lists `Graph` and the causes.
