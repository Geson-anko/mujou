# Networks

A network is the structure that declares how nodes are put together. You write a network as a class that inherits from `mujou.Network`.

## How to declare a network

In the `__init__` of a network, write these three things. It works like giving a PyTorch `nn.Module` its child modules as attributes.

1. Hold nodes, or other networks, as attributes
2. Connect their ports with `connect`
3. Expose the ports that should connect from outside as the network's own attributes

```python
class Stack(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden(784, 256)
        self.l2 = Hidden(256, 256)
        self.l1.connect(self.l2)       # connects l1.next and l2.prev
        self.prev = self.l1.prev       # the exposed PrevPort is the network's previous side
        self.next = self.l2.next       # the exposed NextPort is the network's next side
```

A network that exposes a `PrevPort` and a `NextPort` connects just like a node, as in `stack.connect(head)`.

A network can hold other networks as children (nesting). A node cannot hold other nodes or networks as attributes.

## A network is not a unit of execution

A network is a structure used only for building. It has no hooks and no `on_update`. While running, the nodes inside a network each run on their own, without waiting for each other. Wrapping nodes in a network does not synchronize them.

To compute several layers synchronously, do not use a network; write one node class that holds them all (see [Nodes](nodes.md)).

## Build and run

A network goes through three steps: declare, build, run.

1. Declare: create an instance of the network. Connections can still be added at this point
2. Build: `net.build()` checks the structure and the types, freezes them, and returns the built structure (a `Graph`). It returns the same `Graph` however many times it is called. After building, `connect` is no longer possible
3. Run: `mujou.run(net)` builds the network, runs its `Graph` in the background, and returns `Run`, the handle of the run

Because `mujou.run` calls `build()`, you normally do not call `build()` yourself. A `Graph` is a separate object from the network you declared, and Mujou runs that `Graph`. A `Graph` runs only once.

A `Graph` exposes only `nodes`, the list of paths and nodes, and `port_path(port)`, which returns the path of a port. Running and input/output go through `mujou.run` and `Run`, so you normally do not need the `Graph`.

You can also pass nodes to `mujou.run` in place of networks. If you pass a single node, `mujou.run` runs that node alone, without wrapping it in a network. Its path is described in "Paths" below.

Once the network runs, the host talks to it only through the methods of `Run`.

| Method                | What it does                                                 |
| --------------------- | ------------------------------------------------------------ |
| `run.write({...})`    | Writes input to unconnected ports                            |
| `run.read(port)`      | Reads the latest value of a port and its time                |
| `run.subscribe(port)` | Receives the latest value each time the node sends a new one |
| `run.state_dict()`    | Takes the state (see [State](state.md))                      |
| `run.shutdown()`      | Stops the run and waits until it has stopped                 |
| `run.join()`          | Waits until the run stops, without stopping it               |

`Run` works with the `with` statement and calls `shutdown()` on exit.

## Paths

A node in a network is named by its path: the attribute names from the root, joined with `.`. For example, if the root's attribute `body` is a `Stack` and that `Stack`'s attribute `l1` is a node, the node's path is `"body.l1"`. The state tree also tells nodes apart by this path (see [State](state.md)).

To run several networks in one run, combine them with `mujou.Compose`.

| Network passed to `mujou.run`             | Path of node `l1` |
| ----------------------------------------- | ----------------- |
| `Compose(a, b)`, with `l1` in `b`         | `"1.l1"`          |
| `Compose(enc=a, dec=b)`, with `l1` in `a` | `"enc.l1"`        |

A network passed by position is named `"0"`, `"1"`, ...; a network passed by keyword is named after the keyword. `mujou.run(a, b)` is the same as `mujou.run(Compose(a, b))`. `Compose` also takes nodes, and a node gets a name by the same rule (the path of the node in `Compose(node)` is `"0"`). A `Compose` always adds the names, even around one network or node: in `mujou.run(Compose(enc=a))`, the path of `l1` is `"enc.l1"`.

`mujou.run` with a single network or a single node adds no name in front of the paths. In `mujou.run(node)`, the path of the node is `""`, and the path of its port `prev` is `"prev"`. The state tree also tells this node apart by the path `""` (see [State](state.md)).

Renaming an attribute that holds a node or a network, or changing a connection, changes the paths of nodes and the names of the value slots. A state taken before such a change can then no longer be restored (see [State](state.md)).

How to use networks in detail is in [Building a network](../user-guide/building-a-network.md), [Running a network](../user-guide/running.md) and [Composing networks](../user-guide/compose.md).
