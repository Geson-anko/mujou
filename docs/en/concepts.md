# How Mujou Works

This page explains the ideas behind Mujou: nodes, ports and channels, networks, global time, and state. How to write the code is in the [User Guide](user-guide/index.md).

```mermaid
flowchart TB
    H["Host (the Python code that drives the network)"]
    subgraph N["Network"]
        direction LR
        A["Node A"] <-->|Channel| B["Node B"] <-->|Channel| C["Node C"]
    end
    T["The run's timer"]
    H -->|"through the Run that mujou.run(net) returns: write inputs, read outputs, take the state"| N
    T -.->|"time and dt"| N
```

## Nodes

A node is the unit of computation. You write a node as a class that inherits from `mujou.Node`.

Inside a node, ordinary PyTorch code runs synchronously. It may even use backpropagation (BP), as long as the BP stays inside the node. Between nodes, everything is asynchronous. Mujou calls the node's `on_update(dt)` over and over on a thread of its own, and a node never waits for another node. In each `on_update`, the node reads the latest values its neighbors sent, updates its state, and sends its results. Mujou owns the loop, so a node holds no `while` loop of its own. Mujou also calls other hooks that a node can override, for example when the run starts and stops; their order is in the API reference of [`Node`](api/node.md). There are no epochs, no switch between training and inference, and no hook called for each received message.

The node class decides how much computation one node holds: one layer, or several layers, or a whole model. The node boundaries are the asynchronous boundaries. To compute several layers synchronously, put them in one node class.

A node keeps three rules.

- It does not join computation graphs across nodes. A value is detached from the computation graph when it is sent, so gradients never pass between nodes
- It talks to other nodes and to the host only through its own ports
- It only reads the time. Only the host moves the time

If a node raises an exception, the whole run stops. Mujou never stops only the node that raised and keeps the others running (see [Errors and Shutdown](user-guide/errors.md)).

The default update rule will be [PC-ALM](https://pub.sakana.ai/pc-alm/) (Augmented Lagrangian Predictive Coding; not implemented yet), where each node learns with local updates only. Update rules are not built into Mujou: a node can use any rule that keeps to the rules above.

## Ports and channels

A port is a connection point that a node holds. A channel is one connection between two ports, made by `connect`. A channel has one slot for the value in each direction, and only the sender in that direction writes the slot.

A port declares what it sends and what it receives, as a shape and a dtype. Two ports connect only when the type that each sends matches the type that the other receives. Mujou checks this when you call `connect`, and converts no dtype.

Reading and writing work like this.

- A reader gets the latest value at that moment. It never waits for a new value
- Values do not queue up. A value may be overwritten before anyone reads it, and the same value may be read many times. Both are normal
- Before the sender has written, the reader gets the initial value of the sender's port (zeros by default)
- Every value is a copy of a `torch.Tensor`, detached from the computation graph. Changing a received tensor affects no other node
- Every value carries the time it was sent. To tell whether a value is new, read it with its time (`recv_with_time()`) and compare that time with the last one

A port may connect to several peers. It then receives one value per channel, and the node decides how to combine them (sum, average, ...).

A port may also stay unconnected. Unconnected ports are the inputs and outputs of the network: the host writes input to an unconnected input port, and the written value stays there until the host writes again. The host can also read the latest value that any sending port sent, connected or not.

## Networks

A network declares how nodes are put together. You write it as a class that inherits from `mujou.Network`, much like giving a PyTorch `nn.Module` its child modules.

1. Hold nodes, or other networks, as attributes
2. Connect their ports with `connect`
3. Expose the ports that should connect from outside as the network's own attributes

A network that exposes ports connects to other nodes and networks just like a node does.

A network is used only for building. It has no hooks and runs nothing itself. While running, the nodes inside it each run on their own, so wrapping nodes in a network does not synchronize them.

`mujou.run(net)` checks the structure and the types, freezes it, and runs it. After that, no connection can be added. Each node is named by its path, the attribute names from the root joined with `.`, such as `"body.l1"`. The state uses these paths, so renaming an attribute or changing a connection makes a state taken before the change impossible to restore.

## Global time

Mujou treats a learning network as a system that changes continuously over time. `dt` is the time in seconds since the node's last update, and a node updates its state in proportion to `dt`. For a state $x$ that follows $\frac{dx}{dt} = f(x)$, one update is `x = x + dt * f(x)` (explicit Euler).

Nodes update at different rates, and their rates vary with the load. If the update size were fixed per iteration, a faster node would change its state more. With updates proportional to `dt`, every node follows the same dynamics on the same flow of time; a higher load only makes `dt` longer. Do not cap `dt`, because a capped `dt` drops time that really passed. If a large `dt` makes the update unstable, split it into smaller steps inside one `on_update`.

A node can have a period. It then updates once every period on average, instead of as fast as it can. Even so, `dt` is the time that actually passed, so update in proportion to `dt` in every case.

Each run has one timer, and every node of the run and the host use it. By default, it is the global timer, one per process. The host can make the time run faster or slower than real time. To wait on that time, for example to show an input for two seconds, the host uses `mujou.time.sleep`, not the standard `time.sleep`, which waits in real time. Nodes only read the time.

To take the state of a running network, Mujou pauses the run and stops its timer. The paused time does not enter any `dt` after the run resumes, so to the nodes it is as if the pause never happened. A pause stops the time of every run that uses the same timer; give a run its own timer to keep it apart (see [Controlling Time](user-guide/time.md)). A pause is for taking the state and for observing. Do not make pausing part of how the network learns.

## State

You can take the state of a running network without stopping it, and later run the network again from that state. The state is one tree that holds:

- the state of each node (weights, activities, and other tensors), by node path
- the latest value in each slot, with its time
- the time of each node's last update, and the time of the timer

The values in the slots are part of the state because a network is its nodes together with the values flowing between them. If they went back to their initial values, the neighbors would leave the equilibrium they had reached.

A node collects its own state, as `nn.Module` does: tensors and `nn.Module`s assigned to its attributes go into the state automatically.

All nodes are paused while the state is taken, between two updates, so every value in the tree belongs to the same moment.

| When you take the state | What you get                                                                                                                                                                                              |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| While the network runs  | The state at that moment                                                                                                                                                                                  |
| After a normal end      | The values as the nodes left them after `on_stop`. The timer's time is when `shutdown()` asked the run to stop, unless a node updated or sent later. A restored run does not count the time after the end |
| After an exception      | Nothing: it raises `NetworkError`. Take the state you want to keep while the network runs                                                                                                                 |

Because the tree after a normal end is taken after `on_stop`, a change that `on_stop` makes to the state goes into that tree. For example, `self.w = None` in `on_stop` removes `w` from it. Do not change the state in `on_stop`; release only resources such as I/O there.

To restore, pass the tree as `mujou.run(net, state=tree)`. The network goes on from the time the tree was taken, with the same node states and slot values. The network must have the same node paths and connections as the one the tree was taken from. Storing the tree in a file (`torch.save`, `pickle`, ...) is up to you. See [Saving and Restoring State](user-guide/state.md).
