# 🧠 How Mujou Works

This page explains the nine ideas you need to write programs with Mujou, in the order you meet them, with diagrams and code. It ends with the rules a node keeps. The [Tutorial](tutorial/1-run.md) puts them to work.

!!! abstract "What you will learn"

    - How you write only one update of a node, and Mujou calls it over and over
    - How nodes talk without waiting for each other, through the latest values only
    - How runs, time, state, simulation and learning are built on top of that

    You should have used PyTorch's `nn.Module` and tensors.

## The big picture

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

The parts of the picture, and the sections that explain them:

| Part             | In a line                                                      | Section                                                |
| ---------------- | -------------------------------------------------------------- | ------------------------------------------------------ |
| Node             | A class that holds one update; Mujou calls it over and over    | [1. Nodes](#1-nodes)                                   |
| Port and channel | A node's connection point, and the line between two of them    | [2. Ports and channels](#2-ports-and-channels)         |
| Spec             | The shape and dtype of the tensors a port sends and receives   | [3. Message types](#3-message-types-specs)             |
| Network          | A container to build with; a path names a node inside it       | [4. Networks and paths](#4-networks-and-paths)         |
| Run              | `mujou.run` starts every node and gives the host a `Run`       | [5. Runs](#5-runs)                                     |
| Time             | One timer per run; updates are in proportion to the time `dt`  | [6. Time](#6-time)                                     |
| State            | The state of the whole run in one dict, to go on from later    | [7. State](#7-state)                                   |
| Simulation       | Virtual time you step by hand, as fast as the machine computes | [8. Simulation and testing](#8-simulation-and-testing) |
| Learning         | Update rules live in nodes; no training mode, no epochs        | [9. Learning](#9-learning)                             |

### From PyTorch

If you know PyTorch, start from this table.

| PyTorch                                     | Mujou                                                                |
| ------------------------------------------- | -------------------------------------------------------------------- |
| You call the model's `forward` in your loop | Mujou calls the node's `on_update(dt)`; you write no loop            |
| The arguments and return value of `forward` | A port's `recv()` and `send()`                                       |
| You assign submodules to a parent module    | You assign nodes to a `Network`                                      |
| `model.train()` and `model.eval()`          | None: inference and learning happen at the same time                 |
| `loss.backward()` reaches the whole model   | Gradients stay inside a node; messages pass between nodes            |
| `model.state_dict()`                        | Each node's `state_dict()`, and `run.state_dict()` for the whole run |

## 1. Nodes

!!! abstract "Key idea"

    A node is a class whose `on_update(dt)` holds one update. You write no loop: Mujou calls it over and over.

A node is the unit of computation, written as a class that inherits from `mujou.Node`. This example, `Affine`, receives a value x and sends `a * x + b`:

```python
--8<-- "examples/quickstart.py:node"
```

| Method          | What you write                                                                                | In `Affine`                                               |
| --------------- | --------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `__init__`      | `super().__init__()` first. Make the ports, and keep the settings and the state in attributes | The input `prev`, the output `next`, settings `a` and `b` |
| `on_update(dt)` | `super().on_update(dt)` first. Read from ports → compute → send to ports                      | `self.next.send(self.a * self.prev.recv() + self.b)`      |

### The order of the hooks

Mujou calls the other hooks too. The diagram shows the main path a node takes through one run.

```mermaid
stateDiagram-v2
    [*] --> on_start
    on_start --> on_update : a restore loads the state here
    on_update --> on_update : every period on average
    on_update --> on_pause : run.state_dict()
    on_pause --> on_resume : after the state is copied
    on_resume --> on_update
    on_update --> on_stop : normal end
    on_update --> on_exception : a hook of any node raised
    on_exception --> on_stop
    on_stop --> [*]
```

| Hook                        | When it is called                                                                   |
| --------------------------- | ----------------------------------------------------------------------------------- |
| `on_start()`                | Once when the run starts (also on a restore). Open I/O here                         |
| `on_update(dt)`             | Over and over, every `period` on average. `dt` is the seconds since the last update |
| `on_pause()`, `on_resume()` | When `run.state_dict()` stops the node to take its state, and when it goes on       |
| `on_exception()`            | Once on every node, before `on_stop`, when a hook of any node has raised            |
| `on_stop()`                 | Once when the run stops (also after an exception). Release I/O and other resources  |

The exact order and its fine print are in the API reference of [`Node`](api/node-and-ports.md#mujou.Node).

!!! tip "Call `super()` first in what you override"

    In `__init__` and in every hook, call `super().__init__(...)` or `super().on_update(dt)` first, as `Affine` above does.

### Synchronous inside, asynchronous between

The way computation runs changes at the boundary of a node.

|           | Inside a node                       | Between nodes                                            |
| --------- | ----------------------------------- | -------------------------------------------------------- |
| Execution | Synchronous: ordinary PyTorch code  | Asynchronous: no node waits for another                  |
| Exchange  | Variables and the computation graph | Messages through ports (copies, detached from the graph) |
| Gradients | Recorded, as plain PyTorch does     | Never cross                                              |

The node class, not a setting, decides how much computation one node holds: one layer, several layers, or a whole model. Put layers that should compute together, or learn together by backpropagation, in one node class.

```mermaid
flowchart LR
    subgraph N1["Node 1 (two layers learning together by BP)"]
        direction LR
        L1["Linear"] --> R["ReLU"] --> L2["Linear"]
    end
    subgraph N2["Node 2"]
        L3["Linear"]
    end
    N1 -. "message (asynchronous)" .-> N2
```

Solid arrows are synchronous computation inside a node; the dotted one is an asynchronous message between nodes.

??? question "Why does a node hold no loop?"

    Mujou runs many nodes on a few worker threads. When a node holds only one update, Mujou decides when and on which thread to call it. That is also what lets it stop every node to take the state, and lets a test update the nodes one step at a time.

## 2. Ports and channels

!!! abstract "Key idea"

    A port is a connection point of a node, and a channel is the line between two ports. A reader never waits: it reads the latest value the peer sent.

`connect` joins two ports with one channel, and returns the channel.

```python
self.scale.next.connect(self.shift.prev)  # the output of scale to the input of shift
```

### Nothing queues: only the latest value stays

A channel is not a queue. A value sent replaces the one before it, and `recv()` returns the latest at that moment.

```mermaid
sequenceDiagram
    participant A as Node A (sends)
    participant C as Channel
    participant B as Node B (reads)
    B->>C: recv()
    C-->>B: the initial value (zeros by default, time None)
    A->>C: send(1)
    A->>C: send(2)
    B->>C: recv()
    C-->>B: 2 (1 was replaced before anyone read it)
    B->>C: recv()
    C-->>B: 2 (the same value again)
    A->>C: send(3)
```

| The channel                      | What `recv()` returns                                                | The time from `recv_with_time()` |
| -------------------------------- | -------------------------------------------------------------------- | -------------------------------- |
| The peer has never sent          | The initial value of the peer's port (`initial`; zeros if not given) | `None`                           |
| The peer has sent                | The last value sent (any earlier ones were replaced)                 | The time that value was sent     |
| Nothing sent since the last read | The same value as before                                             | The same time as before          |

Reading never waits: a value comes back at once, even before the peer has sent anything. Every value carries the time it was sent, so `recv_with_time()` tells a newly arrived value from the same one read again.

??? question "Why not a queue?"

    In a queue, values pile up when the sender and the reader run at different rates, and the nodes end up in step with the slower one. When nodes only read and write the latest value, each runs at its own rate without waiting for the others.

### Port and MultiPort

```mermaid
flowchart LR
    subgraph fan["Port: a sender reaches any number of peers"]
        direction LR
        S["sending Port"] --> R1["receiving Port"]
        S --> R2["receiving Port"]
    end
    subgraph gather["MultiPort: gathers peers of one kind"]
        direction LR
        X1["sending Port"] --> M["MultiPort<br/>recv() → dict[Channel, value]"]
        X2["sending Port"] --> M
    end
```

|                        | `Port`                                              | `MultiPort`                                                        |
| ---------------------- | --------------------------------------------------- | ------------------------------------------------------------------ |
| Peers it receives from | At most one                                         | Any number, of one kind                                            |
| What `recv()` returns  | One value                                           | A `dict[Channel, value]`, in the order of `connect`                |
| Use it for             | One-to-one exchange; one-to-many when it only sends | Gathering inputs of one kind; the node decides how to combine them |

Inputs of different roles (a prediction and an error, say) go to separate `Port`s, not to one `MultiPort`. Port connections may form cycles. More in [Ports and Channels](topics/ports.md).

## 3. Message types (specs)

!!! abstract "Key idea"

    A port declares the shape and dtype of what it sends and receives with a spec. Only ports whose specs match connect.

```python
VECTOR = TensorSpec((3,))      # a float32 vector of 3
self.prev = Port(recv=VECTOR)  # receives only
self.next = Port(send=VECTOR)  # sends only
```

`a.connect(b)` succeeds only when:

| Condition                                                                 | Otherwise                                               |
| ------------------------------------------------------------------------- | ------------------------------------------------------- |
| The spec a sends = the spec b receives (and the same the other way round) | `TypeError`                                             |
| A receiving `Port` has at most one peer                                   | `ValueError`; use a `MultiPort` to receive from several |

A message value is a `torch.Tensor`, a Python scalar (int, float, bool, Timestamp), or a dict of them. The kinds of spec:

| Spec                                                      | What it carries                           | Use it for                                                        |
| --------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------------- |
| `TensorSpec(shape, dtype)`                                | One tensor                                | One value, such as an activity or an error                        |
| `IntSpec(bits)`, `FloatSpec`, `BoolSpec`, `TimestampSpec` | One `int`, `float`, `bool` or `Timestamp` | A count, a flag or a time, without wrapping it in a tensor        |
| `DictSpec`                                                | A `TypedDict` of tensors and scalars      | Values that must arrive together, such as an input and its target |

!!! warning "Send values that belong together in one message"

    The values of two ports arrive separately, so a reader may read a new value of one with an old value of the other. Put values that belong together, such as an input and its target, in a `DictSpec`: then the whole dict arrives at once.

=== "Avoid"

    ```python
    self.x = Port(recv=TensorSpec((32, 4)))  # the input
    self.y = Port(recv=TensorSpec((32, 1)))  # the target, arriving apart from x
    ```

=== "Write instead"

    ```python
    class Batch(TypedDict):
        x: torch.Tensor
        y: torch.Tensor


    BATCH = DictSpec(Batch, x=TensorSpec((32, 4)), y=TensorSpec((32, 1)))
    self.batch = Port(recv=BATCH)  # x and y arrive at once
    ```

`send` sends a copy, detached from the computation graph: changing the tensor afterwards does not change what was sent.

## 4. Networks and paths

!!! abstract "Key idea"

    A `Network` is a container for building nodes together; it runs nothing itself. A path, the attribute names from the root joined with dots, names a node inside.

As you give an `nn.Module` its submodules, you assign nodes to a network's attributes and connect their ports. This network, `Pipeline`, puts two nodes in a row, scale and shift: double, then add 1.

```python
--8<-- "examples/quickstart.py:network"
```

```mermaid
flowchart LR
    subgraph net["Pipeline"]
        direction LR
        S["scale<br/>doubles"] -->|"scale.next → shift.prev"| SH["shift<br/>adds 1"]
    end
    X(["x"]) -->|"net.prev (= scale.prev)"| S
    SH -->|"net.next (= shift.next)"| Y(["y = 2x + 1"])
```

The last line of the code exposes the input of the first node (scale) and the output of the last (shift) as the network's own `prev` and `next`, so that the network connects from outside like a node.

With networks inside networks, a path joins the attribute names from the root with dots:

```mermaid
flowchart TB
    R["root network"] -->|body| B["network"]
    R -->|head| H["node<br/>path head"]
    B -->|l1| L1["node<br/>path body.l1"]
    B -->|l2| L2["node<br/>path body.l2"]
```

| How you build it  | How you write it                                                                   | The path name  |
| ----------------- | ---------------------------------------------------------------------------------- | -------------- |
| Inherit `Network` | Assign nodes to attributes and connect them in `__init__`                          | Attribute name |
| `Group(**named)`  | Only put children side by side under names; connect them before passing them to it | Keyword        |

Paths are the keys of the state ([7. State](#7-state)). A network runs nothing itself; while running, the nodes inside it each run on their own.

## 5. Runs

!!! abstract "Key idea"

    `mujou.run(root)` runs every node in the background and gives the host a handle, the `Run`. Through it, the host writes inputs and reads outputs.

The host is your program, the one that uses the network. This runs the `Pipeline` of section 4 and waits for its output:

```python
--8<-- "examples/quickstart.py:run"
```

```mermaid
sequenceDiagram
    participant H as Host
    participant R as Run
    participant N as Nodes (scale, shift)
    H->>R: mujou.run(net)
    Note over R,N: checks the tree and starts every node
    H->>R: write({net.prev: x})
    loop each node, every period on average
        N->>N: on_update(dt)
    end
    H->>R: wait_for(net.next, condition)
    R-->>H: the value that meets it, and the time it was sent
    H->>R: leave the with block
    Note over R,N: calls on_stop on every node, and ends the run
```

What the host does with a `Run`:

| To                                        | Use                                         |
| ----------------------------------------- | ------------------------------------------- |
| Write an input                            | `run.write`                                 |
| Read the latest output                    | `run.read`, `run.read_with_time`            |
| Wait for an output that meets a condition | `run.wait_for`                              |
| Take every value sent, in turn            | `run.subscribe`                             |
| Wait on the run's time                    | `run.timer.sleep`                           |
| Take the state                            | `run.state_dict()`                          |
| End the run                               | Leave the `with` block, or `run.shutdown()` |

### Before and after a run

```mermaid
stateDiagram-v2
    direction LR
    state "a new instance" as New
    state "running" as Running
    state "ended" as Done
    state "RuntimeError" as Err
    [*] --> New
    New --> Running : mujou.run(net)
    Running --> Done : leaving with, run.shutdown(), a node's exception
    Done --> Err : mujou.run(net) again
```

| The root is | What the host may do with its nodes                                       |
| ----------- | ------------------------------------------------------------------------- |
| Running     | Read and write ports through the `Run`; do not read the nodes' attributes |
| Ended       | Read the nodes' attributes. To run again, build new instances             |

More in [Host I/O](topics/host-io.md).

## 6. Time

!!! abstract "Key idea"

    A run has one time axis, its timer. A node moves its state forward in proportion to `dt`, the time since its last update.

```mermaid
flowchart LR
    H["Host<br/>run.timer"] -->|"waits, pauses, changes the speed"| T["The run's timer<br/>(mujou.time.global_timer unless you pass a Timer)"]
    N["Node<br/>self.timer"] -.->|"only reads"| T
```

Each run has one time axis, its timer, used by every node and by the host: `mujou.time.global_timer` unless the run is given a `Timer` of its own. The host waits on it and pauses it through `run.timer`; a node only reads it, with `self.timer`.

### Updating in proportion to `dt`

Each node updates once every `period` on average, but the intervals differ between nodes and vary. So `on_update` moves the state in proportion to the `dt` it is given, not to the number of updates.

<figure class="mujou-diagram">
<svg viewBox="0 0 640 170" role="img" aria-label="The update times of two nodes. Node A updates at even intervals, node B at uneven ones, one of them much longer">
<line class="mujou-axis" x1="90" y1="150" x2="620" y2="150"/>
<text class="mujou-label" x="620" y="166" text-anchor="end">the run's timer →</text>
<text class="mujou-label" x="10" y="54">Node A</text>
<line class="mujou-lane" x1="90" y1="50" x2="620" y2="50"/>
<g class="mujou-tick-a"><line x1="100" y1="38" x2="100" y2="62"/><line x1="160" y1="38" x2="160" y2="62"/><line x1="220" y1="38" x2="220" y2="62"/><line x1="280" y1="38" x2="280" y2="62"/><line x1="340" y1="38" x2="340" y2="62"/><line x1="400" y1="38" x2="400" y2="62"/><line x1="460" y1="38" x2="460" y2="62"/><line x1="520" y1="38" x2="520" y2="62"/><line x1="580" y1="38" x2="580" y2="62"/></g>
<text class="mujou-note" x="130" y="30" text-anchor="middle">dt</text>
<text class="mujou-note" x="190" y="30" text-anchor="middle">dt</text>
<text class="mujou-label" x="10" y="114">Node B</text>
<line class="mujou-lane" x1="90" y1="110" x2="620" y2="110"/>
<g class="mujou-tick-b"><line x1="100" y1="98" x2="100" y2="122"/><line x1="175" y1="98" x2="175" y2="122"/><line x1="235" y1="98" x2="235" y2="122"/><line x1="400" y1="98" x2="400" y2="122"/><line x1="455" y1="98" x2="455" y2="122"/><line x1="530" y1="98" x2="530" y2="122"/><line x1="590" y1="98" x2="590" y2="122"/></g>
<path class="mujou-brace" d="M 237 92 Q 237 84 245 84 L 390 84 Q 398 84 398 92"/>
<text class="mujou-note" x="317" y="78" text-anchor="middle">a busy machine only makes dt longer</text>
</svg>
<figcaption>Each vertical line is an update. The intervals differ between nodes and vary; every update gets the time since the last one as <code>dt</code></figcaption>
</figure>

A node that follows its input $u$ with the time constant $\tau$, say, obeys this differential equation:

$$
\frac{dx}{dt} = \frac{u - x}{\tau}
$$

and `on_update(dt)` moves it forward by `dt`:

$$
x \leftarrow x + \left(1 - e^{-\Delta t / \tau}\right)(u - x)
$$

=== "Avoid"

    ```python
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        u = self.prev.recv()
        self.x += 0.1 * (u - self.x)  # fixed per update: the update count sets the pace
    ```

=== "Write instead"

    ```python
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        u = self.prev.recv()
        self.x += (1 - math.exp(-dt / self.tau)) * (u - self.x)  # in proportion to dt
    ```

Written in proportion to `dt`, nodes updating at different and varying rates follow the same dynamics; a busy machine only makes `dt` longer, and none of the elapsed time is lost. You build this node, `Leaky`, in [Write a Node](tutorial/2-node.md).

### Two kinds of seconds

| Seconds of      | Used by                                                       |
| --------------- | ------------------------------------------------------------- |
| The run's timer | `dt`, `period`, `run.timer.sleep`, the times values were sent |
| Real time       | `timeout` only                                                |

The host can pause the run's timer or change its speed; `dt` and `period` follow the timer, and only `timeout` is measured in real time. More in [Time](topics/time.md).

## 7. State

!!! abstract "Key idea"

    A node's state is collected from its attributes by itself. `run.state_dict()` takes the state of the whole run in one dict, and `mujou.run(..., state=tree)` goes on from there.

### What is in a node's state

```mermaid
flowchart TD
    A["an attribute of the node"] --> Q1{"a tensor?"}
    Q1 -- yes --> S["state (by itself)"]
    Q1 -- no --> Q2{"has state_dict and load_state_dict?<br/>(an nn.Module, say)"}
    Q2 -- yes --> S
    Q2 -- no --> Q3{"changes while running?"}
    Q3 -- yes --> R["add it with register_state"]
    Q3 -- no --> C["a setting, not state"]
```

| Attribute                                | In the state?                        |
| ---------------------------------------- | ------------------------------------ |
| `self.x = SCALAR.zeros()`                | Yes, by itself: a tensor             |
| `self.model = nn.Linear(4, 2)`           | Yes, by itself: a `mujou.Stateful`   |
| `self.count = 0`                         | With `self.register_state("count")`  |
| `self.buffer = []` (a list of tensors)   | With `self.register_state("buffer")` |
| `self.tau = 0.05` (a setting that stays) | No                                   |

The state of the node's random generator is in it too. The rules are in the API reference of [`Node`](api/node-and-ports.md#mujou.Node).

### Saving the whole run and going on

`run.state_dict()` takes the state of the whole run at one time, without ending it: the nodes' states, the latest message on each channel and the time, in one plain dict keyed by paths.

```python
tree = run.state_dict()
# {
#     "timer": {"time_ns": ..., "time_scale": ...},
#     "nodes": {
#         "body.l1": {"state": ..., "last_update_ns": ..., "ports": {...}},
#         ...
#     },
# }
```

| Key                               | What it holds                                                        |
| --------------------------------- | -------------------------------------------------------------------- |
| `timer`                           | The time `time_ns` and the timer's speed `time_scale`                |
| `nodes` → path → `state`          | That node's `state_dict()`                                           |
| `nodes` → path → `last_update_ns` | The time that node last updated                                      |
| `nodes` → path → `ports`          | The latest message of each port (values sent, values the host wrote) |

```mermaid
flowchart LR
    A["a running Run"] -->|"run.state_dict()"| T["the state tree<br/>(a plain dict on the CPU)"]
    T -->|torch.save| F[("file")]
    F -->|torch.load| T2["the state tree"]
    T2 -->|"mujou.run(new_instance, state=tree)"| B["a Run that goes on<br/>on a timer of its own, from the saved time"]
```

See [Save and Resume](tutorial/6-save-and-resume.md).

## 8. Simulation and testing

!!! abstract "Key idea"

    `mujou.run_manually` runs on a virtual timer that only `run.step` moves: as fast as the machine computes, and the same every time.

`mujou.run` is for the real world: when each node updates varies from run to run. To simulate, or to test, you decide the time instead.

|         | `mujou.run`                                 | `mujou.run_manually`                                 |
| ------- | ------------------------------------------- | ---------------------------------------------------- |
| Timer   | The run's timer (`global_timer` by default) | A virtual timer, which moves only with `run.step`    |
| Updates | Each node, every `period` on average        | Once each in `run.step(dt)`; `period` does not apply |
| `dt`    | The time that actually passed               | The time `step` moved                                |
| Results | Vary from run to run                        | The same every time                                  |

Either call first moves the virtual timer `dt` forward, then updates nodes once each. They differ like this:

| Call                   | Nodes updated        | Order                       | What a node reads                                                                       |
| ---------------------- | -------------------- | --------------------------- | --------------------------------------------------------------------------------------- |
| `run.step(dt)`         | Every node           | In parallel, once each      | Values from the start of the step; what is sent becomes visible after every update ends |
| `run.step(dt, *nodes)` | Only the given nodes | One after another, as given | Also what nodes updated earlier sent in this step                                       |

```mermaid
sequenceDiagram
    participant T as Test
    participant R as ManualRun
    participant A as net.a
    participant B as net.b
    T->>R: run.step(h)
    R->>R: moves the virtual timer h forward
    par in parallel
        R->>A: on_update(h)
    and
        R->>B: on_update(h), reading a's value from the start of the step
    end
    Note over A,B: what a and b sent becomes visible now
    R-->>T: returns
```

This test compares `Leaky`, the node of section 6, with the exact solution:

```python
--8<-- "examples/test_leaky.py:test"
```

See [Test a Node](tutorial/3-test.md) and [Simulation](topics/simulation.md).

## 9. Learning

!!! abstract "Key idea"

    An update rule is written inside the nodes. Only messages pass between nodes, and there is no training mode, no inference mode and there are no epochs.

### A network that keeps learning

Classical machine learning fixes the weights after training, then infers. A running Mujou network takes input, produces output and learns at the same time.

```mermaid
flowchart LR
    subgraph old["Classical machine learning"]
        direction LR
        T1["train"] --> F["fix the weights"] --> I["infer"]
    end
    subgraph new["Mujou"]
        direction LR
        IN["input"] --> NW["a running network"] --> OUT["output"]
        NW -->|"learns meanwhile"| NW
    end
```

### Choosing an update rule

Mujou builds no rule into nodes, ports or the scheduler: the nodes decide how they learn.

| Update rule               | Where gradients are taken                           | Read                                             |
| ------------------------- | --------------------------------------------------- | ------------------------------------------------ |
| Backpropagation in a node | Inside one node, with an ordinary `loss.backward()` | [Learn in a Node](tutorial/4-learn-in-a-node.md) |
| PC-ALM (`mujou.pc`)       | Inside each node; neighbors exchange messages       | [Learn with PC-ALM](tutorial/5-learn-with-pc.md) |
| A rule of your own        | As the node decides                                 | [Update Rules](topics/update-rules.md)           |

```mermaid
flowchart LR
    subgraph bp["Backpropagation in a node"]
        direction LR
        X["input"] --> M["one node<br/>forward → loss → backward → update"]
    end
    subgraph pc["PC-ALM"]
        direction LR
        P1["node"] -- activity --> P2["node"]
        P2 -- activity --> P3["node"]
        P3 -. energy gradient .-> P2
        P2 -. energy gradient .-> P1
    end
```

No computation graph crosses nodes, because a port sends a detached copy: what passes between nodes is messages, such as activities, errors and targets.

## Rules a node keeps

The cautions about nodes on the other pages are gathered here.

| Rule                                                                                                  | If you break it                            |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| [Do not block with `time.sleep`](#sleeping)                                                           | The other nodes' `dt` grows                |
| [A node runs once](#running-again)                                                                    | `RuntimeError`                             |
| [Do not read a node's attributes from the host while it runs](#attributes-while-running)              | It races with the node's updates           |
| [Send nothing that depends on the state in `on_start`](#sending-in-on_start)                          | It sends the state from before the restore |
| [Do not change the state in `on_stop`](#the-state-in-on_stop)                                         | The state taken after a normal end changes |
| [Register the values that change while running](#registering-state)                                   | A restore starts them again                |
| [Keep the graph inside one update](#the-extent-of-the-graph)                                          | The graph keeps growing                    |
| [After an exception, the time stands still](#the-time-after-an-exception)                             | No time passes on `self.timer`             |
| [Only the host moves the time, and only through ports does a node talk](#moving-the-time-and-talking) | `RuntimeError`                             |

### Sleeping

**Do not block with `time.sleep`.**

A node shares its worker thread with other nodes. A hook that blocks holds the worker meanwhile, and the other nodes' `dt` grows.

=== "Avoid"

    ```python
    def on_update(self, dt: float) -> None:
        super().on_update(dt)
        time.sleep(0.1)  # to update every 0.1 s
    ```

=== "Write instead"

    ```python
    def __init__(self) -> None:
        super().__init__(period=0.1)  # updates every 0.1 s on average
    ```

| To                                                   | Use                                   |
| ---------------------------------------------------- | ------------------------------------- |
| Update at intervals                                  | `period`                              |
| Wait on a device, a socket or a lock, with no choice | Give the node `dedicated_thread=True` |
| Wait on the host                                     | `run.timer.sleep`                     |

`run.timer.sleep` follows the run's time scale and pauses; `time.sleep` follows neither.

### Running again

**A node runs once.**

Running a node or network that has run raises `RuntimeError`, so build new instances. After the run, the attributes stay readable.

=== "Avoid"

    ```python
    net = Pipeline()
    with mujou.run(net) as run: ...
    with mujou.run(net) as run: ...  # RuntimeError
    ```

=== "Write instead"

    ```python
    with mujou.run(Pipeline()) as run: ...
    with mujou.run(Pipeline()) as run: ...  # new instances
    ```

### Attributes while running

**Do not read a node's attributes from the host while it runs.**

Reading a node's attributes from the host while it runs races with the node's updates. Read the ports with `run.read` instead, or take the state with `run.state_dict()`.

=== "Avoid"

    ```python
    with mujou.run(net) as run:
        x = net.b.x  # races with the node's updates
    ```

=== "Write instead"

    ```python
    with mujou.run(net) as run:
        y = run.read(net.next)  # read a port
        x = run.state_dict()["nodes"]["b"]["state"]["x"]  # take the state
    ```

### Sending in on_start

**Send nothing that depends on the state in `on_start`.**

```mermaid
flowchart LR
    S["on_start()"] -->|"on a restore"| L["loads the state<br/>load_state_dict(state)"] --> U["the first on_update(dt)"]
    S -->|"otherwise"| U
```

A restore loads the state after `on_start`. In `on_start` the attributes still hold their values from before the restore, and a value sent from them sends that old state.

### The state in on_stop

**Do not change the state in `on_stop`.**

The state taken after a normal end is what the nodes left after `on_stop`. In `on_stop`, only release resources such as I/O.

### Registering state

**Register the values that change while running.**

Tensors and modules are collected by themselves; plain values (a counter, the time last seen) and containers of tensors (a list as a replay buffer) need `register_state`, or a restore starts them again.

=== "Avoid"

    ```python
    self.count = 0  # back to 0 after a restore
    ```

=== "Write instead"

    ```python
    self.count = 0
    self.register_state("count")  # register in __init__
    ```

### The extent of the graph

**Keep the graph inside one update.**

Hooks record gradients as plain PyTorch does. Update the parameters inside `torch.no_grad()`, and detach the values you carry to the next update, or the graph keeps growing.

=== "Avoid"

    ```python
    self.last_y = self.model(x)  # carries the graph to the next update
    ```

=== "Write instead"

    ```python
    self.last_y = self.model(x).detach()
    with torch.no_grad():
        for p in self.model.parameters():
            p -= lr * p.grad
    ```

### The time after an exception

**After an exception, the time stands still.**

| Hook                          | `self.timer`            | To measure time, use |
| ----------------------------- | ----------------------- | -------------------- |
| `on_update(dt)`               | Moves                   | `self.timer`         |
| `on_exception()`, `on_stop()` | Stands still at the end | Real time            |

The run's time stops when the run ends, so `self.timer` does not move in `on_exception` and `on_stop`. Measure real time for work such as slowing a robot down over a second.

### Moving the time and talking

**Only the host moves the time, and only through ports does a node talk.**

| Raises `RuntimeError` in a hook                                            | Why                                                     |
| -------------------------------------------------------------------------- | ------------------------------------------------------- |
| `Timer.sleep`, `pause`, `resume`, `set_time_scale`, `VirtualTimer.advance` | It would change the `dt` of every other node of the run |
| `mujou.run`, the methods of `Run` (`write`, `read`, `state_dict`, ...)     | A node talks only through its own ports                 |

A value a node sends is a copy detached from the computation graph, so `backward()` cannot reach another node either.
