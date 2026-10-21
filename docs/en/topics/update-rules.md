# 🧮 Update Rules

An update rule (a learning rule) is what a node writes in `on_update(dt)`. Mujou builds no rule into nodes, ports or the scheduler: each node chooses its own, and the same network can swap them. This page shows how to write a rule that learns across nodes, with feedback alignment as the example; [4. Learn in a Node](../tutorial/4-learn-in-a-node.md) has backpropagation (BP) inside one node, and [PC-ALM](pc.md) the rule of `mujou.pc`.

## Within a node, across nodes

Inside a node, anything PyTorch computes is allowed, BP included. Across nodes, no computation graph can be joined: a port sends a detached copy. A rule across nodes therefore sends what each node needs as messages: activities forward, errors (or other credits) back, and the targets along with the activities. Each node learns from its own state and its neighbors' messages only.

## Feedback alignment

`examples/feedback_alignment.py` learns $\hat y = W_2 \tanh(W_1 x)$ with two nodes. `hidden` sends its activity forward, and `output` sends the error $e = \hat y - y$ back. BP would give `hidden` the credit $W_2^\top e$, but `hidden` never sees `W2`: it uses $B e$ with a fixed random matrix $B$ of its own, and learns anyway, as $W_2$ comes to align with $B^\top$ ([Lillicrap et al., 2016](https://doi.org/10.1038/ncomms13276)).

```sh
uv run python examples/feedback_alignment.py
```

```python
--8<-- "examples/feedback_alignment.py:hidden"
```

```python
--8<-- "examples/feedback_alignment.py:output"
```

The program prints the loss of `output` on each batch before learning from it, which falls from about 1.5 to about 0.02 in 6 s; with `hidden` frozen, it stays above 0.15.

## Two-way ports

`hidden.next` is `Port(send=ACTIVITY, recv=ERROR)` and `output.prev` is `Port(send=ERROR, recv=ACTIVITY)`: one connection carries the activity forward and the error back. Connections may form cycles, so the error could as well come back on ports of its own.

## Relaying the target

The activity carries the target `y` of its sample and the time of the write it came from. Values on two ports travel independently, so a target sent on a port of its own could pair a prediction with the target of another sample. In one message, `output` always compares a prediction with its own target, and `hidden` learns only from the error whose `time` is that of the input it holds. The nodes of `mujou.pc` carry `y`, `mask` and `time` in their `pc.Activity` the same way.

## Telling what is new

`recv()` returns the latest value whether or not it is new. A node that learns once from each sample tells a new one by its time: the time `recv_with_time()` returns, or a time the message carries. Before anyone sends, the time is `None` (or, for a time in the message, whatever the zeros of the initial value say).

## `pc.LearnClock`

Both nodes decide when to learn with a `pc.LearnClock`, a part of `mujou.pc` that any rule can use. In each update, `tick(dt, time_ns, now_ns)` takes the `dt`, the time the input was written at and the time now, and returns a `pc.Tick`:

- `new`: the input is new (its time is not the one seen last)
- `learn`: learn in this update. With `learn_interval`, that is the first update `learn_interval` after each write (and every `learn_interval` while the input stays), but never the update that first sees an input; with `learn_interval=None`, every update
- `elapsed`: the time since the clock last learned, each `dt` capped at `max_dt`. Weigh the step with it: $W \leftarrow W - \frac{\text{elapsed}}{\tau}\,\nabla$ follows the gradient flow $\tau \dot W = -\nabla$, so how far a node learns depends on time, not on how often it updates
- `dt`: the `dt` capped at `max_dt`

The clock has `state_dict` and `load_state_dict`, so an attribute that holds it is in the node's state by itself. The nodes on a path read the one timer of the run, so they learn at about the same time after each write, however deep they are.
