# 🧮 Update Rules

How to choose an update rule is in [How Mujou Works](../concepts.md#learning). This page shows a rule of your own that learns across nodes, with feedback alignment as the example.

## Feedback alignment {#feedback-alignment}

`examples/feedback_alignment.py` learns $\hat y = W_2 \tanh(W_1 x)$ with two nodes, `hidden` and `output`. `hidden` does not know `W2`, and learns anyway with a fixed random matrix $B$ ([Lillicrap et al., 2016](https://doi.org/10.1038/ncomms13276)).

```mermaid
flowchart LR
    Host["Host"] -->|"sample x, y"| H["hidden<br/>h = tanh(W1 x)"]
    H -->|"activity h, target y, time"| O["output<br/>ŷ = W2 h"]
    O -->|"error e = ŷ − y, time"| H
    O -->|"loss"| Loss["read by the host"]
```

|                         | BP                 | Feedback alignment                               |
| ----------------------- | ------------------ | ------------------------------------------------ |
| Credit `hidden` uses    | $W_2^\top e$       | $B e$ ($B$ is a fixed random matrix of `hidden`) |
| `hidden` must know `W2` | Yes                | No                                               |
| Why it learns           | It is the gradient | $W_2$ comes to align with $B^\top$               |

```sh
uv run python examples/feedback_alignment.py
```

```python
--8<-- "examples/feedback_alignment.py:hidden"
```

```python
--8<-- "examples/feedback_alignment.py:output"
```

| Condition        | Printed loss of `output` (on each batch, before learning from it) |
| ---------------- | ----------------------------------------------------------------- |
| Both nodes learn | Falls from about 1.5 to about 0.02 in 6 s                         |
| `hidden` frozen  | Stays above 0.15                                                  |

## Two-way ports {#bidirectional}

This example carries the activity and the error on one connection (`hidden.next` and `output.prev`). Connections [may form cycles](ports.md#connect), so the error could as well come back on ports of its own.

| Port          | `send`     | `recv`     |
| ------------- | ---------- | ---------- |
| `hidden.next` | `ACTIVITY` | `ERROR`    |
| `output.prev` | `ERROR`    | `ACTIVITY` |

## Relaying the target {#relay-targets}

=== "Avoid: the target on a port of its own"

    ```mermaid
    sequenceDiagram
        participant H as hidden
        participant O as output
        H->>O: activity h (sample 2)
        Note over O: the target port still holds y of sample 1
        O->>O: compares the prediction for sample 2 with y of sample 1
        H->>O: target y (sample 2)
    ```

=== "Write instead: one message"

    ```mermaid
    sequenceDiagram
        participant H as hidden
        participant O as output
        H->>O: h, y, time (sample 2)
        O->>O: compares the prediction with its own target
        O->>H: e, time
        H->>H: learns only from the error whose time is that of the input it holds
    ```

The nodes of `mujou.pc` also carry `y`, `mask` and `time` in the same message as the activity, a [`pc.msg.Activity`][mujou.pc.msg.Activity].

`recv()` returns the latest value whether or not it is new, so a node that learns once from each sample tells a new one by its time ([A new value, or the same one again](ports.md#new-or-reread)).

## `pc.clock.LearnClock` {#learnclock}

```python
tick = self.clock.tick(dt, t, self.timer.now())  # dt, the time the input was written, the time now
if tick.learn:
    self.optimizer.step_([grad], tick.elapsed)  # weighted by the time since it last learned
```

[`LearnClock`][mujou.pc.clock.LearnClock], which decides when to learn, is a part of `mujou.pc`, but any rule can use it. Weighing each step with `elapsed` makes how far a node learns depend on time, not on how often it updates ([How mujou.pc Works](pc-concepts.md#learnclock)).
