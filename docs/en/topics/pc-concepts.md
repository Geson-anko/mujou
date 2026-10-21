# 🔩 How mujou.pc Works

This page explains what the parts of `mujou.pc` do, how they connect, and what one update computes. After it, the names in "Nodes of your own" of [PC-ALM](pc.md) and in `examples/pc_merge.py` will be familiar. For how to use it, see [5. Learn with PC-ALM](../tutorial/5-learn-with-pc.md).

!!! abstract "What you will learn"

    - What each part of `mujou.pc` does, which parts you use directly and which you use to write nodes of your own
    - The messages that flow between the nodes, and which way the ports face
    - What a `pc.Layer` does in one `on_update(dt)`
    - What the `Tick` that `pc.LearnClock` returns, `jumps` and `pc.TimeConstants` mean

    You should know PyTorch and backpropagation (BP), and the nodes, ports and `dt` of [How Mujou Works](../concepts.md). You need not know predictive coding.

## What a node holds

In PC-ALM every node holds its own **activity** `h`, and **predicts** it from the activity of the node before. These are all the quantities you need.

| Quantity | Formula                              | Meaning                                                                                              |
| -------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `h_in`   |                                      | The activity that came from the node before                                                          |
| `mu`     | $\mu = f(h_{\mathrm{in}};\theta)$    | The prediction: the output of the node's `module`. $\theta$ are the weights of `module`              |
| `h`      |                                      | The node's own activity, which it sends to the next node                                             |
| `lam`    | $\tau_\lambda\dot\lambda = h - \mu$  | The dual: the deviation from the prediction, $r = h - \mu$, summed up over time                      |
| `delta`  | $\delta = h - \mu + \lambda$         | The node's error                                                                                     |
| `g_in`   | $g_{\mathrm{in}} = -J_h^\top \delta$ | The gradient sent back to the node before ($J_h$ is the Jacobian of `module` with respect to `h_in`) |
| `g`      |                                      | The sum of the `g_in` that came from the next nodes                                                  |

`h` moves to cancel the sum of `delta` and `g`, and `lam` keeps growing while `r` remains. When both stop, `r = 0` (`h` is the forward pass) and `lam` equals the credit of BP, $-\partial\ell/\partial h$. The gradients of the weights then equal those of BP too. The theory is in [the PC-ALM article](https://pub.sakana.ai/pc-alm/).

## 1. The parts

!!! abstract "Key idea"

    You usually use `Chain`, `show`, `TimeConstants`, a loss and `Sample`. You use `LearnClock`, `functional` and `Activity` to write nodes of your own.

| Part                                        | Kind      | What it does                                                                                                            | Mainly used        |
| ------------------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------ |
| `pc.Input`                                  | Node      | Sends the `x` the host wrote as it is, as the activity `h`. Holds neither `h` nor weights                               | When built by hand |
| `pc.Layer`                                  | Node      | Holds one `module`, moves `h` and `lam`, and learns the weights                                                         | When built by hand |
| `pc.Output`                                 | Node      | Holds the last `module` and the loss, gives the prediction `y_hat`, and learns the weights. Holds neither `h` nor `lam` | When built by hand |
| `pc.Chain`                                  | Network   | Makes `Input → Layer → … → Output` and connects them                                                                    | Usually            |
| `pc.show`                                   | Function  | The host writes a sample, shows it for a while, and gets the prediction made before learning                            | Usually            |
| `pc.Sample`, `pc.Activity`, `pc.Prediction` | Messages  | The types of the data that flows ([section 2](#2-connections-and-what-flows))                                           | `Sample` usually   |
| `pc.TimeConstants`                          | Settings  | Holds the speeds in seconds ([section 5](#5-timeconstants))                                                             | Usually            |
| `pc.MSELoss`, `pc.CrossEntropyLoss`         | Losses    | Return the loss of each sample. Given to `Output`                                                                       | Usually            |
| `pc.LearnClock`                             | Part      | Decides when to learn ([section 4](#4-learnclock-and-jumps))                                                            | Nodes of your own  |
| `pc.functional`                             | Functions | Formulas without state (`vjp`, `loss_vjp`, `relax_`, `learn_`). `Layer` and `Output` are written with them too          | Nodes of your own  |
| `pc.activity_spec`                          | Function  | Makes the spec of an `Activity`                                                                                         | Nodes of your own  |

`Layer` and `Output` combine `LearnClock` and `functional`. With the same parts you can also make nodes that `Layer` cannot express, such as a node with two inputs.

## 2. Connections and what flows

!!! abstract "Key idea"

    Activities go forward in an `Activity`, and gradients `g_in` go back. The target `y`, `mask` and time `time` ride in the same message as the activity and are passed on.

```mermaid
flowchart LR
    H1["Host"] -->|"Sample"| I["Input"]
    I -->|"Activity"| L0["Layer"]
    L0 -->|"Activity"| L1["Layer"]
    L1 -->|"Activity"| O["Output"]
    O -->|"Prediction"| H2["Host"]
    O -.->|"g_in"| L1
    L1 -.->|"g_in"| L0
    L0 -.->|"g_in"| I
```

Solid lines go forward, dotted lines back. `Input` does not read the `g_in` it gets (`x` is clamped).

| Message         | Fields                            | Flow                                |
| --------------- | --------------------------------- | ----------------------------------- |
| `pc.Sample`     | `x`, `y`, `mask`                  | Host → `Input.prev`                 |
| `pc.Activity`   | `h`, `jumps`, `y`, `mask`, `time` | `next` → the next node's `prev`     |
| `g_in`          | A tensor (the shape of `h_in`)    | `prev` → the previous node's `next` |
| `pc.Prediction` | `y_hat`, `y`, `mask`, `time`      | `Output.next` → host                |

- `mask`: whether each sample has a target. A sample whose `mask` is False adds nothing to the loss, and `Layer` sets the `g_in` of its row to 0. With every `mask` False, `h` and `lam` still relax and only the weights do not learn. Evaluate this way
- `time`: the time (ns) the host wrote the input. 0 before the first write
- `jumps`: the number of times the sender's `h` jumped ([section 4](#4-learnclock-and-jumps))

### Port directions

| Node     | Port   | Kind        | Receives                                | Sends        |
| -------- | ------ | ----------- | --------------------------------------- | ------------ |
| `Input`  | `prev` | `Port`      | `Sample` (the host writes it)           | —            |
| `Input`  | `next` | `MultiPort` | `g_in` (not read)                       | `Activity`   |
| `Layer`  | `prev` | `Port`      | `Activity`                              | `g_in`       |
| `Layer`  | `next` | `MultiPort` | `g_in` from each peer (summed into `g`) | `Activity`   |
| `Output` | `prev` | `Port`      | `Activity`                              | `g_in`       |
| `Output` | `next` | `Port`      | —                                       | `Prediction` |

`next` is a `MultiPort`, so several nodes can read the activity of one node.

??? question "Why does the target travel in the same message as the activity?"

    The nodes are asynchronous, and a port holds only the latest value. If the target came through another port, `Output` would read the activity and the target at different moments, and could pair different samples. In the same message, `y` takes the path of `x` with the same delay. Even when the host writes the next sample faster than it propagates, each prediction is paired with its own target.

## 3. One update of a Layer

!!! abstract "Key idea"

    `Layer.on_update(dt)` goes "read → `tick` → `vjp` → send `g_in` → `relax_` → send `Activity` → `learn_`".

| Step | Part                | Computation                                                                                                                                                                                    |
| ---- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `LearnClock.tick`   | Reads the `Activity` from `prev` and gets the `Tick` (`dt`, `new`, `learn`, `elapsed`)                                                                                                         |
| 2    | —                   | If it learned its own weights in the previous update, adds the jump they made to `h` (not the re-anchoring of step 4; the third row of the table in [jumps](#jumps))                           |
| 3    | `functional.vjp`    | $\mu = f(h_{\mathrm{in}})$, $g_{\mathrm{in}} = -J_h^\top\delta$ and, to learn, $g_\theta = -J_\theta^\top\delta$, in one VJP. Sends `g_in` from `prev`                                         |
| 4    | `functional.relax_` | Re-anchors `h` if `mu` jumped, then $h \leftarrow h - \frac{dt}{\tau_h}(h - \mu + \lambda + g)$, and then, with the moved `h`, $\lambda \leftarrow \lambda + \frac{dt}{\tau_\lambda}(h - \mu)$ |
| 5    | `Activity`          | Advances its own `jumps` if needed, and sends from `next` `h` and `jumps`, with the `y`, `mask` and `time` it received                                                                         |
| 6    | `functional.learn_` | $\theta \leftarrow \theta - \frac{\mathrm{elapsed}}{\tau_\theta}\cdot\frac{g_\theta}{B}$. Only when `tick.learn` is True and some `mask` is True                                               |

- The `delta` of step 3 is made from `h` and `lam` after step 2 and before step 4
- Gradients are summed over the batch, and only `learn_` divides by the number of samples $B$. So `lam` is the credit of each sample. One update of the weights is a step of SGD down the mean gradient of the batch, with the learning rate `elapsed / tau.theta`
- The `delta` of `vjp` is not a value but a function that makes `delta` from `mu` (`mu` is computed inside `vjp`). Usually `lambda mu: h - mu + lam`. In an update that re-anchors in step 4, pass `lambda mu: (h - mu_last) + lam`, with the deviation `h - mu_last` before re-anchoring ([jumps](#jumps))
- `vjp` detaches `h_in` before it differentiates, so the graph does not reach the previous node

`Output` holds neither `h` nor `lam`. In step 3 it uses `functional.loss_vjp`, sends the gradient $\partial\ell/\partial h_{\mathrm{in}}$ of the loss $\ell = \sum_b \mathrm{mask}_b\,\ell_b$ as `g_in`, and sends a `Prediction`. When it learns and how (`learn_`) are as in `Layer`. It has no `h` to re-anchor, so it does not read `jumps`.

??? question "How does sending g_in back differ from backpropagation?"

    `g_in` is computed with the same VJP as one layer of backpropagation ($-J_h^\top\delta$), and at equilibrium `lam` follows the same recursion as the credit of backpropagation. What differs is what is done with `g_in`:

    1. The sender makes `g_in` from its own `delta`: `delta = h - mu + lam` depends on its own state only
    2. The receiver sums the `g_in` it receives into `g`, and uses it in `relax_` (step 4) as a force that moves its own `h`. It does not pass `g` on
    3. As `h` and `lam` move, its own `delta` changes, and so does the `g_in` it sends in its next update. The credit travels back through the relaxation of each node, over time, not in one backward sweep

    There is no forward phase and backward phase, and no node waits for the others: each node updates with the latest `g` it has, which may be old. `vjp` detaches `h_in`, so no computation graph crosses nodes. The gradient equals that of backpropagation only at equilibrium, which is why a node waits `learn_interval` before it learns.

## 4. LearnClock and jumps

!!! abstract "Key idea"

    `h` and `lam` move in every update; the weights move only in the updates `LearnClock` chooses. `jumps` is the signal to the next node: "the prediction jumped, so move `h` with it".

### LearnClock

In every update a node calls `tick = self.clock.tick(dt, int(act["time"]), self.timer.now().ns)`. What the returned `Tick` means:

| Field     | Meaning                                                                      | Use                           |
| --------- | ---------------------------------------------------------------------------- | ----------------------------- |
| `dt`      | `dt` capped at `max_dt`                                                      | Passed to `relax_`            |
| `new`     | The `time` of the input received differs from last time (a new sample)       | Re-anchor `h`                 |
| `learn`   | Whether to learn in this update                                              | Whether to call `learn_`      |
| `elapsed` | The sum of the capped `dt` since the node last learned, this update included | The weight passed to `learn_` |

`learn` is True as follows. `learn_interval` is the number of seconds given to `Layer`, `Output`, `Chain` and `LearnClock`.

| `learn_interval` | The updates that learn                                                                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Positive seconds | The first update after each of `learn_interval`, twice it, … past the time of the write `time`. But not an update where `new` is True: then the next update learns |
| `None`           | Every update (with a new input too). For a stream whose input changes every frame                                                                                  |
| `math.inf`       | Never. `LearnClock.of` does this too when `tau.theta` is `math.inf`                                                                                                |

- An update where `new` is True does not learn because its `delta` is still that of the input before
- When the input changes before the node learned, the time summed in `elapsed` is dropped
- In a node of your own, make it with `pc.LearnClock.of(tau, period=..., learn_interval=...)` and put it in an attribute. That alone puts it in the node's state

??? question "Why count from the time the host wrote the input, not the time the node received it?"

    So that the nodes on a path learn at about the same time, however deep they are. Counted from the time of receipt, deeper nodes would learn later, from a `delta` that holds the jump of `h` that the previous node made by learning (this diverged). Every node reads the same run's timer, so no barrier is needed.

### jumps

Relaxation assumes that `mu` moves little by little. When `mu` jumps at once and `h` stays, $r = h - \mu$ suddenly grows, and a false error enters `g_in` and the gradients of the weights. So in an update where `mu` jumped, `h` moves by the same amount once (re-anchoring), keeping `r` and `lam`.

$$h \leftarrow \mu_{\text{new}} + (h - \mu_{\text{last}})$$

`mu_last` is the prediction of the previous update. The re-anchored `h` jumps too, so the node advances its own `jumps` by 1 to tell the next node.

| Why `mu` jumps                 | How the node notices                       | How `h` moves                                                                                | Its own `jumps`                      |
| ------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------ |
| A new sample                   | `tick.new` is True                         | Re-anchor                                                                                    | +1                                   |
| The previous node's `h` jumped | The `jumps` received differ from last time | Re-anchor                                                                                    | +1                                   |
| It learned its own weights     | It learned                                 | At the start of the next update, add the jump $f(h_{\mathrm{in}};\theta_{\text{new}}) - \mu$ | +1 unless `learn_interval` is `None` |

```mermaid
flowchart LR
    I["Input<br/>jumps +1 on a write"] -->|"jumps change"| L0["Layer<br/>re-anchors, jumps +1"]
    L0 -->|"jumps change"| L1["Layer<br/>re-anchors, jumps +1"]
    L1 --> O["Output<br/>does not read jumps"]
```

- When the causes coincide in one update, the node re-anchors and advances `jumps` only once
- `Input` advances `jumps` on every write of the host
- With `learn_interval=None` the weights move a little in every update, so the node only adds the jump to its own `h` and does not advance `jumps`. Advancing it every time would make the next node re-anchor every time, and deep chains diverge
- Because of re-anchoring, a new sample does not reset `h` or `lam`. Only `r` and `lam` carry over

## 5. TimeConstants

!!! abstract "Key idea"

    All speeds are set in seconds of the run's timer. Each node moves its equations by the `dt` that actually passed, so nodes with different periods change at the same speed per second.

`pc.TimeConstants(h=, lam=, theta=)` holds the three time constants.

| Value       | The speed of                       | Share moved in one update               | With `math.inf`                                                                                          |
| ----------- | ---------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `tau.h`     | The relaxation of the activity `h` | `dt / tau.h`                            | Not allowed (finite and > 0)                                                                             |
| `tau.lam`   | How fast `lam` sums up `r`         | `dt / tau.lam`                          | `lam` stays 0. `r` does not return to 0, and the fixed point does not match BP (plain predictive coding) |
| `tau.theta` | The learning of the weights        | The learning rate `elapsed / tau.theta` | No learning                                                                                              |

Two values take their defaults from the time constants. `tau.period` (`tau.h / 4`) is the interval between a node's updates, the period of every node of a `Chain`. `tau.max_dt` (`min(tau.h / 2, 3 * tau.lam)`) is the longest `dt` one update counts; the time beyond it is dropped from both the relaxation and `elapsed`.

Make `tau.theta` much larger than `tau.h`. When the weights move more slowly than the activities settle, they learn from the settled `delta` even though inference and learning run at the same time. To choose the values, see [Choosing the time constants](pc.md#choosing-the-time-constants); for the arguments, the [API reference](../api/pc.md).

??? question "Why cap dt at max_dt?"

    `g` is made from `h` as it was when the next node last updated, so the nodes are coupled one step behind. When a load stalls every node and then `dt` grows in all of them at once, this coupling makes the network diverge. Splitting the step inside a node does not prevent it, so the time one update counts is capped.
