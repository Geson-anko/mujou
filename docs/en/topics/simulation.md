# 🕹️ Simulation

`mujou.run` runs a network in real time, for the real world. `mujou.run_manually` runs it in virtual time, as fast as the machine computes, for simulations and tests. This page covers when to use which, the two kinds of step, the threads, and what keeps a simulation deterministic.

!!! abstract "What you will learn"

    - Which of `mujou.run` and `mujou.run_manually` to use
    - What `step(dt)` and `step(dt, *nodes)` update, and what each node reads
    - What keeps a simulation the same, bit for bit, every time

    You should know [runs](../concepts.md#5-runs) and [time](time.md).

## Two ways to run

|                     | `mujou.run`                                   | `mujou.run_manually`                                         |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| For                 | Deploying to the real world (sensors, robots) | Simulations and tests                                        |
| Time                | Real time (`global_timer` by default)         | Virtual time: a `VirtualTimer` that only `step` moves        |
| When a node updates | Every `period` on average, on its own         | Once in each `run.step` that updates it; `period` is ignored |
| `dt`                | The time that actually passed                 | Exactly the time that was stepped                            |
| Speed               | Sleeps to follow the clock                    | No sleeping: as fast as the machine computes                 |
| Results             | Vary from run to run                          | The same, bit for bit, every time                            |
| Returns             | `Run`                                         | `ManualRun`, a `Run` with `step`                             |

Use `mujou.run` when the network meets the real world: real inputs arrive in real time, and the network must keep up. Use `mujou.run_manually` when you decide the time: to simulate an environment, to train faster than real time, or to test a node against the discrete algorithm.

`mujou.run_manually` takes the same arguments as `mujou.run`: `timer`, `inputs`, `state` and `workers`. Its `timer` must be a `VirtualTimer`; by default it is a new one at time 0 (or at the time of `state`).

??? question "Why not `mujou.run` with a large time scale?"

    A `Timer` with `time_scale=10` runs ten times faster, but the nodes still update when the machine lets them: `dt` varies, and so do the results. It also cannot go faster than the time scale, however fast the machine is. A `ManualRun` moves the time only when the last step is done.

## A simulation loop

This loop feeds a sine wave to the `LeakyChain` of [Write a Node](../tutorial/2-node.md) for 2 s of virtual time, in steps of 1 ms:

```python
--8<-- "examples/simulation.py:simulate"
```

Each pass of the loop does three things, in this order:

1. The host writes the input of the next step with `run.write`.
2. `run.step(H)` moves the timer `H` forward, then updates every node once. It returns when every update has ended.
3. Between steps (here, after the last one), the host reads with `run.read` or `run.read_with_time`.

The 2 s of virtual time take well under a second of real time.

## The two kinds of step

!!! abstract "Key idea"

    `step(dt)` updates every node at once, each reading the values from the start of the step. `step(dt, *nodes)` updates the given nodes one after another, each reading what the earlier ones sent.

|                      | `run.step(dt)`                                          | `run.step(dt, *nodes)`                              |
| -------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| Which nodes update   | Every node of the run                                   | Only the given nodes                                |
| How                  | In parallel, on the run's workers and dedicated threads | One after another, in the given order               |
| What a node reads    | The values from the start of the step                   | The values sent by the nodes before it in this step |
| What the others see  | What the nodes sent, once every update has ended        | What a node sent, as soon as its update ends        |
| Known in numerics as | Jacobi                                                  | Gauss-Seidel                                        |

Both move the timer `dt` forward first, then update. Use `step(dt)` unless the order matters to you, as when you compare a node with an algorithm that updates `a`, then `b`.

=== "`run.step(h)`"

    ```mermaid
    sequenceDiagram
        participant H as Host
        participant R as ManualRun
        participant A as net.a
        participant B as net.b
        H->>R: run.step(h)
        R->>R: moves the virtual timer h forward
        par on the workers
            R->>A: on_update(h)
        and
            R->>B: on_update(h), reading a's value from the start of the step
        end
        Note over A,B: what a and b sent becomes visible now
        R-->>H: returns
    ```

=== "`run.step(h, net.a, net.b)`"

    ```mermaid
    sequenceDiagram
        participant H as Host
        participant R as ManualRun
        participant A as net.a
        participant B as net.b
        H->>R: run.step(h, net.a, net.b)
        R->>R: moves the virtual timer h forward
        R->>A: on_update(h)
        A-->>B: send (visible at once)
        R->>B: on_update(h), reading a's new value
        R-->>H: returns
    ```

### Matching an order

Giving every node is not the same as `step(dt)`. In the chain `u → a → b`, the three ways differ like this:

```python
--8<-- "examples/simulation.py:orders"
```

- `step(h)`: `b` reads the `a` of the start of the step
- `step(h, net.b, net.a)`: `b` updates first, so it too reads the old `a`. The same result as `step(h)`
- `step(h, net.a, net.b)`: `b` reads the `a` sent in this step. A different result

So when the messages flow one way only (no loop), the order from the output side back to the input side gives the result of `step(dt)`; a chain is one such network. When values also flow back, as in a loop `a ↔ b` or in a `pc.Chain` of PC-ALM, in general no order gives it.

## Rules of every step

- `period` does not apply. A node updates once in each step that updates it, whatever its period.
- The `dt` a node sees is the time since its own last update; the first one is the time since the start of the run. A node left out of some steps sees their time added to its next `dt`.
- A node given twice in one step updates once: the second time, its time has not moved. For the same reason, `step(0)` right after the start updates no node.
- Every message sent in a step has the time of the step.
- A node that sends on a port connected to a port of its own reads, in the same update, the value from before the step.
- A `VirtualTimer` passed to several runs (`timer=`) is one time axis: a step of one run moves the time of the others too.

## Workers and threads

`step(dt)` runs the updates on the run's `workers` (as many as the CPUs by default, as in `mujou.run`) and on the threads of the nodes with `dedicated_thread=True`. The result does not depend on the number of workers or on which thread runs which node. While the GIL is enabled, one thread runs Python code at a time, so the updates overlap only inside torch operations; the result is still the same.

!!! warning "Keep torch's threads few"

    On the CPU, each worker that runs a torch operation starts further threads of torch's own. With many workers, those threads compete for the cores and the step slows down. The host sets them once, before the run:

    ```python
    --8<-- "examples/simulation.py:threads"
    ```

    The rule of thumb is in [Threads and Devices](threads-and-devices.md#settings-of-the-whole-process).

## Keeping it deterministic

A `ManualRun` gives the same result, bit for bit, every time, as long as the program keeps these rules:

| Rule                                                     | Why                                                                                        |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Draw random numbers from `self.generator`, with a `seed` | The global generator is shared by every node, in an order that varies                      |
| Move the time only with `step`                           | Advancing the timer yourself (`run.timer.advance`) gives the nodes a `dt` you did not plan |
| Call `write` between steps                               | A write during a step reaches some nodes in that step and others in the next               |
| Do not share Python objects between nodes                | Nodes that append to one list, say, do it in a different order each time                   |

### Calls of the host

The host calls `write`, `step` and `read` in turn, on one thread: the thread that steps.

| Call                                            | Rule                                                                                                                                                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `write`, `read`, `read_with_time`, `state_dict` | Call them between steps. While `step(dt)` runs, `read` from another thread returns the value from before the step; while `step(dt, *nodes)` runs, it may return what a node already updated in this step sent |
| `wait_for`, `subscribe`, `run.timer.sleep`      | Do not call them on the thread that steps: nothing new is sent and no time passes until the next step, so they wait until their timeout (forever without one). Use `read` instead                             |

## Step boundaries are only for simulation

Between two steps, every node of a `ManualRun` has finished its update and the time stands still. `mujou.run` has no such moment: the nodes update at their own times, with varying `dt`. So an update rule must not rely on the step boundaries; a network that works only under `run.step(dt)` fails when it is deployed with `mujou.run`.

??? question "Is waiting for every node not against the rules a node keeps?"

    A node never waits for another node ([rules a node keeps](../concepts.md#rules-a-node-keeps)). A step boundary is not part of any node: it happens while the run's timer stands still, as a pause to take the state does. A node does not know that it runs in a `ManualRun`; it reads only its own state and its neighbors' latest values.

## Saving, resuming and errors

- `run.state_dict()` takes the state between steps. `mujou.run_manually(new_instances, state=tree)` goes on from it, on a new `VirtualTimer` that starts at the time of the state. The steps after it give the same result as a run that did not stop ([Save and Resume](../tutorial/6-save-and-resume.md)).
- When a node raises during a step, the updates already running finish, the nodes not started yet do not update, and `step` raises `NetworkError` with the node's exception as its `__cause__`. The run has stopped, and what the nodes sent in that step is not published ([Errors and Shutdown](errors.md)).
- `step` raises `ValueError` for a node not in the run or a `dt` that is negative or not finite, and `RuntimeError` after the run has stopped normally or while the timer is paused.

To test a node with `mujou.run_manually`, see [Test a Node](../tutorial/3-test.md).
