# 🕹️ Simulation

## Two ways to run {#two-ways}

|         | [`mujou.run`][mujou.run]                                   | [`mujou.run_manually`][mujou.run_manually]                                                 |
| ------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| For     | Deploying to the real world (sensors, robots)              | Simulating an environment, training faster than real time, testing                         |
| Time    | Real time (`global_timer` by default); sleeps to follow it | Virtual time: a `VirtualTimer` that only `step` moves; no sleeping, as fast as the machine |
| period  | Each node updates every `period` on average                | Not used; once in each `step` that updates the node                                        |
| Results | Vary from run to run                                       | The same, bit for bit, every time                                                          |
| Returns | [`Run`][mujou.Run]                                         | [`ManualRun`][mujou.ManualRun], a `Run` with `step`                                        |

??? question "Why not `mujou.run` with a large time scale?"

    A `Timer` with `time_scale=10` runs ten times faster, but the nodes still update when the machine lets them: `dt` varies, and so do the results. It also cannot go faster than the time scale, however fast the machine is. A `ManualRun` moves the time only when the last step is done.

## A simulation loop {#loop}

This loop feeds a sine wave to the `LeakyChain` of [Write a Node](../tutorial/2-node.md#chain) for 2 s of virtual time, in steps of 1 ms. One pass writes the input of the next step with `run.write`, then calls `run.step(H)`; reads happen between steps (here, after the last one).

```python
--8<-- "examples/simulation.py:simulate"
```

## The two kinds of step {#steps}

Use `step(dt)` unless the order matters to you, as when you compare a node with an algorithm that updates `a`, then `b`. Both move the timer `dt` forward first, then update; the full rules are in [`ManualRun.step`][mujou.ManualRun.step].

=== "`run.step(h)` (Jacobi)"

    ```mermaid
    sequenceDiagram
        participant H as Host
        participant R as ManualRun
        participant A as net.a
        participant B as net.b
        H->>R: run.step(h)
        R->>R: moves the virtual timer h forward
        par on the workers, in parallel
            R->>A: on_update(h)
        and
            R->>B: on_update(h), reading a's value from the start of the step
        end
        Note over A,B: what a and b sent becomes visible now
        R-->>H: returns
    ```

=== "`run.step(h, net.a, net.b)` (Gauss-Seidel)"

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

### Matching an order {#orders}

Compare three ways to step the chain `u → a → b`. Giving every node is not always the same as `step(dt)`.

```python
--8<-- "examples/simulation.py:orders"
```

| Way to step             | The `a` that `b` reads           | Result                |
| ----------------------- | -------------------------------- | --------------------- |
| `step(h)`               | The `a` of the start of the step | Reference             |
| `step(h, net.b, net.a)` | The old `a` (`b` updates first)  | The same as `step(h)` |
| `step(h, net.a, net.b)` | The `a` sent in this step        | Different             |

### dt across steps {#dt-across-steps}

<figure class="mujou-diagram">
<svg viewBox="0 0 640 170" role="img" aria-label="Three steps. net.a updates in each, with dt equal to h. net.b updates only in the third, with dt equal to 3h, the time since the start of the run">
<line class="mujou-axis" x1="110" y1="150" x2="620" y2="150"/>
<text class="mujou-note" x="130" y="166" text-anchor="middle">start</text>
<text class="mujou-note" x="270" y="166" text-anchor="middle">h</text>
<text class="mujou-note" x="410" y="166" text-anchor="middle">2h</text>
<text class="mujou-note" x="550" y="166" text-anchor="middle">3h</text>
<text class="mujou-label" x="10" y="54">net.a</text>
<line class="mujou-lane" x1="110" y1="50" x2="620" y2="50"/>
<g class="mujou-tick-a"><line x1="270" y1="38" x2="270" y2="62"/><line x1="410" y1="38" x2="410" y2="62"/><line x1="550" y1="38" x2="550" y2="62"/></g>
<text class="mujou-note" x="200" y="30" text-anchor="middle">dt = h</text>
<text class="mujou-note" x="340" y="30" text-anchor="middle">dt = h</text>
<text class="mujou-note" x="480" y="30" text-anchor="middle">dt = h</text>
<text class="mujou-label" x="10" y="114">net.b</text>
<line class="mujou-lane" x1="110" y1="110" x2="620" y2="110"/>
<g class="mujou-tick-b"><line x1="550" y1="98" x2="550" y2="122"/></g>
<path class="mujou-brace" d="M 132 98 Q 132 90 140 90 L 540 90 Q 548 90 548 98"/>
<text class="mujou-note" x="340" y="84" text-anchor="middle">dt = 3h (the time of the two steps it missed adds up)</text>
</svg>
<figcaption>Steps 1 and 2 are <code>step(h, net.a)</code>, step 3 is <code>step(h)</code>. Each vertical line is an update</figcaption>
</figure>

## Keeping it deterministic {#determinism}

Keep these rules, and a `ManualRun` gives the same result, bit for bit, every time, whatever the number of workers and whichever thread runs which node (to keep torch's threads from competing for the cores with many workers, see [Settings of the whole process](threads-and-devices.md#process-settings)).

| Rule                                                                                           | Why                                                                                                            |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Draw random numbers from [`self.generator`][mujou.Node.generator], with a `seed`               | The default generator is shared by every node, in an order that varies ([Seeds](threads-and-devices.md#seed))  |
| Move the time only with `step`; do not call `run.timer.advance`                                | The nodes would get a `dt` you did not plan                                                                    |
| Call `write` between steps                                                                     | A write during a step reaches some nodes in that step and others in the next                                   |
| Do not share Python objects between nodes                                                      | Nodes that append to one list, say, do it in a different order each time                                       |
| On the thread that steps, do not call `wait_for`, `subscribe` or `run.timer.sleep`; use `read` | Nothing is sent and no time passes until the next step, so they wait until their timeout (forever without one) |
| To keep several runs in step, pass one `VirtualTimer` to each                                  | They share one time axis, and a step of one run moves the time of the others too                               |

```python
timer = mujou.time.VirtualTimer()  # several runs share one time axis
with (
    mujou.run_manually(a, timer=timer) as run_a,
    mujou.run_manually(b, timer=timer) as run_b,
):
    run_a.step(0.001)  # run_b's timer moves 1 ms too
```

## Step boundaries are only for simulation {#boundaries}

An update rule must not rely on step boundaries. A network that works only under `run.step(dt)` fails when it is deployed with `mujou.run` ([rules a node keeps](../concepts.md#rules)).

|                               | Between two steps of a `ManualRun`                                                           | `mujou.run`                             |
| ----------------------------- | -------------------------------------------------------------------------------------------- | --------------------------------------- |
| Updates of every node         | Finished                                                                                     | Each node at its own times, out of step |
| Time                          | Stands still. The boundary is part of no node; it happens in between                         | Keeps moving                            |
| A moment when all nodes align | Yes                                                                                          | No                                      |
| What a node reads             | Only its own state and its neighbors' latest values; it does not know it is in a `ManualRun` | The same                                |

## Saving, resuming and errors {#save-and-errors}

Take the state between steps with `run.state_dict()` and pass it to new instances with `mujou.run_manually(..., state=tree)`: the steps after it give the same result as a run that did not stop ([Test a Node](../tutorial/3-test.md#restore)).
When a node raises, `step` raises `NetworkError` ([Errors and Shutdown](errors.md); how to write it in pytest is in [Test a Node](../tutorial/3-test.md#exceptions)).
