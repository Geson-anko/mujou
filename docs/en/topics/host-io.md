# 🏃 Host I/O

The host is the program that runs the network. It talks to a run through the `Run` that `mujou.run` returns: it writes inputs, reads outputs and waits. This page covers each of them, and the initial values of the ports.

## Writing

The host writes to a `Port` of the run that receives and is not connected; a node writes the others. Writing to any other port raises `ValueError`.

```python
t = run.write({net.prev: x, net.teacher: y})  # returns the time of the write
```

- A written value stays until the next write: an input is a signal over time, and the nodes read it many times
- The values of one `write` switch together. A node never reads the new value of one with the old value of the other, so write values that belong together, such as an input and its target, in one `write` (or as one `DictSpec` message)
- The values are copied: changing the tensor after the write does not reach the network
- A value of another shape or dtype raises `ValueError`, and a dict with a missing or an extra key raises `TypeError`, naming the port

## Initial values

| Way                                    | What a reader gets                  | Its time                |
| -------------------------------------- | ----------------------------------- | ----------------------- |
| Nothing                                | Zeros                               | `None`                  |
| `Port(send=..., initial=value)`        | `value`, until the port first sends | `None`                  |
| `mujou.run(net, inputs={port: value})` | `value`, from the first update      | The time the run starts |
| `run.write({port: value})`             | `value`, from the write on          | What `write` returned   |

`initial` is for a port that a node sends on, and `inputs` and `write` are for the ports the host writes. Once `mujou.run` returns, the nodes start updating, so a `write` right after it may come after a few updates that read zeros: to show an input from the first update, pass it as `inputs`.

## Showing an input for a while

Wait on the run's timer with `run.timer.sleep(seconds)`, which follows the time scale and the pauses of the run:

```python
for x in stream:
    run.write({net.prev: x})
    run.timer.sleep(0.1)  # show x for 0.1 s of the run's time
    y = run.read(net.next)  # the output at the end of the showing time
```

The output catches up with a change of input after a delay, which the network's dynamics (the time constants of the nodes, the number of stages) decide; choose a showing time long enough. Reading right after `write` returns the output for an earlier input. When you know the output to expect, wait for it with `wait_for` instead.

## Reading

| Way                         | What it returns                                                         | Suited for                             |
| --------------------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| `run.read(port)`            | The latest value                                                        | Reading whenever it suits the host     |
| `run.read_with_time(port)`  | The latest value and the time it was sent, `(value, Timestamp \| None)` | Telling how new the value is           |
| `run.subscribe(port)`       | An iterator of `(value, Timestamp \| None)`, each newer than the last   | Handling each new output               |
| `run.wait_for(port, until)` | The first `(value, Timestamp \| None)` that meets `until`               | Waiting until an expected output comes |

They read any port of the run that sends, connected or not: reading a connected port observes a value in the middle of the network without changing it. Each returns a new copy on the CPU of what the port last sent with `send` (its initial value until then, with the time `None`).

`subscribe` yields the latest value at once, then waits for each newer one; if the host reads slowly, the values in between are skipped. Its iteration ends when the run ends normally. Leave the loop with `break`, or pass `timeout=`:

```python
--8<-- "examples/subscribe.py:subscribe"
```

`wait_for` checks the latest value, then each newer one, and returns the first that meets the condition. If the run ends normally first, it raises `RuntimeError`.

```python
y, t = run.wait_for(net.next, lambda y: torch.equal(y, expected), timeout=5.0)
```

The timeouts are in real seconds, and the default is to wait forever. When one runs out, `TimeoutError` is raised and the run keeps running (see [Errors and Shutdown](errors.md#timeouts)).

## Ending

| Way                           | What it does                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `with mujou.run(...) as run:` | Calls `shutdown()` when the block ends. Use this normally                                                                      |
| `run.shutdown(timeout=...)`   | Asks every node to stop, and waits until every `on_stop` has returned                                                          |
| `run.join(timeout=...)`       | Waits until the run ends, without asking it to stop: after another thread called `shutdown`, or to wait for a node's exception |

A run keeps running when the host drops it, until `shutdown` or the interpreter's exit, so end each run yourself. After a normal end, `shutdown` does nothing, `write` raises `RuntimeError`, and `run.state_dict()` still takes the last state. If a node raises, the later calls raise `NetworkError` (see [Errors and Shutdown](errors.md)).
