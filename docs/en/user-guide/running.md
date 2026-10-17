# Running a Network

`mujou.run` runs a network in the background and returns `Run`, the handle of the run. The host (the Python program that runs the network) writes input, reads output and ends the run with the methods of `Run`. This page shows how.

## The basic form

```python
--8<-- "examples/quickstart.py:run"
```

1. `mujou.run(net)` builds the network and starts every node. It returns after `on_start` has finished on every node
2. When it returns, each node is repeating `on_update` on its own thread. The host's thread does not stop
3. The host writes input with `write` and reads output with `read`, `subscribe` or `wait_for`. The example above waits with `wait_for` until the expected value arrives
4. When the `with` statement exits, `shutdown()` is called, and it waits until `on_stop` has finished on every node

`mujou.run` takes the following arguments.

| Argument                 | Meaning                                                                                                    | Default                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------- |
| `component, *components` | The networks or nodes to run. Two or more are combined into one run (see [Composing networks](compose.md)) | Required                  |
| `timer`                  | The run's timer (see [Controlling time](time.md))                                                          | `None` (the global timer) |
| `inputs`                 | The initial values of the input. `{port: value}`                                                           | `None`                    |
| `state`                  | The state tree to restore (see [Saving and restoring state](state.md))                                     | `None`                    |
| `pause_timeout`          | The real time in seconds that `state_dict` waits for every node to pause                                   | `30.0`                    |

To run a single node, pass it as `mujou.run(node)` without wrapping it in a network. The path of the node is then `""`, and the paths of its ports are `"prev"` and so on (see [Composing networks](compose.md)).

A network runs only once. Passing a network that was once passed to `mujou.run` to `mujou.run` again, whether it is still running or has ended, raises `RuntimeError`. To run again, create new instances, nodes included.

## Writing input

The host can write to a port that is unconnected, takes one peer, and has a receive type. Nodes write to connected ports, so the host cannot write to them.

| Way                                    | When to use it                                               |
| -------------------------------------- | ------------------------------------------------------------ |
| `mujou.run(net, inputs={port: value})` | Input to show from the first `on_update` (the initial value) |
| `run.write({port: value, ...})`        | To switch the input while the network runs                   |

Without `inputs`, the initial value of the input is zeros. When `mujou.run` returns, the nodes are already repeating `on_update`. So even if you `write` right after `mujou.run`, the nodes read zeros in the updates before the `write`. To show a non-zero input from the first update, pass it with `inputs`.

`write` behaves as follows.

- A written value stays until the next write. Input is "a signal that changes over time", and nodes read the same input many times
- The values of several ports written in one `write` switch at the same time. A node never reads the new value of only one of them. Put values that should switch as a pair, such as an input and its teacher signal, in one `write`
- The values are copied. Changing the tensor after writing it does not reach the network
- `write` returns the time the values were written

### Showing an input for a set time

Decide how long to show an input by waiting on the run's timer. If the run is on the global timer, use `mujou.time.sleep(seconds)`. Do not use the standard `time.sleep`, because it ignores changes of the time scale and the pauses for taking the state.

```python
for x in stream:
    run.write({net.prev: x})
    mujou.time.sleep(0.1)          # show x for 0.1 seconds
    y, t = run.read(net.next)      # take the output at the end of showing as the response to x
```

The output catches up with a change of input after a delay. The network's dynamics (the time constants of the nodes, the number of stages) decide the length of the delay; Mujou does not. You choose a showing time long enough for the output to catch up. Reading right after `write` returns the output for the previous input. To match inputs with outputs, read at the end of the showing time. If you know the output to expect, wait for it with `wait_for` instead (see "Reading output" below).

## Reading output

There are three ways to read output.

| Way                             | What it returns                                                                | Suited for                                  |
| ------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------- |
| `run.read(port)`                | The latest value at that moment and the time it was sent, `(value, Timestamp)` | The host reads whenever it suits it         |
| `run.subscribe(port)`           | An iterator that yields `(value, Timestamp)` in turn                           | Handling the output each time it is renewed |
| `run.wait_for(port, condition)` | The first `(value, Timestamp)` whose value meets `condition`                   | Waiting until an expected output arrives    |

All of them read from any port that has a send type, and return the time as a `mujou.time.Timestamp` (see [Controlling time](time.md)). Reading an unconnected port gives the network's output; reading a connected port observes the value of a node in the middle. The behavior of the nodes does not change.

`read` behaves as follows.

- It returns the value that the node holding the port last sent with `send`. If the node has not sent yet, it returns the port's initial value (`initial`, or zeros). Values sent with `send_to` are not reflected in `read`
- The value is a new copy on the CPU

`subscribe` goes as follows.

1. The first item is the latest value at the time of the call, returned without waiting. If nothing has been sent yet, it is the port's initial value
2. After that, it waits until a new value is sent, and returns the latest value at that moment. If you read slowly, the values in between are skipped
3. When the run ends normally, the iteration ends. When the run ends with a node's exception, the iteration raises `NetworkError`
4. With `timeout=seconds`, the iteration raises `TimeoutError` once that many seconds of real time have passed since the call to `subscribe`, the time spent in the body of the loop included. The run keeps running. The default is forever

Only the host waits; the sending node does not wait for the host. The following example leaves the loop with `break` after 20 values. Without `break` or `timeout`, the loop blocks its thread until the run ends; another thread can end it by calling `shutdown()`.

```python
--8<-- "examples/subscribe.py:subscribe"
```

To wait until an expected output arrives, use `wait_for`. It checks the latest value first, then each newer one, and returns the first that meets the condition. After `timeout` seconds of real time it raises `TimeoutError` (the run keeps running); if the run stops first, it raises `RuntimeError`.

```python
y, t = run.wait_for(net.next, lambda y: torch.equal(y, expected), timeout=5.0)
```

## Ending the run

| Way                           | What it does                                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `with mujou.run(...) as run:` | Calls `shutdown()` when the block exits. Use this normally                                                                         |
| `run.shutdown(timeout=...)`   | Asks every node to stop, and waits until `on_stop` has finished on every node                                                      |
| `run.join(timeout=...)`       | Waits until the run ends, without asking it to stop. Use it after another thread calls `shutdown()`, or after `shutdown` times out |

`timeout` is in seconds of real time, and the default is forever. What happens when it runs out is in "Timeouts" in [Errors and shutdown](errors.md).

Calling `shutdown` after a normal end does nothing. After a normal end, `run.state_dict()` still takes the last state (see [Saving and restoring state](state.md)). Calling `write` on a run that has ended raises `RuntimeError`.

Letting go of `Run` does not stop the run. Always end the run with `shutdown` or a `with` statement.

If any node raises, the whole network stops, and the later methods of `Run` raise `NetworkError`. Details are in [Errors and shutdown](errors.md).
