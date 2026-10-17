# Running a Network

`mujou.run` runs a network in the background and returns `Run`, the handle of the run. The host (the Python program that runs the network) writes input, reads output and ends the run with the methods of `Run`. This page shows how.

## The basic form

```python
--8<-- "examples/quickstart.py:54:62"
```

1. `mujou.run(net)` builds the network and starts every node. It returns after `on_start` has finished on every node
2. When it returns, each node is repeating `on_update` on its own thread. The host's thread does not stop
3. The host writes input with `write`, waits with `mujou.time.sleep`, and reads output with `read`
4. When the `with` statement exits, `shutdown()` is called, and it waits until `on_stop` has finished on every node

`mujou.run` takes the following arguments.

| Argument        | Meaning                                                                                                | Default                   |
| --------------- | ------------------------------------------------------------------------------------------------------ | ------------------------- |
| `*networks`     | The networks or nodes to run. Several are combined into one run (see [Composing networks](compose.md)) | Required                  |
| `timer`         | The run's timer (see [Controlling time](time.md))                                                      | `None` (the global timer) |
| `inputs`        | The initial values of the input. `{port: value}`                                                       | `None`                    |
| `state`         | The state tree to restore (see [Saving and restoring state](state.md))                                 | `None`                    |
| `pause_timeout` | The real time in seconds that `state_dict` waits for every node to pause                               | `30.0`                    |

To run a single node, pass it as `mujou.run(node)` without wrapping it in a network. The path of the node is then `"0"` (see [Composing networks](compose.md)).

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
- `write` returns the time the values were written (a `Timestamp`)

### Showing an input for a set time

Decide how long to show an input by waiting on the run's timer. If the run is on the global timer, use `mujou.time.sleep(seconds)`. Do not use the standard `time.sleep`, because it ignores changes of the time scale and the pauses for taking the state.

```python
for x in stream:
    run.write({net.prev: x})
    mujou.time.sleep(0.1)          # show x for 0.1 seconds
    y, t = run.read(net.next)      # take the output at the end of showing as the response to x
```

The output catches up with a change of input after a delay. The network's dynamics (the time constants of the nodes, the number of stages) decide the length of the delay; Mujou does not. You choose a showing time long enough for the output to catch up. Reading right after `write` returns the output for the previous input. To match inputs with outputs, read at the end of the showing time.

## Reading output

There are two ways to read output.

| Way                   | What it returns                                                                | Suited for                                  |
| --------------------- | ------------------------------------------------------------------------------ | ------------------------------------------- |
| `run.read(port)`      | The latest value at that moment and the time it was sent, `(value, Timestamp)` | The host reads whenever it suits it         |
| `run.subscribe(port)` | An iterator that yields `(value, Timestamp)` in turn                           | Handling the output each time it is renewed |

Both read from any port that has a send type. Reading an unconnected port gives the network's output; reading a connected port observes the value of a node in the middle. The behavior of the nodes does not change.

`read` behaves as follows.

- It returns the value that the node holding the port last sent with `send`. If the node has not sent yet, it returns zeros. Values sent with `send_to` are not reflected in `read`
- The value is a new copy on the CPU

`subscribe` goes as follows.

1. The first item is the latest value at the time of the call, returned without waiting. If nothing has been sent yet, it is zeros
2. After that, it waits until a new value is sent, and returns the latest value at that moment. If you read slowly, the values in between are skipped
3. When the run ends normally, the iteration ends. When the run ends with a node's exception, the iteration raises `NetworkError`

Only the host waits; the sending node does not wait for the host. The iteration ends only when the run ends. It blocks the thread that iterates, so call `shutdown()` from another thread to stop it.

```python
--8<-- "examples/subscribe.py:31:41"
```

## Times: `Timestamp`

The times that `read`, `subscribe` and `write` return are `mujou.time.Timestamp`.

| Form              | Value                                           |
| ----------------- | ----------------------------------------------- |
| `float(ts)`       | Seconds (`float`)                               |
| `ts.ns`           | Nanoseconds (`int`). The exact value            |
| `later - earlier` | The difference of two times. Also a `Timestamp` |
| `a < b` and so on | Which time is earlier                           |

A time is a value on the run's timer. A `Timestamp` does not record which timer it is from, so compare only times from the same timer. If the run is on the global timer, you can also compare with `mujou.time.time()` (seconds as a `float`).

Comparing the time that `write` returns with the time that `read` returns tells whether an output was sent before or after the input was switched.

```python
t_in = run.write({net.prev: x})
mujou.time.sleep(0.1)
y, t_out = run.read(net.next)
if t_out >= t_in:             # y was sent after x was written
    print(float(t_out - t_in))  # seconds from writing x until y was sent
```

## Ending the run

| Way                           | What it does                                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `with mujou.run(...) as run:` | Calls `shutdown()` when the block exits. Use this normally                                                                         |
| `run.shutdown(timeout=...)`   | Asks every node to stop, and waits until `on_stop` has finished on every node                                                      |
| `run.join(timeout=...)`       | Waits until the run ends, without asking it to stop. Use it after another thread calls `shutdown()`, or after `shutdown` times out |

`timeout` is in seconds of real time, and the default is forever. If the run does not end in time, it raises `TimeoutError`. When `shutdown` times out, the request to stop is not withdrawn, and the ending goes on in the background. To wait for that ending, call `join`. When `join` times out, nothing changes, and the run keeps running.

Calling `shutdown` after a normal end does nothing. After a normal end, `run.state_dict()` still takes the last state (see [Saving and restoring state](state.md)). Calling `write` on a run that has ended raises `RuntimeError`.

Letting go of `Run` does not stop the run. Call `shutdown` or use a `with` statement, and always end the run before the program ends.

If any node raises, the whole network stops, and the later methods of `Run` raise `NetworkError`. Details are in [Errors and shutdown](errors.md).
