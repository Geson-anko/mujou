# Examples

These are working programs that show how to use Mujou. They live in `examples/` in the repository. The documentation pages include these files as they are. The tests run every program, so if the code shown here stops working, the tests fail.

## The list

Pick the row closest to what you want to do. The "Explained in" pages include the program and explain it.

| What you want to do                                                 | Program               | Explained in                                                                                                       |
| ------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Run something, anything                                             | `quickstart.py`       | [Quickstart](../getting-started/quickstart.md), [Running a Network](../user-guide/running.md)                      |
| Write a node whose state changes over time                          | `leaky_chain/`        | [Writing a Node](../user-guide/writing-a-node.md), [Global Time](../concepts.md#global-time)                       |
| Branch a network and merge it again                                 | `branching.py`        | [Building a Network](../user-guide/building-a-network.md), [Ports and Channels](../concepts.md#ports-and-channels) |
| Run several networks together                                       | `compose.py`          | [Composing Networks](../user-guide/compose.md), [Networks](../concepts.md#networks)                                |
| Save learned state and continue from it later                       | `save_and_restore.py` | [Saving and Restoring State](../user-guide/state.md), [State](../concepts.md#state)                                |
| Take the states of several runs at one time                         | `pause.py`            | [Saving and Restoring State](../user-guide/state.md), [Controlling Time](../user-guide/time.md)                    |
| Keep receiving the latest output (values in between may be skipped) | `subscribe.py`        | [Running a Network](../user-guide/running.md)                                                                      |
| Train a PyTorch model with BP inside a node                         | `torch_node.py`       | [PyTorch in Nodes](../user-guide/torch.md)                                                                         |
| Keep learning from a stream while its target changes                | `online_learning.py`  | [PyTorch in Nodes](../user-guide/torch.md), [Writing a Node](../user-guide/writing-a-node.md)                      |
| Test a node                                                         | `testing_a_node.py`   | [Testing](../user-guide/testing.md)                                                                                |
| Change how time advances                                            | `timers.py`           | [Controlling Time](../user-guide/time.md), [Global Time](../concepts.md#global-time)                               |
| See what happens when a node raises an exception                    | `errors.py`           | [Errors and Shutdown](../user-guide/errors.md)                                                                     |

What each program shows:

- `quickstart.py`: runs a two-node network with `with mujou.run(...)`, writes input with `write`, and waits with `run.wait_for(..., timeout=...)` until the expected output arrives
- `leaky_chain/`: a node defined in continuous time (a leaky integrator). The update uses the coefficient of the exact solution, so it is stable for any `dt`. The output is compared with the exact solution at the time elapsed from the write (the time that `run.write` returns) to the output
- `branching.py`: sends from one `NextPort` to several nodes and merges them with `PrevPort(multi=True)`. `send_to` sends only to the peer on a Channel that `connect` returned
- `compose.py`: `Compose(enc=..., dec=...)` and `mujou.run(a, b)`, and how node paths (`"enc.scale"`, `"0.scale"`) are named
- `save_and_restore.py`: takes the state with `run.state_dict()`, saves it with `torch.save` (its tensors moved to the CPU), and restores it with `mujou.run(..., state=...)` to continue
- `pause.py`: pauses the global timer with `mujou.time.pause()`, takes the states of two runs on it at that one time, and resumes it with `mujou.time.resume()`. A run restored from one of the states goes on from that time
- `subscribe.py`: with `run.subscribe`, first receives the latest output at that moment, then the latest one each time a newer output is sent. If the next value is sent before one is received, the value in between is skipped. The loop stops with `break` after 20 values
- `online_learning.py`: a teacher node sends a batch from y = a x + b every 10 ms, with the `a` that the host writes; the host changes `a` while the network runs. A learner learns once from each new batch, with no separate training and inference phases. The host follows the learner's loss with `run.wait_for`: it falls, jumps when the target changes, and falls again as the learner catches up
- `torch_node.py`: trains an `nn.Module` inside a node with BP kept inside the node. Setting the seed, and an update that advances `dt` in smaller steps
- `testing_a_node.py`: runs a node with a fixed `dt` and order, using the virtual timer and manual updates of `mujou.testing`, and compares the result bit for bit with a reference computation. It also checks that a run restored from a state taken midway matches a run that did not stop
- `timers.py`: the time point `Timestamp`, the per-run timer `Timer`, and the speed of time (the run's timer and the global timer)
- `errors.py`: the `NetworkError` the host receives and its `__cause__`, the order `on_exception` → `on_stop`, and that `mujou.run` raises an exception from `on_start`

Only `leaky_chain/` is a directory. Its entry point is `main.py`, and the node definitions are in `leaky.py`.

## How to run them

Run them from the root of the repository, in an environment built from source (see [Installation](../getting-started/installation.md)).

```sh
uv run python examples/quickstart.py
uv run python examples/leaky_chain/main.py
```

None of the programs use a GPU or a network connection, and each finishes within a few seconds.
