# 💡 Examples

Working programs that show how to use Mujou, in `examples/` of the repository. The pages of this site include them as they are, so the code shown is the code you can run. They are guides, not tests: the test suite does not run them.

## By what you want to do

| What you want to do                                             | Program                 | Explained in                                                                                       |
| --------------------------------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------- |
| Run something, anything                                         | `quickstart.py`         | [1. Run a Network](tutorial/1-run.md)                                                              |
| Branch a network and merge it again                             | `branching.py`          | [1. Run a Network](tutorial/1-run.md#branching-and-merging), [Ports and Channels](topics/ports.md) |
| Write a node whose state changes in continuous time             | `leaky.py`              | [2. Write a Node](tutorial/2-node.md)                                                              |
| Test a node deterministically                                   | `test_leaky.py`         | [3. Test a Node](tutorial/3-test.md)                                                               |
| Keep learning from a stream, with BP inside a node              | `online_learning.py`    | [4. Learn in a Node](tutorial/4-learn-in-a-node.md)                                                |
| Learn across nodes with a rule of your own                      | `feedback_alignment.py` | [Update Rules](topics/update-rules.md)                                                             |
| Learn a regression with PC-ALM, in a few lines                  | `pc_regression.py`      | [5. Learn with PC-ALM](tutorial/5-learn-with-pc.md)                                                |
| Learn MNIST with PC-ALM, and compare it with BP                 | `mnist/`                | [PC-ALM](topics/pc.md#mnist)                                                                       |
| Save the state and resume from it                               | `save_and_restore.py`   | [6. Save and Resume](tutorial/6-save-and-resume.md)                                                |
| Run several networks together                                   | `group.py`              | [Groups](topics/groups.md)                                                                         |
| Follow every new output (values in between may be skipped)      | `subscribe.py`          | [Host I/O](topics/host-io.md#reading)                                                              |
| Use a timer of a run's own, a time scale, or pause several runs | `timers.py`             | [Time](topics/time.md)                                                                             |
| See what happens when a node raises                             | `errors.py`             | [Errors and Shutdown](topics/errors.md)                                                            |
| Draw the graph of a network, and open it in a browser           | `visualize.py`          | [Visualize a Network](topics/visualize.md)                                                         |

## What each program shows

- `quickstart.py`: a two-node network, run with `with mujou.run(...)`. The host writes an input with `write`, and waits with `wait_for` until the expected output arrives
- `branching.py`: a `Port` that only sends, connected to two branches that a `MultiPort` merges; and a `MultiPort` that sends a different value to each peer with `send_to`, on the channels `connect` returned
- `leaky.py`: the leaky integrator, a node defined in continuous time and advanced by the exact solution, so that it is stable for any `dt`. Run, it steps the input of two in a row and compares them with the exact solution at the time each output was sent
- `test_leaky.py`: tests of the node of `leaky.py` with `mujou.testing`, on a virtual timer with manual steps: against the exact solution, bit for bit against the discrete algorithm, and a resumed run against one that did not stop. Run them with `uv run pytest examples/test_leaky.py`
- `online_learning.py`: a teacher sends batches from y = a x + b with the `a` the host writes and changes; a learner learns from each new batch with plain PyTorch autograd, for the time since the last one, and the host follows its loss as it falls, jumps and falls again
- `feedback_alignment.py`: two nodes learn a two-layer network by feedback alignment: two-way ports, a target relayed with the activity, a learning step once per sample with `pc.LearnClock`
- `pc_regression.py`: `pc.Chain` and `pc.show` learn a regression from a stream of batches, measured before each batch is learned from
- `mnist/`: a 784-256-256-10 network learns MNIST as a `pc.Chain`, and the same network learns by BP from the same initial weights on the same batches. `main.py` is the entry point, `data.py` loads the data, `metrics.py` measures, `bp.py` trains the BP baseline and `watch.py` counts the `dt` of the nodes (`--watch`)
- `save_and_restore.py`: takes the state with `run.state_dict()`, saves it with `torch.save`, and resumes it in a new instance with `mujou.run(..., state=...)`, on a timer of the resumed run's own
- `group.py`: `Group(enc=..., dec=...)`, and the paths of the nodes (`"enc.scale"`)
- `subscribe.py`: `run.subscribe` yields the latest output, then each newer one, and stops with `break`
- `timers.py`: `Timestamp`, a `Timer` of a run's own at a time scale, the scale of the global timer, and pausing the global timer to take the states of two runs at one time; a run resumed from one goes on on a timer of its own
- `errors.py`: the `NetworkError` the host receives and its `__cause__`, the order `on_exception` → `on_stop`, and `mujou.run` raising for an exception in `on_start`
- `visualize.py`: `mujou.viz.to_html` writes the page of a network of two networks and a node, and of a `pc.Chain`, without running them

## How to run them

Run them from the root of the repository, in an environment built from source (see [Installation](install.md)).

```sh
uv run python examples/quickstart.py
uv run python examples/mnist/main.py
```

None of the programs uses a GPU. Each finishes within 15 seconds, except `mnist/`: it downloads MNIST (about 11 MB) on its first run, and its defaults (3 epochs) take about 2 minutes.
