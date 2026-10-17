# Grouping Networks

To run several networks in one run, pass them side by side as `mujou.run(a, b)`, or combine them with `mujou.Group`. This page shows how to write that, and when to split them into separate runs instead.

## How to write it

```python
--8<-- "examples/group.py:group"
```

`mujou.run(a, b)` is the same as `mujou.run(Group(a, b))`. To give names, use `Group` with keyword arguments.

The grouped networks have paths under the names given by `Group`.

| Form                  | Path of the node `l1` in `a` |
| --------------------- | ---------------------------- |
| `mujou.run(a)`        | `"l1"`                       |
| `mujou.run(a, b)`     | `"0.l1"`                     |
| `Group(a, b)`         | `"0.l1"`                     |
| `Group(enc=a, dec=b)` | `"enc.l1"`                   |

The paths are the keys of the state tree. A position and a keyword give different paths. To restore a saved state, group the networks the same way as when the state was saved: the same order, and keywords only where they were used. Otherwise the paths do not match, and `mujou.run(..., state=tree)` raises `ValueError` (see [Saving and restoring state](state.md)).

`mujou.run(node)` with a single node adds no name either, like a single network. The path of the node is `""`, and the paths of its ports are `"prev"` and so on. `Group(node)` adds a name, so there the path of the node is `"0"`.

An invalid keyword (such as an empty name or one containing `.`) raises `ValueError`.

## Connect before grouping

Networks that you connect with `connect` before grouping them run connected. `Group` itself connects nothing. The example above groups them with `Group(enc=enc, dec=dec)` after `enc.connect(dec)`.

To write input and read output, use the ports of the networks you passed as they are (`enc.prev`, `dec.next`). You do not have to go through the attributes of the `Group`.

If you run a network alone with `mujou.run(a)`, that network is built alone. Passing the same network to `mujou.run(a, b)` afterwards raises `RuntimeError`, whether the first run is still running or has ended. Run them together from the start, or create new instances.

## Group them, or run them separately?

Grouped networks become one run. One run shares one timer, one state tree and one ending. Decide whether to group or to split by whether you want to share these.

| What is shared   | When grouped                                                          | In separate runs, each given its own `Timer`           |
| ---------------- | --------------------------------------------------------------------- | ------------------------------------------------------ |
| Connecting       | Possible                                                              | Not possible. Ports of another run cannot be connected |
| Taking the state | Pauses every node at once and takes the state at one time as one tree | Taken per run. Taking it pauses only that run          |
| Time scale       | The same for every node                                               | Can differ per run                                     |
| Exceptions       | One node's exception ends every network                               | Only the run of the node that raised ends              |

In the following cases, do not group them; use separate runs, and give each one a `mujou.time.Timer()`.

- You do not want to pause one when you take the state of the other
- You want to speed up or slow down only one of them
- You want one to keep running even if the other stops with an exception

Even in separate runs, runs given no `timer` both share the global timer. Then taking the state of one stops the time of the other too. If they should not stop each other, give them separate `Timer`s.

```python
timer_a, timer_b = mujou.time.Timer(), mujou.time.Timer()
with mujou.run(a, timer=timer_a) as run_a, mujou.run(b, timer=timer_b) as run_b:
    timer_a.sleep(2.0)   # wait on the timer of a's run
```

How to use `Timer` is in [Controlling time](time.md).
