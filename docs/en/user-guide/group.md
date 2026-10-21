# 🗂️ Grouping Networks

`mujou.run` runs one root. To run several networks in one run, put them side by side in a `mujou.Group`.

This page shows how to write that, and when to split them into separate runs instead.

## How to write it

```python
--8<-- "examples/group.py:group"
```

`Group(enc=a, dec=b)` puts each network at its keyword: the grouped networks have paths under those names.

| Form                             | Path of the node `l1` in `a` |
| -------------------------------- | ---------------------------- |
| `mujou.run(a)`                   | `"l1"`                       |
| `mujou.run(Group(enc=a, dec=b))` | `"enc.l1"`                   |

The paths are the keys of the state tree, so a network has the same paths however it is run.

!!! warning "Group the same way to restore"

    To restore a saved state, group the networks with the same keywords as when the state was saved. Otherwise the paths do not match, and `mujou.run(..., state=tree)` raises `ValueError` (see [Saving and restoring state](state.md)).

`mujou.run(node)` with a single node adds no name either, like a single network. The path of the node is `""`, and the paths of its ports are `"prev"` and so on. In `Group(n=node)`, the path of the node is `"n"`.

## Connect before grouping

Networks whose ports you connect with `connect` before grouping them run connected. `Group` itself connects nothing. The example above groups them with `Group(enc=enc, dec=dec)` after `enc.next.connect(dec.prev)`.

To write input and read output, use the ports of the networks you passed as they are (`enc.prev`, `dec.next`). You do not have to go through the attributes of the `Group`.

!!! warning "A network that ran cannot run again"

    A node runs once. Passing a network that ran (alone with `mujou.run(a)`, say) to a `Group` and running it again raises `RuntimeError`, whether the first run is still running or has ended. Run them together from the start, or create new instances.

## Group them, or run them separately?

Grouped networks become one run. One run shares one timer, one state tree and one ending.

Decide whether to group or to split by whether you want to share these.

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

!!! warning "Runs without a timer share the global timer"

    Even in separate runs, runs given no `timer` both share the global timer. Then taking the state of one stops the time of the other too. If they should not stop each other, give them separate `Timer`s.

```python
import mujou.time

timer_a, timer_b = mujou.time.Timer(), mujou.time.Timer()
with mujou.run(a, timer=timer_a) as run_a, mujou.run(b, timer=timer_b) as run_b:
    timer_a.sleep(2.0)   # wait on the timer of a's run
```

How to use `Timer` is in [Controlling time](time.md).
