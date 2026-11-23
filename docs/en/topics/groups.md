# 🗂️ Groups

[`mujou.run`][mujou.run] runs one root, so to run several networks in one run, put them side by side in a [`Group`][mujou.Group]. How nodes get their paths is in [Networks and paths](../concepts.md#networks).

## Group {#group}

```python
--8<-- "examples/group.py:group"
```

| Root                      | Path of the node `l1` in `a` |
| ------------------------- | ---------------------------- |
| `a`                       | `"l1"`                       |
| `Group(enc=a, dec=b)`     | `"enc.l1"`                   |
| `node` (a node by itself) | `""`                         |
| `Group(n=node)`           | `"n"`                        |

### Rules for grouping {#group-rules}

| Rule                                                   | What it means                                                                                                           |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `Group` connects nothing                               | Connect the networks with `connect`, before or after grouping them                                                      |
| Inputs and outputs stay the networks' own ports        | Use `enc.prev` and `dec.next` as they are for `inputs=` and `read`                                                      |
| Restore with the same keywords                         | The keyword starts the path, which is a key of the state tree                                                           |
| A network that has run alone cannot join a group later | A node runs once ([rules a node keeps](../concepts.md#rules)). Run them together from the start, or build new instances |

## One run or several? {#one-or-many}

| What             | Grouped in one run                                | Separate runs                                                                                                                                                    |
| ---------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Connecting       | Possible                                          | Not possible: ports of another run cannot be connected                                                                                                           |
| Timer            | One, shared                                       | Give each its own `Timer`. Otherwise all share [`global_timer`][mujou.time.global_timer], and taking the state of one pauses the others ([Time](time.md#timers)) |
| Taking the state | Pauses every node, and takes one tree at one time | One tree per run; taking it pauses only that run                                                                                                                 |
| Time scale       | The same for every node                           | May differ per run                                                                                                                                               |
| Exceptions       | One node's exception ends every network           | Only the run of the node that raised ends                                                                                                                        |
