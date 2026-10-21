# 🗂️ Groups

`mujou.run` runs one root. To run several networks in one run, put them side by side in a `mujou.Group`. This page shows how, how paths are named, and when to use separate runs instead.

## Paths

A node is named by its path: the attribute names from the root, joined with `.`.

```python
class Stack(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden(784, 256)
        self.l2 = Hidden(256, 256)
        self.l1.next.connect(self.l2.prev)
        self.prev, self.next = self.l1.prev, self.l2.next


class Classifier(mujou.Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()  # a network inside a network
        self.head = Head()
        self.body.next.connect(self.head.prev)  # body.next is l2.next itself
        self.prev = self.body.prev  # exposes again a port that the child exposed
```

The paths of the nodes of `Classifier` are `"body.l1"`, `"body.l2"` and `"head"`. A run reads the attributes in the order of assignment, so that is the order of the tree. A node given as the root itself is at `""`.

Paths are the keys of the state tree, so a network has the same paths however it is run, and renaming an attribute makes a state saved before the rename impossible to restore.

## Group

```python
--8<-- "examples/group.py:group"
```

`Group(enc=a, dec=b)` puts each network under its keyword.

| Root                      | Path of the node `l1` in `a` |
| ------------------------- | ---------------------------- |
| `a`                       | `"l1"`                       |
| `Group(enc=a, dec=b)`     | `"enc.l1"`                   |
| `node` (a node by itself) | `""`                         |
| `Group(n=node)`           | `"n"`                        |

`Group` connects nothing: connect the networks before or after grouping them, and write and read their ports as they are (`enc.prev`, `dec.next`). To restore a state, group the networks with the same keywords as when it was saved.

A node runs once ([rules a node keeps](../concepts.md#rules-a-node-keeps)), so a network that has run alone cannot join a group: run the networks together from the start, or build new instances.

## One run or several?

Grouped networks are one run, which shares one timer, one state tree and one ending.

| What             | Grouped in one run                                | In separate runs, each on its own `Timer`              |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------ |
| Connecting       | Possible                                          | Not possible: ports of another run cannot be connected |
| Taking the state | Pauses every node, and takes one tree at one time | One tree per run; taking it pauses only that run       |
| Time scale       | The same for every node                           | May differ per run                                     |
| Exceptions       | One node's exception ends every network           | Only the run of the node that raised ends              |

Use separate runs, each given a `mujou.time.Timer()`, when one must not pause while the state of the other is taken, when only one should run faster or slower, or when one should keep running if the other fails.

!!! warning "Runs without a timer share the global timer"

    Runs given no `timer` all run on `mujou.time.global_timer`, so taking the state of one pauses the others too. Give each its own `Timer` to keep them apart (see [Time](time.md)).
