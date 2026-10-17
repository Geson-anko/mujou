# User Guide

This chapter shows how to write nodes in Mujou, put them together into networks, and run them. How to think about nodes, ports, networks, global time and state is in [How Mujou works](../concepts.md). This chapter assumes you have read it.

## Map of the pages

The pages are in the order of the work.

| Page                                        | What you can do after reading it                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [Writing a node](writing-a-node.md)         | Write a node class: declaring ports, hooks, updates in proportion to `dt`, the period, setting up state |
| [Building a network](building-a-network.md) | Connect nodes: branching and merging, nesting, exposed ports, checking the built structure              |
| [Running a network](running.md)             | Run with `mujou.run`, write input from the host, read output, and end the run                           |
| [Composing networks](compose.md)            | Run several networks in one run. Decide when to split them into separate runs instead                   |
| [Saving and restoring state](state.md)      | Take the state while the network runs, save it, and run again from that state                           |
| [Controlling time](time.md)                 | Change how fast time runs. Give a run a timer of its own                                                |
| [PyTorch in nodes](torch.md)                | Write an update rule. Use `nn.Module`, GPUs and BP inside a node                                        |
| [Errors and shutdown](errors.md)            | Know how a node's exception reaches the host and how the network stops                                  |
| [Testing](testing.md)                       | Test nodes deterministically with a virtual timer and a manual run                                      |

## Where to start

If you are new, read [Writing a node](writing-a-node.md), [Building a network](building-a-network.md) and [Running a network](running.md), in that order. These three pages cover everything from writing your own node to running it.

The code examples on each page are taken from the programs in `examples/`, which the tests run.
