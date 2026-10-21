# 🧭 User Guide

This chapter shows how to write nodes in Mujou, put them together into networks, and run them. How to think about nodes, ports, networks, global time and state is in [How Mujou works](../concepts.md). This chapter assumes you have read it.

## 🗺️ Map of the pages

The pages are in the order of the work.

<div class="grid cards" markdown>

- **🧩 Writing a node**

    ______________________________________________________________________

    Write a node class: declaring ports, hooks, updates in proportion to `dt`, the period, setting up state.

    [:octicons-arrow-right-24: Writing a node](writing-a-node.md)

- **🕸️ Building a network**

    ______________________________________________________________________

    Connect nodes: branching and merging, nesting, exposed ports, checking the structure.

    [:octicons-arrow-right-24: Building a network](building-a-network.md)

- **🏃 Running a network**

    ______________________________________________________________________

    Run with `mujou.run`, write input from the host, read output, and end the run.

    [:octicons-arrow-right-24: Running a network](running.md)

- **🗂️ Grouping networks**

    ______________________________________________________________________

    Run several networks in one run. Decide when to split them into separate runs instead.

    [:octicons-arrow-right-24: Grouping networks](group.md)

- **💾 Saving and restoring state**

    ______________________________________________________________________

    Take the state while the network runs, save it, and run again from that state.

    [:octicons-arrow-right-24: Saving and restoring state](state.md)

- **⏱️ Controlling time**

    ______________________________________________________________________

    Change how fast time runs. Give a run a timer of its own.

    [:octicons-arrow-right-24: Controlling time](time.md)

- **🔥 PyTorch in nodes**

    ______________________________________________________________________

    Write an update rule. Use `nn.Module`, GPUs and BP inside a node.

    [:octicons-arrow-right-24: PyTorch in nodes](torch.md)

- **🚨 Errors and shutdown**

    ______________________________________________________________________

    Know how a node's exception reaches the host and how the network stops.

    [:octicons-arrow-right-24: Errors and shutdown](errors.md)

- **🧪 Testing**

    ______________________________________________________________________

    Test nodes deterministically with a virtual timer and a manual run.

    [:octicons-arrow-right-24: Testing](testing.md)

- **🧠 Learning with PC-ALM**

    ______________________________________________________________________

    Train a model split into nodes with `mujou.pc`: stream a dataset, measure the accuracy, choose the time constants.

    [:octicons-arrow-right-24: Learning with PC-ALM](pc.md)

</div>

## 🚩 Where to start

If you are new, read [Writing a node](writing-a-node.md), [Building a network](building-a-network.md) and [Running a network](running.md), in that order. These three pages cover everything from writing your own node to running it.

The code examples on each page are taken from the programs in `examples/`, which you can run as they are.
