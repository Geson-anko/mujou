# Update Rules

Mujou does not build a fixed way of updating a node's state (an update rule) into the framework. An update rule stays inside a node, and each node chooses and swaps its own.

- Nodes, ports and the execution machinery assume no particular learning rule. The update written in a node's `on_update(dt)` is that node's update rule
- The default update rule is PC-ALM (Augmented Lagrangian Predictive Coding, <https://pub.sakana.ai/pc-alm/>). It was chosen because it is reported to perform close to backpropagation (BP) with local updates, but the choice is provisional
- BP is also usable as an update rule, as long as it stays inside one node. How to write it is in [PyTorch in nodes](torch.md)

!!! note "The PC-ALM nodes are not implemented yet"

    The PC-ALM nodes are planned for the module `mujou.pc`, which is not implemented yet. Until then, subclass `mujou.Node` and write the update equations yourself in `on_update(dt)` (see [Writing a node](writing-a-node.md)).
