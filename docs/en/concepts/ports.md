# Ports and Channels

!!! note "To be written"

    A port is a node's end of a connection; a channel joins two ports. Messages carry `torch.Tensor` values stamped with the global time of the write. A reader always gets the latest value and never waits. `TensorSpec` and `DictSpec` declare the shapes.
