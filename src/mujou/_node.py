"""The node, the unit of asynchronous computation."""

from mujou._network import Component


class Node(Component):
    """A unit of asynchronous computation: the leaf of a network.

    Declare the ports as instance attributes in ``__init__`` (``build`` rejects
    ports declared on the class). When the network runs, the framework calls
    the node's lifecycle hooks (``on_start``, ``on_update(dt)``, ...) on a thread of
    its own; in the hooks the node reads its ports with ``recv`` and writes them
    with ``send``. A node must not hold other nodes or networks as attributes; put
    them side by side in a ``Network`` instead.

    Example::

        class Hidden(Node):
            def __init__(self, n_in: int, n_out: int) -> None:
                x, h = TensorSpec((32, n_in)), TensorSpec((32, n_out))
                # Send W^T eps back and receive x from the previous side (one peer).
                self.prev = PrevPort(send=x, recv=x)
                # Send h forward and receive W^T eps from each next peer (many).
                self.next = NextPort(send=h, recv=h)

            def on_update(self, dt: float) -> None:
                x = self.prev.recv()  # torch.Tensor
                feedback = self.next.recv()  # dict[Channel, torch.Tensor]
                ...
                self.next.send(h)
                self.prev.send(wt_eps)


        l1, l2 = Hidden(784, 256), Hidden(256, 256)
        l1.connect(l2)  # l1.next <-> l2.prev
    """
