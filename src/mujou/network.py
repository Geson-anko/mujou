"""Networks: the composites of nodes and other networks (``specs/network.md``)."""

from mujou.component import Component
from mujou.node import Node


class Network(Component):
    """A composite of nodes and other networks, declared as attributes.

    Assign nodes and networks as attributes and connect their ports with
    ``Port.connect``. The children of a network are the nodes and networks
    among its attributes, in the order of assignment: ``mujou.run`` reads them
    when it starts, as ``nn.Module`` finds its submodules, so nodes in a list,
    a tuple or a dict are not children (the run raises ``TypeError`` for
    them). A port assigned to a network is only a reference to it: expose the
    ports the host uses that way (by convention ``prev`` and ``next``, e.g.
    ``self.prev = self.l1.prev``). A network is only the structure of a run:
    its nodes still run independently and asynchronously.

    ``mujou.run(net)`` runs the network; the ``Run`` it returns is how the host
    talks to the running nodes. A node of the network is at the path of its
    attributes from the root (``"body.l1"``), which names it in the state tree
    and in errors.

    Example::

        class Stack(Network):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = Hidden(784, 256)
                self.l2 = Hidden(256, 256)
                self.l1.next.connect(self.l2.prev)
                self.prev = self.l1.prev  # expose the previous side
                self.next = self.l2.next  # expose the next side


        class Classifier(Network):
            def __init__(self) -> None:
                super().__init__()
                self.body = Stack()
                self.head = Head()  # a node with prev, next and teacher ports
                self.body.next.connect(self.head.prev)  # l2.next <-> head.prev
                self.prev = self.body.prev  # the input
                self.teacher = self.head.teacher  # the target, a port of head


        net = Classifier()
        with mujou.run(net, inputs={net.prev: x0, net.teacher: y0}) as run:
            run.write({net.prev: x, net.teacher: y})  # switch together
            run.timer.sleep(0.1)  # present the input for 0.1 s of the run's time
            prediction = run.read(net.head.next)
            state = run.state_dict()
    """


class Group(Network):
    """A network of the given nodes and networks side by side, each at its
    keyword, unconnected unless they were connected before.

    The node ``l1`` of ``Group(enc=a)`` is at ``"enc.l1"``.

    Example::

        enc, dec = Encoder(), Decoder()
        enc.next.connect(dec.prev)
        with mujou.run(Group(enc=enc, dec=dec)) as run:
            state = run.state_dict()  # nodes "enc.l1", ..., "dec.l1", ...
    """

    def __init__(self, **children: Node | Network) -> None:
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)
