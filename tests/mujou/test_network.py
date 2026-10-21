from dataclasses import dataclass, field

from mujou.network import Group, Network
from mujou.tree import walk
from tests.mujou.nodes import Chain, Hidden


class TestGroup:
    def test_puts_each_at_its_keyword(self):
        first, node = Chain(), Hidden()
        tree = walk(Group(enc=first, n=node))
        assert [m.path for m in tree.members] == ["enc.a", "enc.b", "n"]
        assert tree.path(node.prev) == "n.prev"

    def test_nests(self):
        tree = walk(Group(outer=Group(inner=Chain())))
        assert [m.path for m in tree.members] == ["outer.inner.a", "outer.inner.b"]

    def test_keeps_the_connections_made_before(self):
        source, sink = Chain(), Chain()
        channel = source.next.connect(sink.prev)
        tree = walk(Group(source=source, sink=sink))
        assert tree.path(channel.other(source.next)) == "sink.a.prev"


class TestDataclass:
    def test_a_dataclass_network_reads_its_fields(self):
        @dataclass
        class Pair(Network):
            first: Hidden = field(default_factory=Hidden)
            second: Hidden = field(default_factory=Hidden)

            def __post_init__(self) -> None:
                self.first.next.connect(self.second.prev)

        net = Pair()
        tree = walk(net)
        assert [m.path for m in tree.members] == ["first", "second"]
        # A dataclass that compares its fields does not hash: the tree goes by
        # identity.
        assert Pair.__hash__ is None
        walk(Group(a=Pair(), b=Pair()))
