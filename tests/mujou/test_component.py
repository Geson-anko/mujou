from dataclasses import dataclass
from typing import Any

import pytest

from mujou.network import Network
from mujou.node import Node
from mujou.port import MultiPort, Port
from mujou.spec import TensorSpec
from tests.mujou.nodes import Hidden


class TestClassAttributes:
    @pytest.mark.parametrize(
        "part",
        [Port(send=TensorSpec((2,))), MultiPort(recv=TensorSpec((2,))), Hidden()],
        ids=["Port", "MultiPort", "Node"],
    )
    @pytest.mark.parametrize("base", [Node, Network])
    def test_rejects_a_part_of_the_tree_on_the_class(
        self, base: type[Node | Network], part: Any
    ):
        with pytest.raises(TypeError, match=r"^Declared\.part is a \w+ on the class"):
            type("Declared", (base,), {"part": part})

    def test_rejects_slots(self):
        with pytest.raises(TypeError, match=r"^Slotted has __slots__"):

            class Slotted(Node):
                __slots__ = ("out",)

    def test_rejects_a_dataclass_with_slots(self):
        with pytest.raises(TypeError, match=r"^Pipeline has __slots__"):

            @dataclass(slots=True)
            class Pipeline(Network):  # pyright: ignore[reportUnusedClass]
                l1: Hidden

    def test_takes_any_other_value_on_the_class(self):
        class Declared(Node):
            on_device = "cpu"
            spec = TensorSpec((2,))

        assert Declared.on_device == "cpu"
