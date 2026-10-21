import json
import math
import re
from typing import Any

import pytest

from mujou import Group, Network, Node, Port, TensorSpec
from mujou.viz import to_html
from tests.mujou.nodes import Hidden


def graph(html: str) -> Any:
    """The graph that ``html`` embeds."""
    found = re.search(
        r'<script type="application/json" id="mujou-graph">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert found is not None
    return json.loads(found.group(1))


H = "TensorSpec((2, 3), torch.float32)"


class Sink(Node):
    def __init__(self) -> None:
        super().__init__(period=0.02)
        self.prev = Port(send=TensorSpec((2, 3)), recv=TensorSpec((2, 3)))
        self.teacher = Port(recv=TensorSpec((2, 5)))


class Stack(Network):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = Hidden()
        self.l2 = Hidden()
        self.l1.next.connect(self.l2.prev)
        self.prev = self.l1.prev
        self.next = self.l2.next


class Classifier(Network):
    def __init__(self) -> None:
        super().__init__()
        self.body = Stack()
        self.head = Sink()
        self.body.next.connect(self.head.prev)
        self.prev = self.body.prev
        self.teacher = self.head.teacher


class Sensor(Node):
    def __init__(self) -> None:
        super().__init__(period=math.inf)
        self.next = Port(send=TensorSpec((3,)))


class Monitor(Node):
    def __init__(self) -> None:
        super().__init__()
        self.prev = Port(recv=TensorSpec((3,)))


def hidden(path: str, name: str) -> dict[str, Any]:
    return {
        "kind": "node",
        "name": name,
        "path": path,
        "class": "Hidden",
        "period": "0.01",
        "device": "cpu",
        "ports": [
            {
                "name": "prev",
                "path": f"{path}.prev",
                "class": "Port",
                "send": H,
                "recv": H,
                "host": path == "body.l1",
            },
            {
                "name": "next",
                "path": f"{path}.next",
                "class": "MultiPort",
                "send": H,
                "recv": H,
                "host": False,
            },
        ],
    }


class TestToHtml:
    def test_tree(self):
        assert graph(to_html(Classifier()))["root"] == {
            "kind": "network",
            "name": "Classifier",
            "path": "",
            "class": "Classifier",
            "children": [
                {
                    "kind": "network",
                    "name": "body",
                    "path": "body",
                    "class": "Stack",
                    "children": [hidden("body.l1", "l1"), hidden("body.l2", "l2")],
                    "ports": [
                        {"name": "prev", "path": "body.l1.prev"},
                        {"name": "next", "path": "body.l2.next"},
                    ],
                },
                {
                    "kind": "node",
                    "name": "head",
                    "path": "head",
                    "class": "Sink",
                    "period": "0.02",
                    "device": "cpu",
                    "ports": [
                        {
                            "name": "prev",
                            "path": "head.prev",
                            "class": "Port",
                            "send": H,
                            "recv": H,
                            "host": False,
                        },
                        {
                            "name": "teacher",
                            "path": "head.teacher",
                            "class": "Port",
                            "send": None,
                            "recv": "TensorSpec((2, 5), torch.float32)",
                            "host": True,
                        },
                    ],
                },
            ],
            # A port of a network is a reference: to the port of a node.
            "ports": [
                {"name": "prev", "path": "body.l1.prev"},
                {"name": "teacher", "path": "head.teacher"},
            ],
        }

    def test_edges(self):
        # One per channel, from the end first in the tree when both send.
        assert graph(to_html(Classifier()))["edges"] == [
            {"from": "body.l1.next", "to": "body.l2.prev"},
            {"from": "body.l2.next", "to": "head.prev"},
        ]

    def test_edge_from_the_end_that_sends(self):
        monitor, sensor = Monitor(), Sensor()
        sensor.next.connect(monitor.prev)
        found = graph(to_html(Group(monitor=monitor, sensor=sensor)))
        assert found["edges"] == [{"from": "sensor.next", "to": "monitor.prev"}]

    def test_node_as_root(self):
        found = graph(to_html(Sensor()))
        assert found == {
            "root": {
                "kind": "node",
                "name": "Sensor",
                "path": "",
                "class": "Sensor",
                "period": "inf",
                "device": "cpu",
                "ports": [
                    {
                        "name": "next",
                        "path": "next",
                        "class": "Port",
                        "send": "TensorSpec((3,), torch.float32)",
                        "recv": None,
                        "host": False,
                    }
                ],
            },
            "edges": [],
        }

    def test_refuses_what_run_refuses(self):
        a, b = Hidden(), Hidden()
        a.prev.connect(b.next)
        with pytest.raises(ValueError, match=r"^a\.prev is connected to MultiPort"):
            to_html(Group(a=a))  # b is outside the tree

    def test_escapes_the_script(self):
        # A name may be any string: it must not end the script that holds it.
        name = "</script><b>x</b>"
        html = to_html(Group(**{name: Sensor()}))
        assert graph(html)["root"]["children"][0]["name"] == name
        assert "<b>x</b>" not in html

    def test_self_contained(self):
        # It opens offline: no script, style or font from elsewhere.
        assert not re.search(r"""(src|href)=["']?(https?:)?//""", to_html(Stack()))
