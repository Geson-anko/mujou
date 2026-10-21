"""The graph of a network as a page of HTML, to look at in a browser.

``to_html`` reads the tree of a root as ``mujou.run`` does (``tree.py``) and
embeds it, as JSON, in a page that draws it (``viz.html``): the networks as
boxes, which open and close on a click, the nodes with their ports, and the
channels between them. The page is self-contained, so it opens offline.
"""

import json
from importlib import resources
from typing import Any, TypeIs

from mujou.network import Network
from mujou.node import Node
from mujou.port import AnyPort, MultiPort, Port
from mujou.tree import Tree, walk

__all__ = ["to_html"]


def to_html(root: Node | Network) -> str:
    """Return a page of HTML that draws the graph of ``root``.

    The page shows the children of ``root``; click a network to open it, and
    a node to see its ports and their specs. A channel to a node in a closed
    network goes to the network. Save the page and open it in a browser.

    Example::

        from pathlib import Path

        import mujou.viz

        Path("net.html").write_text(mujou.viz.to_html(net))

    Raises:
        TypeError, ValueError: ``root`` is not a tree that ``mujou.run`` runs,
            as ``mujou.run`` raises.
    """
    tree = walk(root)
    graph = {"root": _component(tree, root, ""), "edges": _edges(tree)}
    # "<" escaped, so that no name ends the script that holds the graph.
    data = json.dumps(graph).replace("<", "\\u003c")
    page = resources.files("mujou").joinpath("viz.html").read_text("utf-8")
    return page.replace("/*GRAPH*/", data)


def _component(tree: Tree, component: Node | Network, path: str) -> dict[str, Any]:
    name = path.rpartition(".")[2] or tree.name(path)
    if isinstance(component, Node):
        return {
            "kind": "node",
            "name": name,
            "path": path,
            "class": type(component).__name__,
            "period": f"{component.period:g}",
            "device": str(component.device),
            "ports": [
                _port(tree, attribute, port)
                for attribute, port in vars(component).items()
                if _is_port(port)
            ],
        }
    children: list[dict[str, Any]] = []
    ports: list[dict[str, str]] = []
    for attribute, value in vars(component).items():
        if isinstance(value, Node | Network):
            children.append(_component(tree, value, _join(path, attribute)))
        elif _is_port(value):
            ports.append({"name": attribute, "path": tree.path(value)})
    return {
        "kind": "network",
        "name": name,
        "path": path,
        "class": type(component).__name__,
        "children": children,
        "ports": ports,
    }


def _port(tree: Tree, attribute: str, port: AnyPort) -> dict[str, Any]:
    return {
        "name": attribute,
        "path": tree.path(port),
        "class": type(port).__name__,
        "send": _spec(port.send_spec),
        "recv": _spec(port.recv_spec),
        # An unconnected Port that receives reads what the host writes.
        "host": isinstance(port, Port)
        and port.recv_spec is not None
        and not port.channels,
    }


def _edges(tree: Tree) -> list[dict[str, str]]:
    """One edge per channel, in the order of the tree: from the end that sends
    if only one does, else from the end first in the tree."""
    edges: list[dict[str, str]] = []
    seen: set[int] = set()
    for port in tree.owners:
        for channel in port.channels:
            if id(channel) in seen:
                continue
            seen.add(id(channel))
            a, b = port, channel.other(port)
            if a.send_spec is None:
                a, b = b, a
            edges.append({"from": tree.path(a), "to": tree.path(b)})
    return edges


def _is_port(value: object) -> TypeIs[AnyPort]:
    return isinstance(value, Port | MultiPort)


def _spec(spec: object) -> str | None:
    return None if spec is None else repr(spec)


def _join(path: str, attribute: str) -> str:
    return f"{path}.{attribute}" if path else attribute
