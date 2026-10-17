import copy
import io
import pickle
from collections import OrderedDict
from typing import Any

import pytest
import torch
from torch import nn

from mujou import StateDict


def tree() -> StateDict:
    """A tree of the shape ``Run.state_dict`` returns, with every kind of value."""
    module = nn.Linear(2, 1).state_dict()  # an OrderedDict with _metadata
    x = torch.arange(3.0)
    return StateDict(
        nodes={
            "a": {"x": x, "alias": x, "module": module, "count": 3, "name": "a"},
            "b": {"history": [torch.ones(1), (torch.zeros(1), 0.5)], "size": x.shape},
        },
        slots={"a.next/out": (torch.full((2,), 2.0), 7)},
        last_update_ns={"a": 5, "b": 6},
        timer={"time_ns": 7, "scale": 1.0},
    )


def tensors(value: Any) -> list[torch.Tensor]:
    """Every tensor in ``value``, in order."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensors(v)]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if type(value) in (list, tuple):
        return [t for v in value for t in tensors(v)]
    return []


def assert_same_tree(moved: Any, original: Any) -> None:
    """``moved`` has the structure and values of ``original``."""
    assert type(moved) is type(original)
    if isinstance(original, torch.Tensor):
        assert torch.equal(moved.cpu(), original.cpu())
    elif isinstance(original, dict):
        assert list(moved) == list(original)  # pyright: ignore[reportUnknownArgumentType]
        for key in original:  # pyright: ignore[reportUnknownVariableType]
            assert_same_tree(moved[key], original[key])
    elif type(original) in (list, tuple):
        assert len(moved) == len(original)
        for m, o in zip(moved, original, strict=True):
            assert_same_tree(m, o)
    else:
        assert moved == original


class TestStateDict:
    def test_is_a_dict(self):
        t = tree()
        assert isinstance(t, dict)
        assert t["timer"] == {"time_ns": 7, "scale": 1.0}

    def test_to_moves_every_tensor_into_new_containers(self):
        original = tree()
        before = copy.deepcopy(original)
        moved = original.to("meta")
        assert isinstance(moved, StateDict)
        found = tensors(moved)
        assert len(found) == len(tensors(original)) == 7
        assert all(t.device == torch.device("meta") for t in found)
        # The original keeps its tensors and containers.
        assert_same_tree(original, before)
        assert all(t.device == torch.device("cpu") for t in tensors(original))
        # Every container is new and of the same type, other values are as they were.
        node, module = moved["nodes"]["a"], moved["nodes"]["a"]["module"]
        assert node is not original["nodes"]["a"]
        assert type(module) is OrderedDict
        assert module is not original["nodes"]["a"]["module"]
        assert module._metadata == original["nodes"]["a"]["module"]._metadata  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        history = moved["nodes"]["b"]["history"]
        assert type(history) is list and type(history[1]) is tuple
        assert moved["nodes"]["b"]["size"] == torch.Size([3])
        assert moved["slots"]["a.next/out"][1] == 7
        assert node["count"] == 3 and node["name"] == "a"

    def test_to_keeps_a_shared_tensor_shared(self):
        moved = tree().to("meta")
        node = moved["nodes"]["a"]
        assert node["x"] is node["alias"]

    def test_to_shares_a_tensor_already_on_the_device(self):
        # As torch.Tensor.to does: no copy. (A move to the CPU is tested on CUDA.)
        original = tree()
        moved = original.cpu()
        assert isinstance(moved, StateDict)
        assert_same_tree(moved, original)
        assert moved["nodes"]["a"]["x"] is original["nodes"]["a"]["x"]
        assert moved["nodes"]["a"] is not original["nodes"]["a"]

    def test_pickles_as_a_state_dict(self):
        original = tree()
        again = pickle.loads(pickle.dumps(original))
        assert_same_tree(again, original)
        assert b"mujou\nStateDict" in pickle.dumps(original, protocol=0)

    def test_torch_load_reads_it_with_weights_only(self):
        original = tree()
        buffer = io.BytesIO()
        torch.save(original, buffer)
        buffer.seek(0)
        again = torch.load(buffer)  # weights_only=True by default
        assert_same_tree(again, original)

    @pytest.mark.cuda
    def test_moves_from_cuda_to_the_cpu(self):
        on_cuda = tree().to("cuda")
        assert all(t.device.type == "cuda" for t in tensors(on_cuda))
        moved = on_cuda.cpu()
        assert all(t.device == torch.device("cpu") for t in tensors(moved))
        assert_same_tree(moved, tree())
