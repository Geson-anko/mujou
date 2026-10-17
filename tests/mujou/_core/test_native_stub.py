"""The generated stub of ``mujou._core._native`` (``just stub``) against the module itself.

pyo3 cannot introspect everything, and ``crates/mujou-stubgen`` adds what it
misses (the exceptions); these tests catch the two drifting apart.
"""

import ast
import inspect
from pathlib import Path

import pytest

from mujou._core import _native

STUB = Path(_native.__file__).parent / "_native.pyi"


def stub_module() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"))


def stub_all() -> list[str]:
    for node in stub_module().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("the stub has no __all__")


def stub_classes() -> dict[str, ast.ClassDef]:
    return {n.name: n for n in stub_module().body if isinstance(n, ast.ClassDef)}


class TestStub:
    def test_lists_every_name_of_the_module(self):
        assert sorted(stub_all()) == sorted(_native.__all__)

    def test_is_complete(self):
        # pyo3 marks a module it cannot fully introspect with a module __getattr__.
        names = {n.name for n in stub_module().body if isinstance(n, ast.FunctionDef)}
        assert "__getattr__" not in names

    @pytest.mark.parametrize("name", sorted(stub_classes()))
    def test_classes_keep_their_docstring_and_bases(self, name: str):
        stub = stub_classes()[name]
        runtime = getattr(_native, name)
        assert ast.get_docstring(stub) == inspect.cleandoc(runtime.__doc__)
        bases = [ast.unparse(b) for b in stub.bases]
        assert bases == [b.__name__ for b in runtime.__bases__ if b is not object]
