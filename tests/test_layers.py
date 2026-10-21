"""The import layers of ``src/mujou``: the source of truth for the rule.

A module imports only modules of lower layers (not of its own), and only the
modules in ``NATIVE`` import the extension ``mujou._core._native``. ``mujou.pc``
is written as a user's code is: on the modules in ``PUBLIC`` only, never on
``mujou._core``. Every import is at the top level of its module: none under
``TYPE_CHECKING`` or in a function, which would hide a dependency.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
EXTENSION = "mujou._core._native"

# From the bottom up.
LAYERS: list[set[str]] = [
    {"mujou._core", EXTENSION, "mujou._core.errors"},
    {"mujou._core.spec"},
    {"mujou._core.codec"},
    {"mujou._core.port"},
    {"mujou._core.component"},
    {"mujou._core.node"},
    {"mujou._core.network"},
    {"mujou._core.driver"},
    {"mujou._core.tree"},
    {"mujou._core.wiring"},
    {"mujou._core.state"},
    {"mujou._core.run"},
    {"mujou", "mujou.time", "mujou.testing"},
    {
        "mujou.pc._messages",
        "mujou.pc._time_constants",
        "mujou.pc._losses",
        "mujou.pc._clock",
    },
    {"mujou.pc.functional"},
    {"mujou.pc._input", "mujou.pc._layer", "mujou.pc._output"},
    {"mujou.pc._chain"},
    {"mujou.pc"},
]
RANK = {module: rank for rank, layer in enumerate(LAYERS) for module in layer}

NATIVE = {
    "mujou._core.codec",
    "mujou._core.port",
    "mujou._core.node",
    "mujou._core.wiring",
    "mujou._core.run",
    "mujou.time",
}

PUBLIC = {"mujou", "mujou.time", "mujou.testing"}


def module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def sources() -> dict[str, Path]:
    return {module_name(path): path for path in (SRC / "mujou").rglob("*.py")}


def imports(path: Path, modules: set[str]) -> Iterator[str]:
    """The modules of ``modules`` that ``path`` imports."""
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: a relative import"
            base = node.module or ""
            names = [f"{base}.{alias.name}" for alias in node.names]
            names = [name if name in modules else base for name in names]
        else:
            continue
        yield from (name for name in names if name in modules)


def in_pc(module: str) -> bool:
    return module == "mujou.pc" or module.startswith("mujou.pc.")


def breaks_the_rule(importer: str, imported: str) -> bool:
    if imported == EXTENSION and importer not in NATIVE:
        return True
    if in_pc(importer) and not (in_pc(imported) or imported in PUBLIC):
        return True
    return RANK[imported] >= RANK[importer]


class TestLayers:
    def test_every_module_has_a_layer(self):
        assert sources().keys() | {EXTENSION} == RANK.keys()

    def test_imports_go_down(self):
        modules = sources().keys() | {EXTENSION}
        broken = {
            (importer, imported)
            for importer, path in sources().items()
            for imported in imports(path, modules)
            if breaks_the_rule(importer, imported)
        }
        assert broken == set(), "imports that break the layers"

    def test_every_import_is_at_the_top_level(self):
        nested: list[str] = []
        for path in sources().values():
            module = ast.parse(path.read_text(), str(path))
            nested += [
                f"{path}:{node.lineno}"
                for node in ast.walk(module)
                if isinstance(node, ast.Import | ast.ImportFrom)
                and node not in module.body
            ]
        assert nested == []

    def test_pc_may_import_the_public_modules_only(self):
        assert not breaks_the_rule("mujou.pc._layer", "mujou")
        assert not breaks_the_rule("mujou.pc._layer", "mujou.pc.functional")
        assert breaks_the_rule("mujou.pc._layer", "mujou._core.node")
