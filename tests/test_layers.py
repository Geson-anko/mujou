"""The import layers of ``src/mujou``: the source of truth for the rule.

A module imports only modules of lower layers (not of its own), and only the
modules in ``NATIVE`` import the extension ``mujou._core._native``. Every import
statement counts, wherever it is: at the top, under ``TYPE_CHECKING`` or in a
function. ``ALLOWED`` lists the imports that break the rule today. It only
shrinks: an import that breaks the rule and is not in it fails, and so does an
entry that no longer breaks it.
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
    {"mujou._core.binding"},
    {"mujou._core.component"},
    {"mujou._core.node"},
    {"mujou._core.driver"},
    {"mujou._core.network"},
    {"mujou._core.graph"},
    {"mujou._core.state", "mujou._core.lifetime"},
    {"mujou._core.session"},
    {"mujou._core.run"},
    {"mujou", "mujou.time", "mujou.testing", "mujou.typing"},
    {
        "mujou._core.pc",
        "mujou._core.pc.messages",
        "mujou._core.pc.losses",
        "mujou._core.pc.time_constants",
    },
    {"mujou._core.pc.functional"},
    {"mujou._core.pc.learner"},
    {"mujou._core.pc.input", "mujou._core.pc.layer", "mujou._core.pc.output"},
    {"mujou.pc.functional"},
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

# (importer, imported)
ALLOWED = {
    ("mujou", "mujou.time"),
    ("mujou", "mujou.typing"),
    ("mujou._core.binding", EXTENSION),
    ("mujou._core.binding", "mujou._core.node"),
    ("mujou._core.component", "mujou._core.graph"),
    ("mujou._core.component", "mujou._core.network"),
    ("mujou._core.component", "mujou._core.node"),
    ("mujou._core.network", "mujou._core.graph"),
    ("mujou._core.port", "mujou._core.binding"),
    ("mujou._core.session", EXTENSION),
}


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


def breaks_the_rule(importer: str, imported: str) -> bool:
    if imported == EXTENSION and importer not in NATIVE:
        return True
    return RANK[imported] >= RANK[importer]


class TestLayers:
    def test_every_module_has_a_layer(self):
        assert sources().keys() | {EXTENSION} == RANK.keys()

    def test_imports_go_down_but_the_allowed_ones(self):
        modules = sources().keys() | {EXTENSION}
        broken = {
            (importer, imported)
            for importer, path in sources().items()
            for imported in imports(path, modules)
            if breaks_the_rule(importer, imported)
        }
        assert broken - ALLOWED == set(), "imports that break the layers"
        assert ALLOWED - broken == set(), "fixed: remove them from ALLOWED"
