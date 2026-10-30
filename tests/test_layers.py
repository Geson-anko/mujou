"""The import layers of ``src/mujou``: the source of truth for the rule.

A module imports only modules of lower layers (not of its own), and only the
modules in ``NATIVE`` import the extension ``mujou._native``. ``mujou.pc``
is written as a user's code is: on the documented entries in ``PUBLIC`` only,
never on the modules behind them (``mujou.node``, ``mujou.port``, ...). Every import is at the top level of its module: none under
``TYPE_CHECKING`` or in a function, which would hide a dependency. A name is
imported from the module that defines it, not through another that imported
it (except the modules in ``REEXPORTING``), which would hide the dependency
too: the modules of ``mujou`` take the timers from the extension, and
``mujou.testing`` from ``mujou.time``, which re-exports them.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
EXTENSION = "mujou._native"

# From the bottom up.
LAYERS: list[set[str]] = [
    {EXTENSION},
    {"mujou.errors", "mujou.types"},
    {"mujou.spec", "mujou.current"},
    {"mujou.codec"},
    {"mujou.port"},
    {"mujou.component"},
    {"mujou.node"},
    {"mujou.network"},
    {"mujou.driver"},
    {"mujou.tree"},
    {"mujou.wiring"},
    {"mujou.state"},
    {"mujou.execution"},
    {"mujou.time"},
    {"mujou", "mujou.testing", "mujou.viz"},
    {
        "mujou.pc.msg",
        "mujou.pc.optim",
        "mujou.pc.relax",
        "mujou.pc.time_constants",
    },
    {"mujou.pc.clock", "mujou.pc.functional"},
    {"mujou.pc.input", "mujou.pc.layer", "mujou.pc.output", "mujou.pc.loss"},
    {"mujou.pc.chain"},
    {"mujou.pc"},
]
RANK = {module: rank for rank, layer in enumerate(LAYERS) for module in layer}

NATIVE = {
    "mujou.types",
    "mujou.spec",
    "mujou.current",
    "mujou.codec",
    "mujou.port",
    "mujou.node",
    "mujou.driver",
    "mujou.wiring",
    "mujou.execution",
    "mujou.time",
}

PUBLIC = {"mujou", "mujou.time", "mujou.testing", "mujou.types"}
# The modules that re-export names of other modules: those of PUBLIC and the
# entry of mujou.pc. Every other module is imported only for what it defines.
REEXPORTING = PUBLIC | {"mujou.pc"}


def module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def sources() -> dict[str, Path]:
    return {module_name(path): path for path in (SRC / "mujou").rglob("*.py")}


def imports(path: Path, modules: set[str]) -> Iterator[str]:
    """The modules of ``modules`` that ``path`` imports."""
    for node in ast.walk(ast.parse(path.read_text("utf-8"), str(path))):
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


def defined(path: Path) -> set[str]:
    """The names that the module at ``path`` defines at its top level."""
    names: set[str] = set()
    for node in ast.parse(path.read_text("utf-8"), str(path)).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.TypeAlias):
            names.add(node.name.id if isinstance(node, ast.TypeAlias) else node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


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
            module = ast.parse(path.read_text("utf-8"), str(path))
            nested += [
                f"{path}:{node.lineno}"
                for node in ast.walk(module)
                if isinstance(node, ast.Import | ast.ImportFrom)
                and node not in module.body
            ]
        assert nested == []

    def test_names_come_from_the_module_that_defines_them(self):
        modules = sources()
        borrowed = [
            f"{importer}: {node.module}.{alias.name}"
            for importer, path in modules.items()
            for node in ast.walk(ast.parse(path.read_text("utf-8"), str(path)))
            if isinstance(node, ast.ImportFrom)
            and node.module in modules
            and node.module not in REEXPORTING
            for alias in node.names
            if f"{node.module}.{alias.name}" not in modules.keys() | {EXTENSION}
            and alias.name not in defined(modules[node.module])
        ]
        assert borrowed == []

    def test_pc_may_import_the_public_modules_only(self):
        assert not breaks_the_rule("mujou.pc.layer", "mujou")
        assert not breaks_the_rule("mujou.pc.layer", "mujou.pc.functional")
        assert breaks_the_rule("mujou.pc.layer", "mujou.node")
