"""The dependency rule, asserted rather than remembered.

Hexagonal architecture is a claim about which direction imports point, and a
claim nothing checks is a claim that decays. It decayed here within one
commit: instrumenting the interactor added `from libs.agent_core.logging_setup
import ...` to domain/interactors/run_agent_turn.py, which reads as harmless -
it is only logging - and quietly made the domain unusable without the
composition layer it is supposed to know nothing about.

The rule, from the inside out:

    domain/     imports the standard library and itself. Nothing else.
    libs/       may import domain. Not adapters.
    adapters/   may import domain and libs.

`libs/` not importing `adapters/` is the load-bearing one after the domain
rule: the composition root is the single exception, and it does its adapter
imports inside the function that needs them, which is why open_runtime can be
imported by a test with no asyncpg, redis, httpx or openai installed.

Parsed with `ast` rather than imported, so the check itself never needs the
dependencies it is checking for.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Third-party packages the domain is allowed to name. Only the ones that
# describe data rather than fetch it: an entity may be a pydantic model, but
# nothing in domain/ may open a socket.
DOMAIN_ALLOWED_THIRD_PARTY: frozenset[str] = frozenset()


def python_files(package: str) -> list[pathlib.Path]:
    return sorted(
        path for path in (ROOT / package).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def imported_roots(path: pathlib.Path) -> set[str]:
    """Every top-level module name this file imports, at any depth.

    Includes imports written inside a function, which is where a layering
    violation hides most comfortably - it does not run at import time, so
    nothing notices until the module is used in the environment that lacks it.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relative imports have no module name to check and cannot leave
            # their own package anyway.
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", python_files("domain"), ids=lambda p: p.name)
def test_domain_imports_nothing_outward(path):
    """The rule that makes the domain the stable centre.

    stdlib `logging` is fine - it is the standard library, and `extra=` is an
    interface it defines. The formatter that renders those fields lives in
    libs/, and the domain must not know that it exists.
    """
    forbidden = imported_roots(path) & {"libs", "adapters", "scripts", "seeds", "ui"}
    assert not forbidden, (
        f"{path.relative_to(ROOT)} imports {sorted(forbidden)}. The domain is "
        "the centre: nothing in domain/ may depend on a layer outside it. Log "
        "through stdlib logging with extra=; ids arrive by contextvars."
    )


@pytest.mark.parametrize("path", python_files("domain"), ids=lambda p: p.name)
def test_domain_imports_no_infrastructure_packages(path):
    """No driver reaches the domain, whatever it is called.

    Naming them individually rather than allowlisting: a new one arriving in
    domain/ should fail this test on the day it arrives, and an allowlist that
    has to be updated to add a dependency is an allowlist that gets updated.
    """
    infrastructure = {
        "asyncpg", "psycopg", "psycopg2", "redis", "httpx", "requests",
        "openai", "langgraph", "langchain", "langchain_core", "fastapi",
        "uvicorn", "starlette", "aiohttp", "boto3",
    }
    found = imported_roots(path) & infrastructure
    assert not found, (
        f"{path.relative_to(ROOT)} imports {sorted(found)}. A driver in the "
        "domain means the entity layer cannot be tested without it, and the "
        "port it should be behind is not doing its job."
    )


@pytest.mark.parametrize("path", python_files("libs"), ids=lambda p: p.name)
def test_libs_does_not_import_adapters_at_module_level(path):
    """libs/ may know the domain. It must not know the adapters - except in
    the composition root, which is what a composition root is for, and even
    there only inside the function.

    Module level specifically: `open_runtime` imports asyncpg, redis, httpx
    and openai inside the function body precisely so that importing
    composition.py stays free of them, and a test can exercise assemble_* with
    fakes on a machine where none are installed. Moving one of those to the
    top of the file would pass a normal test run and break that property
    silently.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in tree.body:  # module level only
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.startswith("adapters")]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("adapters"):
            offenders.append(node.module)

    if path.name == "composition.py":
        # The exception, and narrowly: the composition root names adapters
        # because assembling them is its entire job.
        return

    assert not offenders, (
        f"{path.relative_to(ROOT)} imports {sorted(offenders)} at module level. "
        "Only the composition root wires adapters."
    )


def test_composition_root_defers_its_driver_imports():
    """The property that keeps the unit tests dependency-free, asserted
    directly rather than inferred from the fact that they currently pass."""
    path = ROOT / "libs" / "agent_core" / "composition.py"
    tree = ast.parse(path.read_text(), filename=str(path))

    module_level: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_level.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level.add(node.module.split(".")[0])

    drivers = {"asyncpg", "redis", "httpx", "openai"}
    assert not (module_level & drivers), (
        "composition.py imports a driver at module level. open_runtime imports "
        "them inside the function so that assemble_* stays testable without "
        "any of them installed."
    )


def test_the_layering_check_actually_sees_files():
    """A parametrised test over an empty list passes by saying nothing.

    This is the test that fails if the package is moved or renamed, rather
    than the suite quietly going green over zero files.
    """
    assert len(python_files("domain")) >= 10
    assert len(python_files("libs")) >= 5
