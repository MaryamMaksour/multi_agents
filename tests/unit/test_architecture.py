"""The dependency rule, enforced rather than aspirational.

    domain/   entities, ports, one interactor. Imports the stdlib and itself.
    libs/     classification, prompts, the composition root. May import domain.
    adapters/ Postgres, Redis, the model, HTTP. May import both.

Parsed rather than imported, for two reasons. An import test only sees the
modules it manages to import, so a violation inside a module that needs
asyncpg installed would pass by being skipped. And parsing sees imports
written inside functions, which is where a violation hides most comfortably -
`from adapters...` on line 400 of a method reads as a circular-import
workaround rather than as an architecture change.

This test earned itself immediately: adding logging to the interactor
introduced `from libs...` into domain/, which reads as harmless - it is only
logging - and quietly made the core unusable without the layer it is supposed
to know nothing about.

The composition root is the one exception, and only for third-party packages:
it imports asyncpg, redis, httpx and openai inside the functions that need
them, so the wiring layer stays importable on a machine with none of them
installed.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# What each layer may import, of our own top-level packages. Anything not
# listed is a third-party or stdlib import, which no layer is restricted in.
ALLOWED: dict[str, frozenset[str]] = {
    "domain": frozenset({"domain"}),
    "libs": frozenset({"domain", "libs"}),
    "adapters": frozenset({"domain", "libs", "adapters"}),
}

OUR_PACKAGES = frozenset(ALLOWED) | {"ui", "scripts", "tests"}

# The composition root, and the only file in libs/ allowed to name an
# adapter. Wiring concrete implementations to ports is what it is for; every
# other file in libs/ that imported one would be a layer deciding its own
# infrastructure.
COMPOSITION_ROOT = Path("libs/agent_core/composition.py")


def source_files(package: str) -> list[Path]:
    return sorted(p for p in (ROOT / package).rglob("*.py") if "__pycache__" not in p.parts)


def imported_roots(path: Path) -> set[tuple[str, int]]:
    """Every top-level package this file imports, with the line it is on.

    Relative imports resolve to the file's own package, which is always
    allowed and never interesting here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[tuple[str, int]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                found.add((node.module.split(".")[0], node.lineno))

    return found


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_a_layer_imports_only_what_it_is_allowed_to(layer: str):
    allowed = ALLOWED[layer]
    violations = []

    for path in source_files(layer):
        if path.relative_to(ROOT) == COMPOSITION_ROOT:
            continue
        for root, lineno in imported_roots(path):
            if root in OUR_PACKAGES and root not in allowed:
                violations.append(
                    f"{path.relative_to(ROOT)}:{lineno} imports {root}"
                )

    assert not violations, (
        f"{layer}/ may import {sorted(allowed)} and nothing else of ours:\n  "
        + "\n  ".join(sorted(violations))
    )


def test_the_domain_imports_no_third_party_package():
    """The core is stdlib and itself.

    Not a stylistic preference: a domain that imports pydantic or asyncpg is
    a domain whose entities are shaped by a library, and the port boundary
    stops being a boundary.
    """
    stdlib_or_ours = OUR_PACKAGES | sys.stdlib_module_names | {"__future__"}
    violations = [
        f"{path.relative_to(ROOT)}:{lineno} imports {root}"
        for path in source_files("domain")
        for root, lineno in imported_roots(path)
        if root not in stdlib_or_ours
    ]
    assert not violations, "domain/ imports a third-party package:\n  " + "\n  ".join(
        sorted(violations)
    )


def test_the_rule_would_actually_catch_a_violation(tmp_path: Path):
    """The test above passes on an empty set of files too. This one shows the
    parser sees an import written inside a function, which is the case the
    whole exercise is about."""
    offender = tmp_path / "sneaky.py"
    offender.write_text(
        "def build():\n"
        "    from adapters.outbound.db.postgres_db_adapter import X\n"
        "    return X\n",
        encoding="utf-8",
    )
    assert ("adapters", 2) in imported_roots(offender)
