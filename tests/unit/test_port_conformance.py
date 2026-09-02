"""Every adapter still fits the port it is meant to satisfy.

The ports are `typing.Protocol`, which is structural and checked by a type
checker rather than at runtime. Nothing in a normal test run notices when an
adapter drifts: a renamed method, a parameter that changed name, a port that
grew a method nobody implemented. The failure arrives later, as an
AttributeError inside a request, in a stack trace pointing at the caller
rather than at the mismatch.

So this compares them directly, by name and by signature. It is the cheapest
test in the suite and the one most likely to catch a rename.

It also imports every module in the project, which is worth more than it
sounds: `domain/ports/agent_registry_port.py` spent its whole life importing
a class that did not exist, and nothing caught it because nothing imported
the module.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

from adapters.outbound.registry.file_agent_registry_adapter import (
    FileAgentRegistryAdapter,
)
from adapters.outbound.schema.postgres_introspection_adapter import (
    PostgresIntrospectionAdapter,
)
from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter
from domain.ports.agent_registry_port import AgentRegistryPort
from domain.ports.schema_port import SchemaPort

# Ports whose bodies are still `...` describe their methods in prose rather
# than in code, so there is nothing to compare a signature against yet. They
# are listed here rather than skipped silently, so finishing one is visible
# as a line to delete.
PORTS_STILL_DESCRIBED_IN_PROSE = set()

PAIRS = [
    (SchemaPort, PostgresIntrospectionAdapter),
    (AgentRegistryPort, FileAgentRegistryAdapter),
]


def port_methods(port) -> dict:
    return {
        name: value for name, value in vars(port).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }


@pytest.mark.parametrize("port,adapter", PAIRS, ids=lambda x: x.__name__)
def test_the_adapter_implements_every_method_the_port_declares(port, adapter):
    for name in port_methods(port):
        assert hasattr(adapter, name), (
            f"{adapter.__name__} is missing {name}(), which {port.__name__} declares"
        )


@pytest.mark.parametrize("port,adapter", PAIRS, ids=lambda x: x.__name__)
def test_the_signatures_match(port, adapter):
    """Parameter names, not just arity. Everything here is called with
    keywords somewhere, so a renamed parameter is as breaking as a missing
    one and just as invisible."""
    for name, declared in port_methods(port).items():
        implemented = getattr(adapter, name)
        expected = list(inspect.signature(declared).parameters)
        actual = list(inspect.signature(implemented).parameters)
        assert expected == actual, f"{adapter.__name__}.{name}{tuple(actual)}"


@pytest.mark.parametrize("port,adapter", PAIRS, ids=lambda x: x.__name__)
def test_async_methods_stay_async(port, adapter):
    """A port method that stops being a coroutine breaks every caller's
    await, and only at the moment it is called."""
    for name, declared in port_methods(port).items():
        if inspect.iscoroutinefunction(declared):
            assert inspect.iscoroutinefunction(getattr(adapter, name)), (
                f"{adapter.__name__}.{name} is not async but {port.__name__} says it is"
            )


def test_a_port_with_no_methods_is_flagged_rather_than_passing_silently():
    """A Protocol whose body is `...` accepts anything, so the tests above
    would pass against an empty class. Naming those ports keeps that state
    deliberate."""
    for port, _ in PAIRS:
        if not port_methods(port):
            assert port in PORTS_STILL_DESCRIBED_IN_PROSE, (
                f"{port.__name__} declares no methods and is not listed as "
                "still-in-prose. Either it lost them, or the list is stale."
            )


def test_the_sql_tool_adapter_declares_every_tool_it_dispatches():
    """The one adapter that is not behind a Protocol. Its contract is the
    tool schemas it hands the model against the handlers it dispatches on -
    a name in one and not the other is an UnknownToolError mid-answer."""
    adapter = SqlToolAdapter(
        db=None, embeddings=None, cache=None, allowed_tables=[], schema={},
        filters={}, dist_op="<=>", vector_ttl_seconds=900,
    )
    declared = {s["function"]["name"] for s in adapter.get_tool_schemas()}
    assert declared <= set(adapter._handlers)


# --------------------------------------------------------------------------
# everything imports
# --------------------------------------------------------------------------

# Needs a package that is not installed here, and installing a Redis client
# to prove a module parses is not a trade worth making.
NEEDS_OPTIONAL_DEPENDENCIES = {"adapters.outbound.cache.redis_cache_adapter"}


def project_modules():
    import adapters
    import domain
    import libs

    for package in (domain, adapters, libs):
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if not info.ispkg:
                yield info.name


@pytest.mark.parametrize("module", sorted(project_modules()))
def test_every_module_imports(module):
    """Cheap, and it catches the class of bug that only appears the first
    time somebody imports a file - a wrong module path in an import, a name
    that was renamed in one place."""
    if module in NEEDS_OPTIONAL_DEPENDENCIES:
        pytest.importorskip(module.rsplit(".", 1)[-1].split("_")[0])
    importlib.import_module(module)
