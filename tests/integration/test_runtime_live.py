"""The real process, started for real.

Everything else stops one layer short of this. The unit tests build
interactors from fakes; the other integration tests connect to Postgres but
assemble nothing. `open_runtime` is the only code path that does what a
container does on boot - read config, load the registry, open pools as the
authenticator, SET ROLE, introspect, verify the GRANTs, check history is
writable, and build the tools - and every one of those steps has failed here
at least once while being written.

What is deliberately not covered: asking a question. That needs a model, and
a test that silently skips without an API key is a test that reports green
for a path nobody ran. The boundary is drawn at the last step before the
model is called.

Requires Postgres and Redis, seeded through seeds/004_history.sql. Skipped
when either is unreachable.
"""

from __future__ import annotations

import os

import pytest

asyncpg = pytest.importorskip("asyncpg")
pytest.importorskip("redis")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

pytestmark = pytest.mark.integration

ENVIRONMENT = {
    "PG_HOST": os.getenv("PGHOST", "localhost"),
    "PG_PORT": os.getenv("PGPORT", "55432"),
    "PG_DBNAME": os.getenv("PGDATABASE", "library_dev"),
    "PG_USER": "app_authenticator",
    "PG_PASSWORD": "dev_authenticator",
    "REDIS_URL": os.getenv("TEST_REDIS_URL", "redis://localhost:56379/1"),
    "AGENTS_REGISTRY_PATH": "seeds/agents.example.json",
    # Never used on any path here. A real key would mean these tests could
    # spend money, which is not a property a test suite should have.
    "QWEN_API_KEY": "unused-no-model-is-called",
}


@pytest.fixture
def environment(monkeypatch):
    import importlib

    from libs.agent_core import config

    for name, value in ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    importlib.reload(config)
    yield config
    monkeypatch.undo()
    importlib.reload(config)


async def runtime_for(environment, agent_key):
    from libs.agent_core.composition import open_runtime

    try:
        return await open_runtime(agent_key)
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Postgres or Redis unavailable: {e}")
        raise


# --------------------------------------------------------------------------
# a sub-agent process
# --------------------------------------------------------------------------


async def test_a_sub_agent_process_starts_and_knows_its_scope(environment):
    runtime = await runtime_for(environment, "catalog")
    try:
        assert runtime.kind == "sub_agent"
        assert runtime.agent.spec.name == "catalog"
        assert set(runtime.allowed_tables_or_empty()) == {
            "authors", "books", "publishers",
        }
    finally:
        await runtime.aclose()


async def test_two_sub_agent_processes_get_different_scopes(environment):
    catalog = await runtime_for(environment, "catalog")
    circulation = await runtime_for(environment, "circulation")
    try:
        a = set(catalog.allowed_tables_or_empty())
        b = set(circulation.allowed_tables_or_empty())
    finally:
        await catalog.aclose()
        await circulation.aclose()

    assert "members" not in a
    assert "authors" not in b


async def test_the_sub_agents_tools_are_bound_to_its_own_tables(environment):
    """The end of the chain. Registry, to role, to introspection, to the
    allowlist the tool enforces - all of it built by the same call a
    container makes."""
    runtime = await runtime_for(environment, "catalog")
    try:
        tools = runtime.turn.agent_loop._tools
        assert tools._allowed_tables == {"authors", "books", "publishers"}

        answer = await tools.call_tool(
            "get_filter", {"columns": ["summary", "genre"], "table_name": "books"},
        )
    finally:
        await runtime.aclose()

    # Classified from the real catalogue, not from anything written down.
    assert "embed_summary" in answer["summary"]
    assert "novel" in answer["genre"]


async def test_a_sub_agent_keeps_no_conversation_history(environment):
    runtime = await runtime_for(environment, "catalog")
    try:
        assert runtime.turn.use_conversation_history is False
        # The shared method first, then this deployment's own prompt.
        assert runtime.turn.system_prompt.startswith("How to work")
        assert "You answer questions about" in runtime.turn.system_prompt
    finally:
        await runtime.aclose()


async def test_an_unknown_agent_key_refuses_to_start(environment):
    """A typo in AGENT_KEY should be a container that will not come up, not
    one that comes up serving nothing."""
    from domain.exceptions import UnknownAgentError

    with pytest.raises(UnknownAgentError):
        await runtime_for(environment, "catalogue")


# --------------------------------------------------------------------------
# the orchestrator process
# --------------------------------------------------------------------------


async def test_the_orchestrator_starts_and_knows_who_it_routes_to(environment):
    runtime = await runtime_for(environment, "")
    try:
        assert runtime.kind == "orchestrator"
        assert set(runtime.routes_to) == {"catalog", "circulation"}
        assert runtime.turn.use_conversation_history is True
    finally:
        await runtime.aclose()


async def test_the_orchestrator_has_no_database_tools(environment):
    """Everything factual it says has to come from an agent's reply."""
    runtime = await runtime_for(environment, "")
    try:
        names = {
            s["function"]["name"]
            for s in runtime.turn.agent_loop._tools.get_tool_schemas()
        }
    finally:
        await runtime.aclose()

    assert names == {"catalog", "circulation"}


# --------------------------------------------------------------------------
# history is checked, not created
# --------------------------------------------------------------------------


async def test_a_missing_history_table_stops_startup_with_an_instruction(environment):
    """The service does not create it - that is DDL. So it says what to run.

    Discovered at startup rather than partway through answering: a history
    write happens after the model has been called and the SQL has run, so the
    failure would otherwise surface as a lost answer rather than a missing
    table.
    """
    import dataclasses

    from adapters.outbound.registry.file_agent_registry_adapter import (
        FileAgentRegistryAdapter,
    )
    from libs.agent_core.composition import open_runtime

    registry = FileAgentRegistryAdapter("seeds/agents.example.json")
    original = registry._agents["catalog"]
    registry._agents["catalog"] = dataclasses.replace(
        original, history_table="history_that_was_never_created"
    )

    import libs.agent_core.composition as composition

    # Prove the services are actually up first. Without this, an unreachable
    # Postgres raises here too and the test passes for the wrong reason -
    # RuntimeError is what a connection failure looks like from outside.
    healthy = await runtime_for(environment, "catalog")
    await healthy.aclose()

    def fake_registry(_path):
        return registry

    real = composition.FileAgentRegistryAdapter
    composition.FileAgentRegistryAdapter = fake_registry
    try:
        with pytest.raises(RuntimeError, match="004_history.sql"):
            await open_runtime("catalog")
    finally:
        composition.FileAgentRegistryAdapter = real


async def test_the_service_can_write_its_history_but_not_an_agents_tables(environment):
    """The privilege split, from the running process's own connection.

    The service holds SELECT and INSERT on history and nothing else. An agent
    role holds SELECT on its tables and nothing else. Neither can do the
    other's job, which is what keeps a bug in one from becoming access in the
    other.
    """
    from domain.exceptions import DatabaseError

    runtime = await runtime_for(environment, "catalog")
    try:
        service_db = runtime.turn.history._db
        await service_db.fetch("SELECT count(*) FROM history_catalog")

        with pytest.raises(DatabaseError):
            await service_db.fetch("SELECT count(*) FROM books")
    finally:
        await runtime.aclose()


# --------------------------------------------------------------------------
# through the HTTP edge, as a container serves it
# --------------------------------------------------------------------------


def test_health_reports_the_scope_a_container_resolved(environment):
    from adapters.inbound.http.app import create_app
    from libs.agent_core.composition import open_runtime

    async def opener():
        return await open_runtime("catalog")

    try:
        with TestClient(create_app(opener)) as client:
            body = client.get("/health").json()
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Postgres or Redis unavailable: {e}")
        raise

    assert body["kind"] == "sub_agent"
    assert body["agent"] == "catalog"
    assert set(body["tables"]) == {"authors", "books", "publishers"}


def test_the_orchestrator_container_refuses_the_sub_agent_endpoint(environment):
    from adapters.inbound.http.app import create_app
    from libs.agent_core.composition import open_runtime

    async def opener():
        return await open_runtime("")

    try:
        with TestClient(create_app(opener)) as client:
            response = client.post("/run", json={"user_input": "x"})
    except Exception as e:
        if "connect" in str(e).lower() or "refused" in str(e).lower():
            pytest.skip(f"Postgres or Redis unavailable: {e}")
        raise

    assert response.status_code == 404
    assert "/ask" in response.json()["detail"]
