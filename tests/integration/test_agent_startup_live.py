"""Startup against a real Postgres, through real agent roles.

The unit tests use a fake that reports what a role was granted. That fake
encodes an assumption - that connecting as `app_catalog` shows three tables
and connecting as `app_circulation` shows four - and an assumption is exactly
what a fake cannot check. This file connects as those roles and reads the
answer out of the database.

It is also the only place where the two halves of a deployment are compared
against each other for real: seeds/agents.example.json says which tables each
agent should have, seeds/003_roles.sql grants them, and nothing but this
test notices if somebody edits one and not the other.

Requires the development database:

    docker compose -f deploy/docker-compose.dev.yml up -d
    docker exec -i multi_agents_dev_db psql -U dev -d library_dev < seeds/001_schema.sql
    python3 seeds/002_generate_data.py | docker exec -i multi_agents_dev_db psql -U dev -d library_dev
    docker exec -i multi_agents_dev_db psql -U dev -d library_dev < seeds/003_roles.sql

Skipped when it is not reachable, so `pytest` stays green without it.
"""

from __future__ import annotations

import dataclasses
import os

import pytest

asyncpg = pytest.importorskip("asyncpg")

from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter  # noqa: E402
from adapters.outbound.registry.file_agent_registry_adapter import (  # noqa: E402
    FileAgentRegistryAdapter,
)
from adapters.outbound.schema.postgres_introspection_adapter import (  # noqa: E402
    PostgresIntrospectionAdapter,
)
from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter  # noqa: E402
from domain.exceptions import DatabaseError, GrantMismatchError  # noqa: E402
from libs.agent_core.agent_startup import start_agent, start_all  # noqa: E402

pytestmark = pytest.mark.integration

HOST = os.getenv("PGHOST", "localhost")
PORT = int(os.getenv("PGPORT", "55432"))
DATABASE = os.getenv("PGDATABASE", "library_dev")
REGISTRY = "seeds/agents.example.json"

# From seeds/003_roles.sql. Development passwords; a real deployment's roles
# are created by the provisioner with credentials it generates.
PASSWORDS = {"app_catalog": "dev_catalog", "app_circulation": "dev_circulation"}


async def pool_as(role: str):
    try:
        return await asyncpg.create_pool(
            host=HOST, port=PORT, user=role, password=PASSWORDS[role],
            database=DATABASE, min_size=1, max_size=2, timeout=3,
        )
    except Exception as e:
        pytest.skip(f"development database unavailable as {role}: {e}")


async def port_for(spec):
    """A SchemaPort connected as the agent's own role.

    This is the line the whole design turns on. Connect as `dev` here instead
    and every test in this file still passes while the property it claims to
    check is gone - which is why the roles, not an admin user, are what the
    fixtures use.
    """
    pool = await pool_as(spec.db_role)
    return PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool)), pool


def registry():
    return FileAgentRegistryAdapter(REGISTRY)


# --------------------------------------------------------------------------
# the registry and the GRANTs agree
# --------------------------------------------------------------------------


async def test_the_shipped_registry_starts_against_the_shipped_roles():
    """agents.example.json and 003_roles.sql are one deployment written
    twice. If they ever disagree, this is what says so."""
    for spec in await registry().list_active():
        schema, pool = await port_for(spec)
        try:
            ready = await start_agent(spec, schema)
            assert set(ready.allowed_tables) == set(spec.tables)
        finally:
            await pool.close()


async def test_each_agent_sees_only_its_own_tables():
    catalog = await registry().get("catalog")
    circulation = await registry().get("circulation")

    scopes = {}
    for spec in (catalog, circulation):
        schema, pool = await port_for(spec)
        try:
            scopes[spec.name] = set((await start_agent(spec, schema)).allowed_tables)
        finally:
            await pool.close()

    assert scopes["catalog"] == {"authors", "publishers", "books"}
    assert scopes["circulation"] == {"books", "branches", "loans", "members"}

    # The overlap is real and deliberate - circulation reads book titles to
    # name what was borrowed - so an empty intersection would be the wrong
    # assertion. What matters is what each does *not* get.
    assert "members" not in scopes["catalog"]
    assert "authors" not in scopes["circulation"]


async def test_all_agents_come_up_together():
    ready = await start_all(registry(), lambda spec: _port_only(spec))
    assert sorted(r.spec.name for r in ready) == ["catalog", "circulation"]


async def _port_only(spec):
    """start_all's factory returns a port, not a pool, so these leak a
    connection until the event loop closes. Acceptable in a test; a real
    composition root keeps the pools and closes them in its lifespan."""
    schema, _pool = await port_for(spec)
    return schema


# --------------------------------------------------------------------------
# drift is caught, in both directions
# --------------------------------------------------------------------------


async def test_a_registry_that_understates_an_agent_is_refused():
    """The dangerous direction, against a real role: the file says two
    tables, the GRANTs give three. Nobody reading the registry would know
    the agent can also read `publishers`."""
    spec = await registry().get("catalog")
    narrowed = dataclasses.replace(spec, tables=["authors", "books"])

    schema, pool = await port_for(spec)
    try:
        with pytest.raises(GrantMismatchError, match="publishers"):
            await start_agent(narrowed, schema)
    finally:
        await pool.close()


async def test_a_registry_promising_an_ungranted_table_is_refused():
    """The other direction: the file lists a table the role cannot read, so
    the agent would fail partway through somebody's question."""
    spec = await registry().get("catalog")
    widened = dataclasses.replace(spec, tables=[*spec.tables, "members"])

    schema, pool = await port_for(spec)
    try:
        with pytest.raises(GrantMismatchError, match="members"):
            await start_agent(widened, schema)
    finally:
        await pool.close()


async def test_an_agent_declaring_nothing_takes_what_the_role_grants():
    spec = await registry().get("circulation")
    undeclared = dataclasses.replace(spec, tables=None)

    schema, pool = await port_for(spec)
    try:
        ready = await start_agent(undeclared, schema)
        assert set(ready.allowed_tables) == {"books", "branches", "loans", "members"}
    finally:
        await pool.close()


# --------------------------------------------------------------------------
# what startup produced actually drives the agent's tools
# --------------------------------------------------------------------------


async def test_the_loaded_schema_classifies_the_real_columns():
    spec = await registry().get("catalog")
    schema, pool = await port_for(spec)
    try:
        ready = await start_agent(spec, schema)
        kinds = {
            name: cf.kind.name
            for name, cf in ready.schema.classified["books"].items()
        }
    finally:
        await pool.close()

    assert kinds["summary"] == "SEMANTIC"        # embed_summary exists
    assert kinds["embed_summary"] == "VECTOR_STORAGE"
    assert kinds["page_count"] == "OPERATOR"
    assert kinds["added_at"] == "DATETIME"
    assert kinds["genre"] == "ENUM"              # 10 distinct, under the cutoff
    assert kinds["isbn"] == "TEXT"               # 420 distinct, over it


async def test_startup_output_drives_a_real_sql_tool_adapter():
    """End to end, no glue: registry -> role connection -> verification ->
    classification -> the adapter the model calls."""
    spec = await registry().get("catalog")
    schema, pool = await port_for(spec)
    try:
        ready = await start_agent(spec, schema)
        adapter = SqlToolAdapter(
            db=PostgresDatabaseAdapter(pool), embeddings=None, cache=None,
            allowed_tables=list(ready.allowed_tables),
            schema=ready.schema.schema,
            filters=ready.schema.filters,
            lsit_values={}, dist_op="<=>", vector_ttl_seconds=900,
        )

        guidance = await adapter.call_tool(
            "get_filter", {"columns": ["Genre", "summary"], "table_name": "BOOKS"}
        )
        rows = await adapter.call_tool("db_execute", {
            "query": "SELECT title_en FROM books WHERE genre = $1 LIMIT $2 OFFSET $3",
            "params": ["Novel", 3, 0], "offset": 0,
            "count_query": "SELECT count(*) FROM books WHERE genre = $1",
            "count_params": ["Novel"],
        })
    finally:
        await pool.close()

    # Case folded on both the table and the column, which is the fix that
    # came with feature 2.
    assert "get_lsit_values" in guidance["Genre"]
    assert "embed_summary" in guidance["summary"]
    assert "rows" in rows


async def test_the_agent_still_cannot_reach_another_agents_table():
    """Startup narrows what the model is told about. The database is what
    stops it anyway - the layer that holds when the validator is wrong."""
    spec = await registry().get("catalog")
    pool = await pool_as(spec.db_role)
    try:
        with pytest.raises(DatabaseError):
            await PostgresDatabaseAdapter(pool).fetch("SELECT count(*) FROM members")
    finally:
        await pool.close()
