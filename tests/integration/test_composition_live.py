"""The authenticator pattern, against a real Postgres.

One login for the whole service, and every agent pool becomes its agent with
`SET ROLE` on each new connection. That is a convenience argument until you
ask what the authenticator can read *before* it becomes anybody - and the
answer has to be "nothing", or the pattern is a hole rather than a boundary.

That answer comes from NOINHERIT in seeds/003_roles.sql. With it, membership
grants the *right to become* a role rather than that role's privileges, so
the floor is empty instead of being the union of every agent's access. Every
test here is ultimately about that one word, because it is the difference
between narrowing and decoration.

Requires the development database, seeded as in the other integration tests.
Skipped when it is not reachable.
"""

from __future__ import annotations

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
from domain.exceptions import DatabaseError  # noqa: E402
from libs.agent_core.agent_startup import start_agent  # noqa: E402
from libs.agent_core.composition import _agent_pool  # noqa: E402

pytestmark = pytest.mark.integration

HOST = os.getenv("PGHOST", "localhost")
PORT = int(os.getenv("PGPORT", "55432"))
DATABASE = os.getenv("PGDATABASE", "library_dev")
AUTHENTICATOR = ("app_authenticator", "dev_authenticator")

LIBRARY_TABLES = {"authors", "publishers", "books", "branches", "members", "loans"}
# The service's own, from seeds/004_history.sql. Not any agent's data.
HISTORY_TABLES = {"history_catalog", "history_circulation", "history_orchestrator"}


async def authenticator_pool(spec=None):
    """A pool logged in as the authenticator - as an agent, if given one."""
    user, password = AUTHENTICATOR
    kwargs = dict(host=HOST, port=PORT, user=user, password=password,
                  database=DATABASE, min_size=1, max_size=2, timeout=3)
    try:
        if spec is None:
            return await asyncpg.create_pool(**kwargs)
        return await _agent_pool(asyncpg, spec, **kwargs)
    except Exception as e:
        pytest.skip(f"development database unavailable as {user}: {e}")


def registry():
    return FileAgentRegistryAdapter("seeds/agents.example.json")


# --------------------------------------------------------------------------
# the floor: what the authenticator holds on its own
# --------------------------------------------------------------------------


async def test_the_authenticator_reaches_no_agent_data_before_it_becomes_anybody():
    """The whole argument in one assertion. If NOINHERIT were dropped from
    003_roles.sql this would return every table in the database, and every
    other test in this file would still pass.

    It is not literally nothing: the authenticator holds SELECT and INSERT on
    the history tables, which are its own and not any agent's. That is the
    one thing it is meant to do without becoming somebody, so the assertion
    is about agent data rather than about an empty list - a stricter claim
    that happened to be true before history existed would have to be relaxed
    the first time it did.
    """
    pool = await authenticator_pool()
    try:
        schema = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))
        visible = set(await schema.list_tables())
    finally:
        await pool.close()

    assert visible & LIBRARY_TABLES == set()
    assert visible <= HISTORY_TABLES


async def test_every_membership_is_granted_without_inheritance():
    """Asserted from the catalogue, not only from behaviour.

    Since PostgreSQL 16 each membership carries its own inherit_option, taken
    from the member role's NOINHERIT at the moment of the GRANT. So the
    property above depends on the order of two lines in 003_roles.sql, and a
    later agent added with a plain `GRANT x TO app_authenticator` would get an
    inheriting membership with nothing in the SQL looking wrong.

    The behavioural tests here would still catch that - but only for an agent
    whose tables some other test happens to name. This catches it for any
    agent, including one added tomorrow.
    """
    pool = await authenticator_pool()
    try:
        rows = await PostgresDatabaseAdapter(pool).fetch("""
            SELECT r.rolname, m.inherit_option
            FROM pg_auth_members m
            JOIN pg_roles r ON r.oid = m.roleid
            WHERE m.member = (SELECT oid FROM pg_roles WHERE rolname = current_user)
        """)
    finally:
        await pool.close()

    assert rows, "the authenticator is a member of no agent role at all"
    inheriting = [r["rolname"] for r in rows if r["inherit_option"]]
    assert not inheriting, (
        f"{inheriting} are granted WITH INHERIT, so the authenticator holds "
        "their privileges before SET ROLE narrows anything"
    )


async def test_the_authenticator_cannot_select_from_a_table_it_has_not_become():
    pool = await authenticator_pool()
    try:
        with pytest.raises(DatabaseError):
            await PostgresDatabaseAdapter(pool).fetch("SELECT count(*) FROM books")
    finally:
        await pool.close()


# --------------------------------------------------------------------------
# a pool that has already become its agent
# --------------------------------------------------------------------------


async def test_an_agent_pool_sees_exactly_that_agents_tables():
    spec = await registry().get("catalog")
    pool = await authenticator_pool(spec)
    try:
        schema = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))
        assert set(await schema.list_tables()) == {"authors", "books", "publishers"}
    finally:
        await pool.close()


async def test_a_different_agents_pool_sees_a_different_set():
    spec = await registry().get("circulation")
    pool = await authenticator_pool(spec)
    try:
        schema = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))
        assert set(await schema.list_tables()) == {
            "books", "branches", "loans", "members",
        }
    finally:
        await pool.close()


async def test_the_role_is_set_on_every_connection_not_just_the_first():
    """`init` runs per connection and the pool is dedicated to one agent, so
    there is no connection in it serving a different role. Asserted by
    forcing several concurrent acquisitions."""
    import asyncio

    spec = await registry().get("catalog")
    pool = await authenticator_pool(spec)
    try:
        async def current_role():
            async with pool.acquire() as connection:
                await asyncio.sleep(0.01)   # hold it, so the next one opens another
                return await connection.fetchval("SELECT current_user")

        assert set(await asyncio.gather(*(current_role() for _ in range(4)))) == {
            "app_catalog",
        }
    finally:
        await pool.close()


async def test_an_agent_pool_still_cannot_reach_another_agents_table():
    """SET ROLE narrowed it, and the database enforces the narrowing - not
    the validator, which is application code and has bugs."""
    spec = await registry().get("catalog")
    pool = await authenticator_pool(spec)
    try:
        with pytest.raises(DatabaseError):
            await PostgresDatabaseAdapter(pool).fetch("SELECT count(*) FROM members")
    finally:
        await pool.close()


async def test_resetting_the_role_lands_on_nothing_rather_than_everything():
    """The failure mode NOINHERIT rules out. If the authenticator inherited,
    RESET ROLE would be an escalation to the union of every agent's access;
    here it is a descent to the authenticator's own floor, which holds the
    history tables and no agent data at all."""
    spec = await registry().get("catalog")
    pool = await authenticator_pool(spec)
    try:
        async with pool.acquire() as connection:
            await connection.execute("RESET ROLE")
            visible = await connection.fetchval(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        assert visible == len(HISTORY_TABLES)
    finally:
        await pool.close()


# --------------------------------------------------------------------------
# startup, through the pool the composition root actually builds
# --------------------------------------------------------------------------


async def test_startup_verification_passes_through_an_authenticator_pool():
    """The unit tests hand start_agent a port connected however the test
    liked. This is the connection the composition root really builds."""
    for spec in await registry().list_active():
        pool = await authenticator_pool(spec)
        try:
            ready = await start_agent(
                spec, PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool)),
            )
            assert set(ready.allowed_tables) == set(spec.tables)
        finally:
            await pool.close()


async def test_two_agents_pools_do_not_leak_into_each_other():
    """Both log in as the same user. If SET ROLE were per-session in a way
    the pool did not respect, these two would converge on one scope."""
    catalog = await registry().get("catalog")
    circulation = await registry().get("circulation")

    catalog_pool = await authenticator_pool(catalog)
    circulation_pool = await authenticator_pool(circulation)
    try:
        a = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(catalog_pool))
        b = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(circulation_pool))

        assert "members" not in set(await a.list_tables())
        assert "authors" not in set(await b.list_tables())
    finally:
        await catalog_pool.close()
        await circulation_pool.close()
