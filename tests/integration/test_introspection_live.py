"""Introspection against a real Postgres.

The unit tests check the translation - which query is sent, how rows become
entities. They cannot check the assumption underneath it: that
information_schema reports what this design expects it to, and only what the
connected role may read. That assumption is the whole security argument, so
it is worth a test that actually connects.

Requires the development database:

    docker compose -f deploy/docker-compose.dev.yml up -d
    docker exec -i multi_agents_dev_db psql -U dev -d library_dev < seeds/001_schema.sql
    python3 seeds/002_generate_data.py | docker exec -i multi_agents_dev_db psql -U dev -d library_dev
    docker exec -i multi_agents_dev_db psql -U dev -d library_dev < seeds/003_roles.sql

Skipped when it is not reachable, so `pytest` stays green without it. Point
elsewhere with PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE.
"""

from __future__ import annotations

import os

import pytest

asyncpg = pytest.importorskip("asyncpg")

from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter  # noqa: E402
from adapters.outbound.schema.postgres_introspection_adapter import (  # noqa: E402
    PostgresIntrospectionAdapter,
)

pytestmark = pytest.mark.integration

HOST = os.getenv("PGHOST", "localhost")
PORT = int(os.getenv("PGPORT", "55432"))
DATABASE = os.getenv("PGDATABASE", "library_dev")
ADMIN = (os.getenv("PGUSER", "dev"), os.getenv("PGPASSWORD", "dev"))

# From seeds/003_roles.sql. Development passwords; a real deployment's roles
# are created by the provisioner with credentials it generates.
CATALOG = ("app_catalog", "dev_catalog")
CIRCULATION = ("app_circulation", "dev_circulation")


async def connect(credentials):
    user, password = credentials
    try:
        return await asyncpg.create_pool(
            host=HOST, port=PORT, user=user, password=password,
            database=DATABASE, min_size=1, max_size=2, timeout=3,
        )
    except Exception as e:  # not running, not seeded, or roles not created
        pytest.skip(f"development database unavailable as {user}: {e}")


async def introspect(credentials=ADMIN):
    pool = await connect(credentials)
    return pool, PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))


# --------------------------------------------------------------------------
# the schema is read, not declared
# --------------------------------------------------------------------------


async def test_lists_the_development_tables():
    """The library tables. The history tables from seeds/004_history.sql are
    also present for an administrative connection, and are excluded here
    rather than asserted on - they are the service's, not the schema under
    test, and an agent role never sees them at all."""
    pool, schema = await introspect()
    try:
        listed = set(await schema.list_tables())
    finally:
        await pool.close()

    assert {t for t in listed if not t.startswith("history_")} == {
        "authors", "publishers", "books", "branches", "members", "loans",
    }


async def test_describes_books_with_its_real_columns():
    pool, schema = await introspect()
    try:
        books = (await schema.describe(("books",)))["books"]

        assert books.has_column("title_en")
        assert books.has_column("page_count")
        assert books.column("page_count").sql_type == "integer"
        assert books.column("added_at").sql_type == "timestamp with time zone"
        assert books.column("price").sql_type == "numeric"
    finally:
        await pool.close()


async def test_vector_columns_are_recognised_in_a_real_database():
    """The claim that survives contact with pgvector: an extension type
    reports data_type='USER-DEFINED', and only udt_name identifies it."""
    pool, schema = await introspect()
    try:
        books = (await schema.describe(("books",)))["books"]

        assert books.column("embed_summary").is_vector is True
        assert books.column("embed_summary").sql_type == "vector"
        assert books.column("summary").is_vector is False
    finally:
        await pool.close()


async def test_embedding_partners_are_discovered_from_the_real_schema():
    """No list of searchable columns anywhere - the pairing is in the
    database and is read out of it."""
    pool, schema = await introspect()
    try:
        books = (await schema.describe(("books",)))["books"]

        assert books.embedding_partner("title_en").name == "embed_title_en"
        assert books.embedding_partner("title_ar").name == "embed_title_ar"
        assert books.embedding_partner("summary").name == "embed_summary"
        assert books.embedding_partner("genre") is None
        assert books.embedding_partner("page_count") is None
    finally:
        await pool.close()


async def test_several_tables_come_back_in_one_call():
    pool, schema = await introspect()
    try:
        described = await schema.describe(("books", "authors", "loans"))
        assert set(described) == {"books", "authors", "loans"}
        assert described["loans"].column("borrowed_at").sql_type == "timestamp with time zone"
    finally:
        await pool.close()


# --------------------------------------------------------------------------
# distinct counts, on real data
# --------------------------------------------------------------------------


async def test_distinct_counts_match_the_seeded_data():
    """Generation is deterministic, so these are exact rather than
    approximate - and they are the numbers the ENUM cutoff is chosen
    against."""
    pool, schema = await introspect()
    try:
        assert await schema.distinct_count("books", "genre") == 10
        assert await schema.distinct_count("books", "language") == 3
        assert await schema.distinct_count("books", "isbn") == 420
        assert await schema.distinct_count("members", "membership_tier") == 3
        assert await schema.distinct_count("loans", "status") == 3
    finally:
        await pool.close()


# --------------------------------------------------------------------------
# the security property - the reason the design works
# --------------------------------------------------------------------------


async def test_an_agent_role_sees_only_its_own_tables():
    """information_schema reports only what the connected role holds a
    privilege on. This is what makes the GRANTs the single source of truth
    for an agent's scope - there is no table list in code to drift from
    them."""
    pool, schema = await introspect(CATALOG)
    try:
        assert set(await schema.list_tables()) == {"authors", "books", "publishers"}
    finally:
        await pool.close()


async def test_a_different_agent_role_sees_a_different_set():
    pool, schema = await introspect(CIRCULATION)
    try:
        assert set(await schema.list_tables()) == {"books", "branches", "loans", "members"}
    finally:
        await pool.close()


async def test_describing_a_table_outside_the_role_returns_nothing_for_it():
    """Asking is not an error - the answer is simply that it is not there.
    The caller sees the absence and can say so."""
    pool, schema = await introspect(CATALOG)
    try:
        described = await schema.describe(("books", "members", "loans"))

        assert "books" in described
        assert "members" not in described
        assert "loans" not in described
    finally:
        await pool.close()


async def test_reading_another_agents_table_is_refused_by_the_database():
    """The layer beneath the SQL validator: even with the validator wrong,
    the role cannot read this."""
    from domain.exceptions import DatabaseError

    pool = await connect(CATALOG)
    db = PostgresDatabaseAdapter(pool)
    try:
        with pytest.raises(DatabaseError):
            await db.fetch("SELECT count(*) FROM members")
    finally:
        await pool.close()
