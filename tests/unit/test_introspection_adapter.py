"""PostgresIntrospectionAdapter, against a fake database.

What is checked here is the translation: which query would be sent, and how
the rows that come back become entities. Whether the query returns the right
thing from a real Postgres is a different question, answered by
tests/integration/test_introspection_live.py.

The pairing of the two matters. These run in milliseconds on every save; the
live ones need a database and confirm the assumptions these are built on.
"""

from __future__ import annotations

import pytest

from adapters.outbound.schema.postgres_introspection_adapter import PostgresIntrospectionAdapter
from domain.entities.table_schema import TableSchema
from domain.exceptions import DatabaseError

from tests.conftest import FakeDatabase


def column_row(table, name, data_type, udt=None, nullable="YES"):
    """A row shaped like information_schema.columns returns."""
    return {
        "table_name": table, "column_name": name,
        "data_type": data_type, "udt_name": udt or data_type,
        "is_nullable": nullable,
    }


def adapter(rows=None, schema="public"):
    return PostgresIntrospectionAdapter(FakeDatabase(rows), schema=schema)


# --------------------------------------------------------------------------
# list_tables
# --------------------------------------------------------------------------


async def test_lists_the_tables_the_connection_can_read():
    db = FakeDatabase([[{"table_name": "books"}, {"table_name": "authors"}]])
    assert await PostgresIntrospectionAdapter(db).list_tables() == ("books", "authors")


async def test_list_tables_is_scoped_to_the_configured_schema():
    """Not hardcoded to public - a deployment may keep its tables elsewhere,
    and that assumption is invisible until it is wrong."""
    db = FakeDatabase([[]])
    await PostgresIntrospectionAdapter(db, schema="library").list_tables()

    query, params = db.queries[0]
    assert "table_schema = $1" in query
    assert params == ("library",)


async def test_only_base_tables_are_listed():
    """A view has no GRANT story of its own here and cannot be introspected
    the same way; including them would offer the model tables it cannot
    reason about."""
    db = FakeDatabase([[]])
    await PostgresIntrospectionAdapter(db).list_tables()
    assert "BASE TABLE" in db.queries[0][0]


async def test_no_readable_tables_is_an_empty_tuple_not_an_error():
    """A role with no grants is a real configuration, and the caller decides
    what to do about it."""
    assert await PostgresIntrospectionAdapter(FakeDatabase([[]])).list_tables() == ()


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------


async def test_describe_builds_a_table_schema_per_table():
    db = FakeDatabase([[
        column_row("books", "id", "integer", nullable="NO"),
        column_row("books", "title_en", "text"),
        column_row("authors", "id", "integer"),
    ]])
    result = await PostgresIntrospectionAdapter(db).describe(("books", "authors"))

    assert set(result) == {"books", "authors"}
    assert isinstance(result["books"], TableSchema)
    assert result["books"].column_names == ("id", "title_en")
    assert result["books"].column("id").nullable is False


async def test_a_vector_column_is_recognised_and_named_usefully():
    """data_type is the unhelpful 'USER-DEFINED' for an extension type, so
    udt_name is what identifies it and what is worth reporting."""
    db = FakeDatabase([[column_row("books", "embed_summary", "USER-DEFINED", udt="vector")]])
    result = await PostgresIntrospectionAdapter(db).describe(("books",))

    column = result["books"].column("embed_summary")
    assert column.is_vector is True
    assert column.sql_type == "vector"


async def test_an_ordinary_column_is_not_marked_as_a_vector():
    db = FakeDatabase([[column_row("books", "title_en", "text")]])
    result = await PostgresIntrospectionAdapter(db).describe(("books",))

    column = result["books"].column("title_en")
    assert column.is_vector is False
    assert column.sql_type == "text"


async def test_another_user_defined_type_is_not_treated_as_a_vector():
    """Only udt_name = 'vector' is a pgvector column; an enum or a domain
    type also reports USER-DEFINED."""
    db = FakeDatabase([[column_row("books", "status", "USER-DEFINED", udt="book_status")]])
    result = await PostgresIntrospectionAdapter(db).describe(("books",))
    assert result["books"].column("status").is_vector is False


async def test_columns_keep_their_declared_order():
    """ordinal_position, so a schema reads the way it was written."""
    db = FakeDatabase([[
        column_row("books", "id", "integer"),
        column_row("books", "title_en", "text"),
        column_row("books", "genre", "text"),
    ]])
    result = await PostgresIntrospectionAdapter(db).describe(("books",))
    assert result["books"].column_names == ("id", "title_en", "genre")


async def test_a_table_the_connection_cannot_read_is_simply_absent():
    """Not present-and-empty: an empty TableSchema would read as "a table
    with no columns", which is a different and misleading claim."""
    db = FakeDatabase([[column_row("books", "id", "integer")]])
    result = await PostgresIntrospectionAdapter(db).describe(("books", "members"))

    assert "books" in result
    assert "members" not in result


async def test_describe_with_no_tables_asks_the_database_nothing():
    db = FakeDatabase()
    assert await PostgresIntrospectionAdapter(db).describe(()) == {}
    assert db.queries == []


async def test_table_names_are_lowercased_before_matching():
    db = FakeDatabase([[]])
    await PostgresIntrospectionAdapter(db).describe(("BOOKS", "Authors"))
    assert db.queries[0][1][1] == ["books", "authors"]


async def test_describe_is_one_round_trip_for_every_table():
    """It runs at agent startup; one query per table would make a wide schema
    noticeably slow to start."""
    db = FakeDatabase([[]])
    await PostgresIntrospectionAdapter(db).describe(("a", "b", "c", "d"))
    assert len(db.queries) == 1


# --------------------------------------------------------------------------
# distinct_count - the one call that touches data
# --------------------------------------------------------------------------


async def test_returns_the_count():
    db = FakeDatabase([[{"n": 10}]])
    assert await PostgresIntrospectionAdapter(db).distinct_count("books", "genre") == 10


async def test_an_empty_result_counts_as_zero():
    assert await PostgresIntrospectionAdapter(FakeDatabase([[]])).distinct_count("books", "genre") == 0


@pytest.mark.parametrize("bad", [
    'id") FROM books; DROP TABLE books --',
    "id; DELETE FROM books",
    "id, (SELECT 1)",
    "*",
    "",
])
async def test_an_injected_column_name_never_reaches_the_database(bad):
    """Identifiers cannot be parameterised, so this is the one place in the
    adapter where injection is possible."""
    db = FakeDatabase([[{"n": 1}]])
    with pytest.raises(DatabaseError):
        await PostgresIntrospectionAdapter(db).distinct_count("books", bad)
    assert db.queries == []


@pytest.mark.parametrize("bad", ["books; DROP TABLE authors", "public.books", "books--"])
async def test_an_injected_table_name_never_reaches_the_database(bad):
    db = FakeDatabase([[{"n": 1}]])
    with pytest.raises(DatabaseError):
        await PostgresIntrospectionAdapter(db).distinct_count(bad, "genre")
    assert db.queries == []


async def test_valid_identifiers_are_quoted_in_the_query():
    """Validation rejects anything that is not a bare identifier; quoting
    then means a name that happens to be a reserved word still works."""
    db = FakeDatabase([[{"n": 3}]])
    await PostgresIntrospectionAdapter(db).distinct_count("books", "genre")

    query = db.queries[0][0]
    assert '"genre"' in query
    assert '"books"' in query
