"""SqlToolAdapter - dispatch, guards, cursors and vector tokens.

The largest adapter and the one the model interacts with most, so most of
what it does is refuse things: a table it may not read, a query without a
page limit, an offset that would walk the whole table. Each refusal is a
returned message rather than an exception, because the model has to be able
to correct itself - so these tests assert on the *content* of the refusal,
not just that something went wrong.

Everything is faked. The queries never reach a database; what is being tested
is which query would have been sent, and with what parameters.
"""

from __future__ import annotations

import json

import pytest

from adapters.outbound.tools import sql_tool_adapter as mod
from adapters.outbound.tools.sql_tool_adapter import MAX_OFFSET, SqlToolAdapter
from domain.exceptions import UnknownToolError, ToolExecutionError

from tests.conftest import FakeCache, FakeDatabase, FakeEmbeddings

# No module-level asyncio mark: pytest.ini sets asyncio_mode = auto, so async
# tests are collected on their own and a blanket mark would misfire on the
# synchronous cursor tests below.

SCHEMA = {
    "books": {"columns": "id int4\ntitle_en text\ngenre text\npage_count int4"},
    "authors": {"columns": "id int4\nname_en text"},
}
FILTERS = {
    "books": {"genre": "ENUM: one of novel, poetry", "page_count": "OPERATOR"},
    "authors": {"name_en": "SEMANTIC"},
}


def adapter(db=None, cache=None, embeddings=None, **kw) -> SqlToolAdapter:
    params = dict(
        db=db or FakeDatabase(),
        embeddings=embeddings or FakeEmbeddings(),
        cache=cache or FakeCache(),
        allowed_tables=["books", "authors"],
        schema=SCHEMA,
        filters=FILTERS,
        lsit_values={},
        dist_op="<=>",
        vector_ttl_seconds=900,
    )
    params.update(kw)
    return SqlToolAdapter(**params)


def paged(query="SELECT id FROM books LIMIT $1 OFFSET $2", params=(10, 0)):
    return dict(
        query=query, params=list(params), offset=0,
        count_query="SELECT count(*) FROM books", count_params=[],
    )


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


async def test_an_unknown_tool_raises_rather_than_returning_an_error():
    """The loop turns this into a message for the model; the adapter's job is
    to be unambiguous that the name does not exist."""
    with pytest.raises(UnknownToolError):
        await adapter().call_tool("no_such_tool", {})


async def test_a_handler_failure_is_wrapped_as_a_tool_execution_error():
    with pytest.raises(ToolExecutionError):
        await adapter().call_tool("get_table_schema", {"wrong_kwarg": 1})


async def test_every_declared_tool_is_dispatchable():
    a = adapter()
    for schema in a.get_tool_schemas():
        assert schema["function"]["name"] in a._handlers


# --------------------------------------------------------------------------
# schema and filters - scoped to this agent
# --------------------------------------------------------------------------


async def test_returns_the_schema_of_an_allowed_table():
    result = await adapter().call_tool("get_table_schema", {"tables": ["books"]})
    assert result["books"] == SCHEMA["books"]


async def test_a_table_outside_the_agent_is_refused_by_name():
    result = await adapter().call_tool("get_table_schema", {"tables": ["members"]})
    assert "not allowed" in str(result["members"]).lower()


async def test_table_names_are_matched_case_insensitively():
    result = await adapter().call_tool("get_table_schema", {"tables": ["BOOKS"]})
    assert result["books"] == SCHEMA["books"]


async def test_filters_come_back_per_column():
    result = await adapter().call_tool(
        "get_filter", {"columns": ["genre", "page_count"], "table_name": "books"}
    )
    assert result["genre"].startswith("ENUM")
    assert result["page_count"] == "OPERATOR"


async def test_an_unclassified_column_says_so_rather_than_failing():
    """A missing entry must not raise - the model asked a reasonable question
    and needs an answer it can act on."""
    result = await adapter().call_tool(
        "get_filter", {"columns": ["nonexistent"], "table_name": "books"}
    )
    assert "not found" in result["nonexistent"].lower()


async def test_filter_lookup_folds_case_on_the_column_name():
    """table_name was already lowercased before the lookup; the column name
    was not. A model that writes "Genre" got "column not found" for a column
    it had just been shown, and no amount of retrying would have fixed it."""
    result = await adapter().call_tool(
        "get_filter", {"columns": ["Genre", "PAGE_COUNT"], "table_name": "books"}
    )
    assert result["Genre"].startswith("ENUM")
    assert result["PAGE_COUNT"] == "OPERATOR"


async def test_the_answer_keeps_the_spelling_the_model_asked_with():
    """Only the lookup folds. The model matches the reply to its own request
    by key, so answering "genre" to a question about "Genre" would leave it
    unable to pair them."""
    result = await adapter().call_tool(
        "get_filter", {"columns": ["Genre"], "table_name": "books"}
    )
    assert "Genre" in result and "genre" not in result


async def test_a_genuinely_unknown_column_still_says_not_found():
    """Folding case must not turn every miss into a hit."""
    result = await adapter().call_tool(
        "get_filter", {"columns": ["GenreX"], "table_name": "books"}
    )
    assert "not found" in result["GenreX"].lower()


async def test_filters_for_a_disallowed_table_are_refused():
    result = await adapter().call_tool(
        "get_filter", {"columns": ["id"], "table_name": "members"}
    )
    assert "not allowed" in str(result).lower()


# --------------------------------------------------------------------------
# db_execute - the guards
# --------------------------------------------------------------------------


async def test_a_well_formed_paginated_query_runs():
    db = FakeDatabase([[{"id": 1}], [{"count": 1}]])
    result = await adapter(db=db).call_tool("db_execute", paged())
    assert result["rows"] == [{"id": 1}]
    assert result["has_more"] is False


async def test_a_query_without_a_limit_placeholder_is_refused():
    """Without it a model can pull an entire table into its context."""
    result = await adapter().call_tool("db_execute", paged(query="SELECT id FROM books"))
    assert "limit" in result["error"].lower()


async def test_a_query_without_an_offset_placeholder_is_refused():
    result = await adapter().call_tool(
        "db_execute", paged(query="SELECT id FROM books LIMIT $1")
    )
    assert "offset" in result["error"].lower()


async def test_a_page_larger_than_a_hundred_rows_is_refused():
    result = await adapter().call_tool("db_execute", paged(params=(500, 0)))
    assert "100" in result["error"]


async def test_an_offset_beyond_the_cap_is_refused():
    """Deep offsets scan everything before them; past a point the model
    should be narrowing its filter instead."""
    result = await adapter().call_tool("db_execute", paged(params=(10, MAX_OFFSET + 1)))
    assert str(MAX_OFFSET) in result["error"]


async def test_params_without_a_limit_and_offset_are_refused():
    result = await adapter().call_tool("db_execute", paged(params=(10,)))
    assert "limit" in result["error"].lower()


async def test_a_non_numeric_limit_is_a_message_not_a_traceback():
    """`int("ten")` used to escape as ValueError and reach the model as a
    Python traceback; the other guards all return a sentence it can act on."""
    result = await adapter().call_tool("db_execute", paged(params=("ten", 0)))
    assert "integer" in result["error"].lower()


async def test_a_disallowed_table_is_refused_before_reaching_the_database():
    db = FakeDatabase()
    result = await adapter(db=db).call_tool(
        "db_execute", paged(query="SELECT * FROM members LIMIT $1 OFFSET $2")
    )
    assert "not allowed" in result["error"].lower()
    assert db.queries == [], "a refused query must never be sent"


async def test_a_write_is_refused():
    result = await adapter().call_tool(
        "db_execute", paged(query="DELETE FROM books LIMIT $1 OFFSET $2")
    )
    assert "select" in result["error"].lower()


async def test_the_count_query_is_validated_too():
    """It is a second query with its own text; validating only the first
    would leave an unchecked path to the database."""
    args = paged()
    args["count_query"] = "SELECT count(*) FROM members"
    result = await adapter().call_tool("db_execute", args)
    assert "not allowed" in result["error"].lower()


# --------------------------------------------------------------------------
# pagination
# --------------------------------------------------------------------------


async def test_more_rows_than_one_page_reports_has_more_with_a_cursor():
    db = FakeDatabase([[{"id": i} for i in range(10)], [{"count": 42}]])
    result = await adapter(db=db).call_tool("db_execute", paged())

    assert result["has_more"] is True
    assert result["next_cursor"]


async def test_the_last_page_reports_no_cursor():
    db = FakeDatabase([[{"id": 1}], [{"count": 1}]])
    result = await adapter(db=db).call_tool("db_execute", paged())

    assert result["has_more"] is False
    assert result["next_cursor"] == ""


async def test_a_cursor_restores_the_query_and_advances_the_offset():
    """The cursor carries the query, so the model does not rebuild it - and
    cannot quietly change it between pages."""
    db = FakeDatabase([[{"id": i} for i in range(10)], [{"count": 42}]])
    first = await adapter(db=db).call_tool("db_execute", paged())

    db2 = FakeDatabase([[{"id": i} for i in range(10, 20)], [{"count": 42}]])
    second = await adapter(db=db2).call_tool(
        "execute_next_cursor", {"cursor": first["next_cursor"]}
    )

    assert second["rows"][0]["id"] == 10
    sent_query, sent_params = db2.queries[0]
    assert "FROM books" in sent_query
    assert sent_params[-1] == 10, "the offset should have advanced by one page"


def test_a_cursor_round_trips_its_payload():
    payload = {"offset": 20, "query": "SELECT 1", "resolved_params": [10, 20]}
    assert mod._decode_cursor(mod._encode_cursor(payload)) == payload


def test_an_oversized_cursor_is_rejected():
    """The cursor comes back from the model and is decompressed; without a
    bound this is a decompression bomb."""
    huge = mod._encode_cursor({"pad": "x" * 500_000})
    with pytest.raises(ValueError):
        mod._decode_cursor(huge, max_bytes=1024)


# --------------------------------------------------------------------------
# vector tokens
# --------------------------------------------------------------------------


async def test_embedding_a_query_returns_a_short_token():
    """The vector itself must never enter the model's context - a thousand
    floats would crowd out the conversation."""
    cache = FakeCache()
    result = await adapter(cache=cache).call_tool("embed_query_tool", {"query": "sea view"})

    token = result["vector_token"]
    assert token.startswith("vec_")
    assert len(token) < 32
    assert cache.store[token] == [0.1] * 8


async def test_a_vector_token_is_resolved_to_its_vector_at_query_time():
    cache = FakeCache()
    a = adapter(cache=cache)
    token = (await a.call_tool("embed_query_tool", {"query": "sea view"}))["vector_token"]

    db = FakeDatabase([[{"id": 1}], [{"count": 1}]])
    a2 = adapter(db=db, cache=cache)
    await a2.call_tool("db_execute", dict(
        query="SELECT id FROM books WHERE embed_summary <=> $1::vector < 0.35 LIMIT $2 OFFSET $3",
        params=[token, 10, 0], offset=0,
        count_query="SELECT count(*) FROM books", count_params=[],
    ))

    _, sent_params = db.queries[0]
    # pgvector's text form, not a Python list: asyncpg has no codec for the
    # vector type, and a list reaches it as "expected str, got list".
    assert sent_params[0] == "[" + ",".join(["0.1"] * 8) + "]", (
        "the token should have been swapped for the vector"
    )


async def test_tokens_in_the_count_query_are_resolved_too():
    cache = FakeCache()
    a = adapter(cache=cache)
    token = (await a.call_tool("embed_query_tool", {"query": "x"}))["vector_token"]

    db = FakeDatabase([[{"id": 1}], [{"count": 1}]])
    await adapter(db=db, cache=cache).call_tool("db_execute", dict(
        query="SELECT id FROM books WHERE embed_summary <=> $1::vector < 0.3 LIMIT $2 OFFSET $3",
        params=[token, 10, 0], offset=0,
        count_query="SELECT count(*) FROM books WHERE embed_summary <=> $1::vector < 0.3",
        count_params=[token],
    ))

    _, count_params = db.queries[1]
    assert count_params[0] == "[" + ",".join(["0.1"] * 8) + "]"


# --------------------------------------------------------------------------
# distinct values
# --------------------------------------------------------------------------


async def test_distinct_values_are_read_from_the_database():
    db = FakeDatabase([[{"genre": "novel"}, {"genre": "poetry"}]])
    result = await adapter(db=db).call_tool("get_lsit_values", {"table": "books", "column": "genre"})

    assert "novel" in str(result)
    assert "SELECT DISTINCT" in db.queries[0][0]


async def test_a_disallowed_table_is_refused_without_querying():
    db = FakeDatabase()
    result = await adapter(db=db).call_tool(
        "get_lsit_values", {"table": "members", "column": "id"}
    )
    assert "error" in result
    assert db.queries == []


async def test_a_column_not_in_the_schema_is_refused_without_querying():
    """The column name is interpolated into SQL, so it is checked against the
    schema before it can reach a query."""
    db = FakeDatabase()
    result = await adapter(db=db).call_tool(
        "get_lsit_values", {"table": "books", "column": "not_a_column"}
    )
    assert "error" in result
    assert db.queries == []


async def test_an_injection_in_a_column_name_never_reaches_the_database():
    db = FakeDatabase()
    with pytest.raises(ToolExecutionError):
        await adapter(db=db).call_tool(
            "get_lsit_values", {"table": "books", "column": "id; DROP TABLE books"}
        )
    assert db.queries == []


async def test_many_distinct_values_are_summarised_rather_than_listed():
    """Twenty values are useful context; four hundred are noise."""
    db = FakeDatabase([[{"genre": f"g{i}"} for i in range(50)]])
    result = await adapter(db=db).call_tool(
        "get_lsit_values", {"table": "books", "column": "genre"}
    )
    assert "50" in str(result)


# --------------------------------------------------------------------------
# semantic row sampling
# --------------------------------------------------------------------------


async def test_table_records_are_fetched_by_vector_distance():
    db = FakeDatabase([[{"row_txt": "a book"}]])
    result = await adapter(db=db).call_tool(
        "get_table_records", {"query": "sea view", "table_name": "books"}
    )

    assert result["rows"] == ["a book"]
    assert "<=>" in db.queries[0][0]


async def test_the_sample_size_is_clamped():
    """A model asking for 500 sample rows has misunderstood the tool."""
    db = FakeDatabase([[{"row_txt": "x"}]])
    await adapter(db=db).call_tool(
        "get_table_records", {"query": "q", "table_name": "books", "mx": 500}
    )
    assert db.queries[0][1][-1] <= 6


async def test_table_records_refuse_a_disallowed_table():
    db = FakeDatabase()
    result = await adapter(db=db).call_tool(
        "get_table_records", {"query": "q", "table_name": "members"}
    )
    assert "error" in result
    assert db.queries == []
