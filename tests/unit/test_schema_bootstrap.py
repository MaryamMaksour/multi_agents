"""Startup: introspection and probing, without a database.

SchemaPort is a Protocol, so a fake is a class with three methods. That is
what makes the expensive question testable at all: which columns get a
`count(DISTINCT ...)` and which are left alone. The fake records every probe,
so "only text columns are probed" is an assertion rather than a claim.

The other thing worth pinning is what happens when a probe fails. It is the
one place in startup that reads data, and the promise is that failing it
costs guidance quality and nothing else - never the agent's ability to start.
"""

from __future__ import annotations

import pytest

from domain.entities.column_filter import FilterKind
from domain.entities.table_schema import ColumnSchema, TableSchema
from domain.exceptions import DatabaseError
from libs.agent_core.schema_bootstrap import (
    AgentSchema,
    columns_needing_a_count,
    count_distinct_values,
    load_agent_schema,
    render_columns,
)

BOOKS = TableSchema(name="books", columns=(
    ColumnSchema("id", "integer", nullable=False),
    ColumnSchema("title_en", "text"),
    ColumnSchema("summary", "text"),
    ColumnSchema("genre", "text"),
    ColumnSchema("isbn", "text"),
    ColumnSchema("page_count", "integer"),
    ColumnSchema("added_at", "timestamp with time zone"),
    ColumnSchema("embed_title_en", "vector", is_vector=True),
    ColumnSchema("embed_summary", "vector", is_vector=True),
))

AUTHORS = TableSchema(name="authors", columns=(
    ColumnSchema("id", "integer", nullable=False),
    ColumnSchema("name_en", "text"),
))

COUNTS = {("books", "genre"): 10, ("books", "isbn"): 420, ("authors", "name_en"): 28}


class FakeSchemaPort:
    """SchemaPort with a fixed schema and a call log.

    `fails` names columns whose probe raises, which is how the degrade path
    is exercised without a database that can be made to misbehave.
    """

    def __init__(self, tables=(BOOKS, AUTHORS), counts=None, fails=()):
        self._tables = {t.name: t for t in tables}
        self._counts = COUNTS if counts is None else counts
        self._fails = set(fails)
        self.probes: list[tuple[str, str]] = []
        self.reads: list[tuple[str, str, int]] = []
        self.described: list[tuple[str, ...]] = []

    async def list_tables(self) -> tuple[str, ...]:
        return tuple(self._tables)

    async def describe(self, tables: tuple[str, ...]) -> dict[str, TableSchema]:
        self.described.append(tuple(tables))
        return {n: self._tables[n] for n in tables if n in self._tables}

    async def distinct_count(self, table: str, column: str) -> int:
        self.probes.append((table, column))
        if (table, column) in self._fails:
            raise DatabaseError(f"cannot read {table}.{column}")
        return self._counts.get((table, column), 1)

    async def distinct_values(self, table: str, column: str, limit: int) -> tuple[str, ...]:
        self.reads.append((table, column, limit))
        if (table, column) in self._fails:
            raise DatabaseError(f"cannot read {table}.{column}")
        return tuple(f"{column}-{i}" for i in range(min(3, limit)))


# --------------------------------------------------------------------------
# which columns are worth a probe
# --------------------------------------------------------------------------


def test_only_text_columns_need_a_count():
    """Everything else is settled by an earlier rule in the precedence chain,
    so a count could not change the answer."""
    assert set(columns_needing_a_count(BOOKS)) == {"isbn", "genre"}


def test_a_semantic_column_is_not_probed():
    """`summary` has an embedding partner, so rule 2 decides it before
    cardinality is ever consulted."""
    assert "summary" not in columns_needing_a_count(BOOKS)


def test_vector_numeric_and_date_columns_are_not_probed():
    needed = columns_needing_a_count(BOOKS)
    for column in ("embed_summary", "page_count", "added_at", "id"):
        assert column not in needed


async def test_no_column_is_probed_more_than_once():
    port = FakeSchemaPort()
    await load_agent_schema(port)
    assert len(port.probes) == len(set(port.probes))


async def test_probing_can_be_turned_off_entirely():
    """The escape hatch for a large database: count(DISTINCT) is the only
    call here that scans data."""
    port = FakeSchemaPort()
    result = await load_agent_schema(port, probe_cardinality=False)

    assert port.probes == []
    assert result.classified["books"]["genre"].kind is FilterKind.TEXT


# --------------------------------------------------------------------------
# a failed probe degrades, it does not stop startup
# --------------------------------------------------------------------------


async def test_a_failed_probe_leaves_the_column_as_text():
    port = FakeSchemaPort(fails={("books", "genre")})
    result = await load_agent_schema(port)
    assert result.classified["books"]["genre"].kind is FilterKind.TEXT


async def test_a_failed_probe_does_not_stop_the_other_columns():
    port = FakeSchemaPort(fails={("books", "genre")})
    result = await load_agent_schema(port)

    assert ("authors", "name_en") in port.probes
    assert result.classified["books"]["isbn"].kind is FilterKind.TEXT
    assert "books" in result.filters and "authors" in result.filters


async def test_failed_probes_are_reported_rather_than_swallowed():
    """Silently thinner guidance is the kind of degradation nobody notices
    until the model starts guessing values."""
    port = FakeSchemaPort(fails={("books", "genre")})
    result = await load_agent_schema(port)
    assert result.unprobed == ("books.genre",)


async def test_nothing_is_reported_when_every_probe_succeeds():
    assert (await load_agent_schema(FakeSchemaPort())).unprobed == ()


async def test_counts_are_read_and_applied():
    port = FakeSchemaPort()
    result = await load_agent_schema(port)

    assert result.classified["books"]["genre"].kind is FilterKind.ENUM
    assert result.classified["books"]["isbn"].kind is FilterKind.TEXT


# --------------------------------------------------------------------------
# what comes back
# --------------------------------------------------------------------------


async def test_the_table_list_comes_from_the_connection():
    """Not from configuration. An agent connecting as its own role sees only
    the tables it was granted, so introspection *is* the allowlist."""
    result = await load_agent_schema(FakeSchemaPort())
    assert result.tables == ("authors", "books")


async def test_an_explicit_table_list_is_used_instead_of_listing():
    port = FakeSchemaPort()
    result = await load_agent_schema(port, tables=("books",))

    assert result.tables == ("books",)
    assert port.described == [("books",)]


async def test_a_table_the_connection_cannot_read_is_simply_absent():
    """describe() omits what the role cannot see rather than raising, so
    asking for too much narrows to what is permitted."""
    result = await load_agent_schema(FakeSchemaPort(), tables=("books", "members"))
    assert result.tables == ("books",)


async def test_filters_are_the_shape_the_adapter_indexes():
    """SqlToolAdapter reads self._filters[table][column] and returns the
    value to the model, so the leaves must already be strings."""
    result = await load_agent_schema(FakeSchemaPort())
    assert isinstance(result.filters["books"]["genre"], str)
    assert result.filters["books"]["genre"] == result.classified["books"]["genre"].guidance


async def test_vector_columns_are_present_in_the_filters():
    result = await load_agent_schema(FakeSchemaPort())
    assert "embed_summary" in result.filters["books"]


async def test_the_distance_operator_reaches_the_guidance():
    result = await load_agent_schema(FakeSchemaPort(), dist_op="<->")
    assert "<->" in result.filters["books"]["summary"]


async def test_table_keys_are_lowercased_to_match_the_adapter():
    """The adapter lowercases table_name before indexing, so a key that is
    not already lowercase is a table the model can never reach."""
    result = await load_agent_schema(FakeSchemaPort())
    assert all(key == key.lower() for key in result.filters)


async def test_the_result_is_frozen():
    result = await load_agent_schema(FakeSchemaPort())
    assert isinstance(result, AgentSchema)
    with pytest.raises(Exception):
        result.tables = ()


# --------------------------------------------------------------------------
# the rendered column block
# --------------------------------------------------------------------------


def test_every_column_appears_on_its_own_line():
    rendered = render_columns(BOOKS)
    assert len(rendered.splitlines()) == len(BOOKS.columns)


def test_each_line_starts_with_the_name_and_the_type():
    assert "page_count integer" in render_columns(BOOKS)


def test_not_null_is_shown():
    """The model can skip an IS NOT NULL guard it does not need, and knows
    to expect nulls where it is absent."""
    assert "id integer NOT NULL" in render_columns(BOOKS)
    assert "genre text\n" in render_columns(BOOKS) + "\n"


def test_the_rendered_block_parses_back_to_the_column_names():
    """The adapter reads this string back with a regex to validate a column
    in get_lsit_values. If the two disagree, every column looks unknown."""
    from adapters.outbound.tools.sql_tool_adapter import _extract_column_names

    parsed = _extract_column_names({"columns": render_columns(BOOKS)})
    assert parsed == set(BOOKS.column_names)


async def test_the_loader_output_drives_the_adapter_unchanged():
    """The whole point: no glue between this and the adapter's constructor."""
    from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter

    loaded = await load_agent_schema(FakeSchemaPort())
    adapter = SqlToolAdapter(
        db=None, embeddings=None, cache=None,
        allowed_tables=list(loaded.tables),
        schema=loaded.schema,
        filters=loaded.filters,
        lsit_values={},
        dist_op="<=>",
        vector_ttl_seconds=900,
    )

    answer = await adapter.call_tool(
        "get_filter", {"columns": ["genre", "summary"], "table_name": "books"}
    )
    # The values themselves, not the name of the tool that would list them -
    # they were read at startup and are already in the sentence.
    assert "genre-0" in answer["genre"]
    assert "embed_summary" in answer["summary"]


# --------------------------------------------------------------------------
# count_distinct_values on its own
# --------------------------------------------------------------------------


async def test_counting_returns_the_values_and_no_failures():
    counts, unprobed = await count_distinct_values(FakeSchemaPort(), BOOKS)
    assert counts["genre"] == 10
    assert unprobed == ()


async def test_counting_separates_the_failures_from_the_results():
    port = FakeSchemaPort(fails={("books", "isbn")})
    counts, unprobed = await count_distinct_values(port, BOOKS)

    assert "isbn" not in counts
    assert counts["genre"] == 10
    assert unprobed == ("books.isbn",)


# --------------------------------------------------------------------------
# reading the values of the columns that turned out to be enums
# --------------------------------------------------------------------------


async def test_enum_columns_have_their_values_read():
    port = FakeSchemaPort()
    result = await load_agent_schema(port)

    assert ("books", "genre") in [(t, c) for t, c, _ in port.reads]
    assert "genre-0" in result.filters["books"]["genre"]


async def test_only_enum_columns_are_read():
    """One query per enum column, and none for anything else. isbn has 420
    distinct values, so reading them is a cost with no use - and a list that
    long in a prompt is noise that crowds out the schema."""
    port = FakeSchemaPort()
    await load_agent_schema(port)

    read = {column for _, column, _ in port.reads}
    assert "genre" in read
    assert "isbn" not in read
    assert "summary" not in read      # semantic, decided before cardinality
    assert "page_count" not in read   # numeric, likewise


async def test_the_read_is_capped_at_the_enum_cutoff():
    """The same number that decided the column was an enum, not a second one
    that could drift from it."""
    from domain.entities.column_filter import ENUM_MAX_DISTINCT

    port = FakeSchemaPort()
    await load_agent_schema(port)

    assert all(limit == ENUM_MAX_DISTINCT for _, _, limit in port.reads)


async def test_a_failed_value_read_leaves_the_column_an_enum():
    """Degrade, never raise - the same rule the count probe follows. The
    guidance falls back to naming the tool that lists values."""
    port = FakeSchemaPort(fails={("books", "genre")})
    result = await load_agent_schema(port)

    assert result.classified["books"]["genre"].kind is FilterKind.TEXT


async def test_values_are_not_read_when_probing_is_off():
    port = FakeSchemaPort()
    await load_agent_schema(port, probe_cardinality=False)

    assert port.reads == []
