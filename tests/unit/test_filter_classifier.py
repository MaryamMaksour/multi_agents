"""The classifier - what replaces four hand-maintained lists.

Two different things are worth protecting here, and they fail differently.

The *classification* fails loudly enough to find: a column classified
OPERATOR when it is text produces SQL the database rejects. The *guidance*
fails quietly - it is a sentence handed to a model, and a wrong sentence
produces a query that runs, returns rows, and is wrong. So the guidance tests
are not decoration; they pin the few facts a model cannot recover on its own:
which column holds the embedding, which distance operator the index was built
for, and which tool to call before filtering.

Wording is deliberately not asserted. The tests check that a sentence
contains the embed column's name, or the configured operator, or the tool
name - never the whole sentence, which would break on every edit and protect
nothing.
"""

from __future__ import annotations

import pytest

from domain.entities.column_filter import ENUM_MAX_DISTINCT, ColumnFilter, FilterKind
from domain.entities.table_schema import ColumnSchema, TableSchema
from libs.agent_core.filter_classifier import (
    DATETIME_TYPES,
    DEFAULT_DIST_OP,
    NUMERIC_TYPES,
    build_guidance,
    classify_column,
    classify_table,
)


def books() -> TableSchema:
    """A cut-down books table, matching seeds/001_schema.sql.

    Deliberately keeps one of each case: a semantic column with a partner, a
    partnerless text column with few values, a partnerless text column with
    many, a numeric, a date, and the vector columns themselves.
    """
    return TableSchema(name="books", columns=(
        ColumnSchema("id", "integer", nullable=False),
        ColumnSchema("title_en", "text"),
        ColumnSchema("summary", "text"),
        ColumnSchema("genre", "text"),
        ColumnSchema("isbn", "text"),
        ColumnSchema("page_count", "integer"),
        ColumnSchema("price", "numeric"),
        ColumnSchema("added_at", "timestamp with time zone"),
        ColumnSchema("embed_title_en", "vector", is_vector=True),
        ColumnSchema("embed_summary", "vector", is_vector=True),
    ))


COUNTS = {"title_en": 420, "summary": 97, "genre": 10, "isbn": 420}


def kind_of(column_name: str, distinct_count: int | None = None) -> FilterKind:
    table = books()
    return classify_column(table, table.column(column_name), distinct_count)


def guidance_for(column_name: str, **kwargs) -> str:
    table = books()
    column = table.column(column_name)
    kind = classify_column(table, column, COUNTS.get(column_name))
    return build_guidance(table, column, kind, **kwargs)


# --------------------------------------------------------------------------
# the precedence chain, one rule at a time
# --------------------------------------------------------------------------


def test_a_vector_column_is_storage():
    assert kind_of("embed_summary") is FilterKind.VECTOR_STORAGE


def test_a_column_with_an_embedding_partner_is_semantic():
    assert kind_of("summary") is FilterKind.SEMANTIC


def test_a_numeric_column_takes_operators():
    assert kind_of("page_count") is FilterKind.OPERATOR
    assert kind_of("price") is FilterKind.OPERATOR


def test_a_timestamp_column_is_a_datetime():
    assert kind_of("added_at") is FilterKind.DATETIME


def test_a_text_column_with_few_values_is_an_enum():
    assert kind_of("genre", 10) is FilterKind.ENUM


def test_a_text_column_with_many_values_is_plain_text():
    assert kind_of("isbn", 420) is FilterKind.TEXT


# --------------------------------------------------------------------------
# the order of those rules, which is where a rewrite would break it
#
# Each of these passes under the correct order and fails under a plausible
# wrong one. Testing the rules individually would not catch a reordering.
# --------------------------------------------------------------------------


def test_a_vector_column_is_never_semantic():
    """`embed_summary` is text-adjacent and sits next to a semantic column,
    but it is the storage, not the thing searched. Rule 1 before rule 2."""
    assert kind_of("embed_summary") is not FilterKind.SEMANTIC


def test_semantic_wins_over_enum_even_when_values_are_few():
    """A short column - a title in a tiny table - has both an embedding
    partner and low cardinality. Rule 2 before rule 5, or every semantic
    column in a small table silently stops being searchable."""
    assert kind_of("summary", 3) is FilterKind.SEMANTIC


def test_a_numeric_column_with_few_values_is_still_an_operator():
    """`page_count` in a small table might have 5 distinct values. Rule 3
    before rule 5: the model must still be able to write `< 300`."""
    assert kind_of("page_count", 5) is FilterKind.OPERATOR


def test_a_date_column_with_few_values_is_still_a_datetime():
    """Same trap as above, for rule 4. Ranges must stay available."""
    assert kind_of("added_at", 4) is FilterKind.DATETIME


# --------------------------------------------------------------------------
# the ENUM cutoff
# --------------------------------------------------------------------------


def test_exactly_at_the_cutoff_is_an_enum():
    assert kind_of("genre", ENUM_MAX_DISTINCT) is FilterKind.ENUM


def test_one_over_the_cutoff_is_text():
    assert kind_of("genre", ENUM_MAX_DISTINCT + 1) is FilterKind.TEXT


def test_an_unmeasured_column_falls_through_to_text():
    """None means "not measured", not "zero". ENUM is an optimisation and
    its absence must never raise."""
    assert kind_of("genre", None) is FilterKind.TEXT


def test_an_empty_column_is_text_not_an_enum():
    """Zero passes `<= 20` but means the column has no values at all. Calling
    it an ENUM promises the model a short list to choose from, and
    get_lsit_values would answer that everything is NULL."""
    assert kind_of("genre", 0) is FilterKind.TEXT


def test_the_cutoff_does_not_exceed_what_the_tool_will_list():
    """A cross-file invariant. SqlToolAdapter._get_lsit_values returns a
    sample plus a count above 20 rather than the full list, and the ENUM
    guidance tells the model to call it. A higher cutoff here would classify
    columns as ENUM whose values the tool then refuses to enumerate."""
    assert ENUM_MAX_DISTINCT <= 20


# --------------------------------------------------------------------------
# the type tables
# --------------------------------------------------------------------------


def test_the_type_tables_cannot_be_mutated():
    """Module-level constants read on every classification. A single .add()
    anywhere - `NUMERIC_TYPES.add("text")` - would reclassify every text
    column in every table as OPERATOR, and nothing would report it."""
    with pytest.raises(AttributeError):
        NUMERIC_TYPES.add("text")
    with pytest.raises(AttributeError):
        DATETIME_TYPES.add("text")


@pytest.mark.parametrize("sql_type", [
    "integer", "bigint", "smallint", "numeric", "real", "double precision",
])
def test_every_postgres_numeric_type_is_recognised(sql_type):
    assert sql_type in NUMERIC_TYPES


@pytest.mark.parametrize("sql_type", [
    "date",
    "timestamp with time zone", "timestamp without time zone",
    "time with time zone", "time without time zone",
])
def test_every_postgres_datetime_type_is_recognised(sql_type):
    """Spelled as information_schema reports them, not as the DDL writes
    them: `timestamptz` in a CREATE TABLE comes back as `timestamp with time
    zone`, and matching the short form would classify every timestamp column
    as text."""
    assert sql_type in DATETIME_TYPES


def test_text_is_in_neither_table():
    assert "text" not in NUMERIC_TYPES and "text" not in DATETIME_TYPES


# --------------------------------------------------------------------------
# classify_table
# --------------------------------------------------------------------------


def test_every_column_appears_in_the_result():
    result = classify_table(books(), COUNTS)
    assert tuple(result) == books().column_names


def test_the_result_is_keyed_by_column_name_not_by_the_column_object():
    result = classify_table(books(), COUNTS)
    assert result["genre"].column == "genre"
    assert isinstance(result["genre"].column, str)


def test_each_entry_is_a_column_filter_with_all_three_fields():
    entry = classify_table(books(), COUNTS)["summary"]
    assert isinstance(entry, ColumnFilter)
    assert entry.column == "summary"
    assert entry.kind is FilterKind.SEMANTIC
    assert isinstance(entry.guidance, str) and entry.guidance


def test_vector_columns_are_kept_rather_than_filtered_out():
    """A model that asks about `embed_summary` should be told what it is.
    Dropping it here would reach it as SqlToolAdapter's "column not found",
    which reads like a mistake on the model's part."""
    result = classify_table(books(), COUNTS)
    assert result["embed_summary"].kind is FilterKind.VECTOR_STORAGE


def test_it_works_with_no_counts_at_all():
    """The guard clause. `None` is the default, and `None.get` does not
    exist - without it this is an AttributeError at startup."""
    result = classify_table(books())
    assert result["genre"].kind is FilterKind.TEXT


def test_without_counts_only_the_enum_columns_change():
    """ENUM is the only rule that needs data. Everything else is derived
    from the schema alone and must be unaffected."""
    with_counts = classify_table(books(), COUNTS)
    without = classify_table(books())

    changed = {n for n in with_counts if with_counts[n].kind is not without[n].kind}
    assert changed == {"genre"}


def test_an_empty_counts_dict_behaves_like_none():
    assert classify_table(books(), {})["genre"].kind is FilterKind.TEXT


def test_counts_for_unknown_columns_are_ignored():
    """Counts are looked up per column, so a stale key names a column that
    is simply never asked for."""
    result = classify_table(books(), {**COUNTS, "column_that_was_dropped": 2})
    assert tuple(result) == books().column_names


def test_a_table_with_no_columns_gives_an_empty_result():
    assert classify_table(TableSchema("empty", ())) == {}


# --------------------------------------------------------------------------
# guidance - the sentences the model reads
# --------------------------------------------------------------------------


def test_semantic_guidance_names_the_embedding_column():
    """The model cannot derive `embed_summary` from `summary` - the prefix
    convention lives in our code, not in its head."""
    assert "embed_summary" in guidance_for("summary")


def test_semantic_guidance_names_the_tool_that_produces_the_vector():
    """Vectors are never pasted into SQL. The model calls embed_query_tool,
    gets a token, and passes the token as a parameter."""
    text = guidance_for("summary")
    assert "embed_query_tool" in text
    assert "token" in text


def test_semantic_guidance_uses_the_default_distance_operator():
    assert DEFAULT_DIST_OP in guidance_for("summary")


def test_semantic_guidance_uses_the_operator_it_was_given():
    """A deployment that sets DIST_OP must not have the model told to write
    a different operator than the index was built for."""
    text = guidance_for("summary", dist_op="<->")
    assert "<->" in text
    assert "<=>" not in text


def test_the_distance_threshold_is_offered_only_for_cosine():
    """0.35 is a cosine distance. Under inner product it is meaningless, so
    it is left out rather than misapplied."""
    assert "0.35" in guidance_for("summary", dist_op="<=>")
    assert "0.35" not in guidance_for("summary", dist_op="<#>")


def test_semantic_guidance_still_offers_exact_matching():
    """Embedding a column adds a way to search it; it does not remove =."""
    text = guidance_for("summary")
    assert "ILIKE" in text or "=" in text


def test_vector_storage_guidance_names_the_column_it_belongs_to():
    """Points the model back at `summary` rather than leaving it to strip the
    prefix itself."""
    text = guidance_for("embed_summary")
    assert "summary" in text
    assert "embed_summary" not in text.replace("embed_summary's", "")


def test_operator_guidance_names_the_actual_sql_type():
    assert "integer" in guidance_for("page_count")


def test_datetime_guidance_asks_for_a_range_rather_than_equality():
    """Equality on a timestamptz matches almost nothing - the model has to
    be told to bracket it."""
    text = guidance_for("added_at")
    assert ">=" in text and "<" in text


def test_enum_guidance_points_at_the_tool_that_lists_values():
    """Named exactly as SqlToolAdapter registers it. The spelling is wrong
    and deliberately so - see docs/deferred.md. A "corrected" name here
    sends the model to a tool that does not exist."""
    text = guidance_for("genre")
    assert "get_lsit_values" in text
    assert "books" in text and "genre" in text


def test_enum_guidance_lists_the_values_when_it_has_them():
    table = books()
    text = build_guidance(
        table, table.column("genre"), FilterKind.ENUM,
        enum_values=["novel", "poetry", "drama"],
    )
    assert "novel" in text and "poetry" in text and "drama" in text
    assert "get_lsit_values" not in text


def test_text_guidance_warns_against_listing_the_values():
    """Models call tools speculatively. On `isbn` that returns a 420-value
    sample which is noise in the context and crowds out the schema."""
    text = guidance_for("isbn")
    assert "Do not call get_lsit_values" in text


def test_every_kind_has_guidance():
    """No kind may fall through to a generic sentence. Asked of `summary`,
    which has an embedding partner, so every branch has what it needs."""
    table = books()
    column = table.column("summary")
    for kind in FilterKind:
        assert build_guidance(table, column, kind).strip()


def test_semantic_guidance_on_a_column_with_no_partner_says_so():
    """Not reachable through classify_table, which only reaches SEMANTIC when
    the partner exists. Reachable by a direct call, and the bare
    AttributeError it used to raise named neither the column nor the reason."""
    table = books()
    with pytest.raises(ValueError, match="genre"):
        build_guidance(table, table.column("genre"), FilterKind.SEMANTIC)


def test_an_unknown_kind_is_refused_rather_than_given_text_guidance():
    """If a FilterKind is ever added without guidance, this stops the agent
    at startup instead of silently handing the model the TEXT sentence."""
    with pytest.raises(ValueError):
        build_guidance(books(), books().column("genre"), "not_a_kind")


# --------------------------------------------------------------------------
# the operator reaches the guidance through classify_table
#
# build_guidance takes dist_op and classify_table takes dist_op; nothing
# proves they are connected until the value is followed end to end.
# --------------------------------------------------------------------------


def test_classify_table_threads_the_distance_operator_through():
    result = classify_table(books(), COUNTS, dist_op="<->")
    assert "<->" in result["summary"].guidance
    assert "<=>" not in result["summary"].guidance


def test_classify_table_defaults_to_the_cosine_operator():
    result = classify_table(books(), COUNTS)
    assert DEFAULT_DIST_OP in result["summary"].guidance
