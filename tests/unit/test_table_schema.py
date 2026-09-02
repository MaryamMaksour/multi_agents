"""TableSchema and ColumnSchema - the shapes introspection fills in.

Small, but the embedding-partner rule is the one piece of real logic: it is
what decides that a column can be searched semantically, and it decides it by
looking at what is in the database rather than by consulting a list someone
maintains. Getting it wrong is quiet - a searchable column classified as
plain text simply stops being searchable.
"""

from __future__ import annotations

import dataclasses

import pytest

from domain.entities.table_schema import EMBED_COLUMN_PREFIX, ColumnSchema, TableSchema


def books() -> TableSchema:
    """A cut-down version of the development schema's books table."""
    return TableSchema(name="books", columns=(
        ColumnSchema("id", "integer", nullable=False),
        ColumnSchema("title_en", "text"),
        ColumnSchema("title_ar", "text"),
        ColumnSchema("genre", "text"),
        ColumnSchema("page_count", "integer"),
        ColumnSchema("added_at", "timestamp with time zone"),
        ColumnSchema("embed_title_en", "vector", is_vector=True),
        ColumnSchema("embed_title_ar", "vector", is_vector=True),
    ))


# --------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------


def test_finds_a_column_by_name():
    assert books().column("genre").sql_type == "text"


def test_lookup_is_case_insensitive():
    """Unquoted SQL identifiers fold to lowercase, and the model does not
    reliably match the case it was shown."""
    assert books().column("GENRE") is not None
    assert books().has_column("Title_EN")


def test_an_absent_column_is_none_rather_than_an_error():
    """The SQL tool uses this to tell the model a column does not exist -
    that is an answer to return, not a failure."""
    assert books().column("nonexistent") is None
    assert not books().has_column("nonexistent")


def test_column_names_are_reported_in_order():
    assert books().column_names[:3] == ("id", "title_en", "title_ar")


def test_an_empty_table_schema_is_valid():
    empty = TableSchema(name="new_table")
    assert empty.columns == ()
    assert not empty.has_column("anything")


# --------------------------------------------------------------------------
# embedding partners - what makes a column semantically searchable
# --------------------------------------------------------------------------


def test_a_column_with_an_embed_partner_is_searchable():
    partner = books().embedding_partner("title_en")
    assert partner is not None
    assert partner.name == "embed_title_en"
    assert partner.is_vector


def test_a_column_without_a_partner_has_none():
    assert books().embedding_partner("genre") is None
    assert books().embedding_partner("page_count") is None


def test_a_vector_column_is_not_its_own_partner():
    """embed_title_en stores title_en's embedding; it does not have one."""
    assert books().embedding_partner("embed_title_en") is None


def test_an_absent_column_has_no_partner():
    assert books().embedding_partner("nonexistent") is None


def test_the_partner_must_actually_be_a_vector():
    """A text column called embed_notes is not an embedding - matching on the
    name alone would classify it as searchable and produce SQL comparing text
    to a vector."""
    schema = TableSchema(name="t", columns=(
        ColumnSchema("notes", "text"),
        ColumnSchema("embed_notes", "text"),
    ))
    assert schema.embedding_partner("notes") is None


def test_partner_lookup_is_case_insensitive():
    assert books().embedding_partner("TITLE_EN") is not None


def test_the_prefix_is_defined_in_one_place():
    """The classifier, the SQL the model writes and any backfill script all
    depend on this convention agreeing."""
    assert EMBED_COLUMN_PREFIX == "embed_"
    assert books().column(f"{EMBED_COLUMN_PREFIX}title_en") is not None


# --------------------------------------------------------------------------
# filterable columns
# --------------------------------------------------------------------------


def test_vector_columns_are_excluded_from_filterable():
    """They are storage. A query filters on title_en, using embed_title_en to
    do it - offering the vector itself invites a comparison against raw
    floats."""
    filterable = books().filterable_columns
    assert all(not c.is_vector for c in filterable)
    assert len(filterable) == 6
    assert "embed_title_en" not in [c.name for c in filterable]


def test_a_table_with_no_vectors_keeps_every_column():
    schema = TableSchema(name="t", columns=(
        ColumnSchema("a", "text"), ColumnSchema("b", "integer"),
    ))
    assert len(schema.filterable_columns) == 2


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------


def test_a_schema_cannot_be_edited_in_place():
    """It describes something already true in the database. Code wanting a
    different shape should introspect again, not rewrite the description."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        books().name = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        ColumnSchema("id", "integer").sql_type = "text"


def test_columns_default_to_nullable():
    """information_schema reports is_nullable = 'YES' for most columns, so
    the permissive value is the right default to omit."""
    assert ColumnSchema("x", "text").nullable is True
    assert ColumnSchema("x", "text").is_vector is False
