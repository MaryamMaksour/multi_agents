"""Database structure as the core sees it: discovered, never declared.

SqlToolAdapter already receives `schema` through its constructor, so nothing
about that adapter changes - this is about where that argument comes from.
Today it would be a hand-written dict; introspection produces it instead, and
the composition root is the only place that notices.

Nothing here is specific to any deployment - these are the shapes that
introspection fills in at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

# The convention tying a text column to the vector column holding its
# embedding: `summary` is searchable semantically because `embed_summary`
# exists. Defined once here because the classifier, the SQL the model writes,
# and any embedding backfill all have to agree on it.
EMBED_COLUMN_PREFIX = "embed_"


@dataclass(frozen=True)
class ColumnSchema:
    """One column, as the database reports it.

    `is_vector` is a boolean rather than the raw Postgres udt_name on
    purpose. Recognising a pgvector column means knowing that extension types
    report data_type='USER-DEFINED' and are identified by udt_name - a
    Postgres detail that belongs in the introspection adapter, not in an
    entity the domain reads. The adapter decides; this records the decision.

    Frozen because a schema is a description of something already true. Code
    that wants a different shape should introspect again, not edit this.
    """

    name: str
    sql_type: str
    is_vector: bool = False
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    """One table and its columns."""

    name: str
    columns: tuple[ColumnSchema, ...] = ()

    def column(self, name: str) -> ColumnSchema | None:
        """The named column, or None. Case-insensitive: SQL identifiers fold
        to lowercase unquoted, and the model does not always match case."""
        target = (name or "").lower()
        for column in self.columns:
            if column.name.lower() == target:
                return column
        return None

    def has_column(self, name: str) -> bool:
        return self.column(name) is not None

    def embedding_partner(self, name: str) -> ColumnSchema | None:
        """The vector column holding this column's embedding, if there is one.

        This is what makes a column semantically searchable, and it is
        discovered rather than declared - no list of "these columns have
        embeddings" to maintain and drift.

        A vector column is not its own partner: `embed_summary` stores
        `summary`'s embedding, it does not have one of its own.
        """
        column = self.column(name)
        if column is None or column.is_vector:
            return None

        partner = self.column(f"{EMBED_COLUMN_PREFIX}{column.name}")
        return partner if partner is not None and partner.is_vector else None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    @property
    def filterable_columns(self) -> tuple[ColumnSchema, ...]:
        """Everything except the vector columns.

        Those are storage: a query filters on `summary`, using
        `embed_summary` to do it. Offering them to the model as filterable in
        their own right invites a comparison against a raw vector.
        """
        return tuple(c for c in self.columns if not c.is_vector)
