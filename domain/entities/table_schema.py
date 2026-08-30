"""Database structure as the core sees it: discovered, never declared.

SqlToolAdapter already receives `schema` through its constructor, so nothing
about that adapter changes - this is about where that argument comes from.
Today it would be a hand-written dict; introspection produces it instead, and
the composition root is the only place that notices.

Nothing here is specific to any deployment - these are the shapes that
introspection fills in at runtime.

Verified against the development schema: introspecting through an agent's own
Postgres role returns exactly the tables that role was granted, so a
TableSchema collection built this way is already scoped to the agent. No
table list in code, and no second list to drift out of sync with the GRANTs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSchema:
    """One column as reported by the database.

    Fields to define:

        name        str  - column name
        sql_type    str  - information_schema.data_type, e.g. 'integer',
                           'text', 'timestamp with time zone'
        udt_name    str  - needed because extension types all report
                           data_type='USER-DEFINED'; udt_name='vector' is how
                           a pgvector column is recognised
        nullable    bool

    TODO: decide whether to keep `udt_name` on the entity or collapse it into
    a computed `is_vector` flag here. Keeping the raw value is more honest but
    leaks a Postgres detail into the domain; a boolean is cleaner but assumes
    pgvector is the only extension type that will ever matter.
    """

    # TODO: implement
    ...


@dataclass(frozen=True)
class TableSchema:
    """One table and its columns.

    Fields to define:

        name     str
        columns  tuple[ColumnSchema, ...]

    Useful behaviour to add:

        column(name)          - lookup, None if absent
        has_column(name)      - the SQL tool needs this to reject unknown
                                columns before building a query
        embedding_partner(c)  - returns the embed_<c> column if one exists.
                                This is what makes a text column semantically
                                searchable, and it is discoverable rather
                                than declared.
    """

    # TODO: implement
    ...


# TODO: the convention that ties a text column to its vector column. Keep it
# in one place - the classifier, the SQL builder, and any embedding backfill
# script all have to agree on it.
EMBED_COLUMN_PREFIX = "embed_"
