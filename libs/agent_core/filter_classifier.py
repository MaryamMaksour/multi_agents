"""Turns an introspected schema into a per-column filter classification.

This is the replacement for the four hand-maintained lists. It runs once per
agent at startup and produces the lookup the SQL tool serves from, so
_get_filter becomes a dictionary read rather than a scan of four lists.

Placed in libs/ rather than domain/ because it is a derivation over domain
entities, not an entity or a use case. It has no I/O of its own - it takes
already-introspected data and returns a classification, which makes it a
pure function and trivial to test.

Measured on seeds/001_schema.sql, books table:

    title_en, title_ar, summary   -> SEMANTIC  (embed_ partner present)
    page_count, price, *_id, ...  -> OPERATOR  (numeric data_type)
    added_at                      -> DATETIME  (timestamptz)
    genre 10, language 3 distinct -> ENUM
    isbn 420, shelf_code 399      -> TEXT      (effectively unique)
"""

from __future__ import annotations


# TODO: which data_type values map to OPERATOR. From information_schema:
#   integer, bigint, smallint, numeric, real, double precision
# Keep this as data, not a chain of if/elif - it is a lookup.
NUMERIC_TYPES: frozenset[str] = ...

# TODO: which map to DATETIME:
#   date, timestamp without time zone, timestamp with time zone, time*
DATETIME_TYPES: frozenset[str] = ...


def classify_column(table, column, has_embedding: bool, distinct_count: int | None):
    """Decide one column's FilterKind.

    Precedence, and the order matters:

        1. the column IS a vector column       -> not filterable directly,
                                                  it is storage for another
                                                  column's semantics
        2. it has an embed_<name> partner      -> SEMANTIC
        3. its type is numeric                 -> OPERATOR
        4. its type is a date/time             -> DATETIME
        5. it is text with low cardinality     -> ENUM
        6. otherwise                           -> TEXT

    Note what is NOT here: the old code had two unconditional overrides, one
    for name/shortname and one for address/location, that fired regardless of
    which list the column came from. Those encoded one deployment's column
    names into shared logic. Their legitimate purpose - "search these two
    columns together" - belongs in per-deployment annotation, not here.

    TODO: implement.
    """
    # TODO: implement
    ...


def classify_table(table, distinct_counts: dict[str, int] | None = None):
    """Classify every column in a TableSchema.

    Takes the table schema plus, optionally, distinct counts for its text
    columns. Without them, step 5 cannot run and text columns fall through to
    TEXT - which should degrade quality, never raise. Make that explicit:
    ENUM is an optimisation, and the classifier must work without it.

    TODO: implement.
    """
    # TODO: implement
    ...


def build_guidance(table_name: str, column_filter, enum_values=None) -> str:
    """Render the sentence the model reads for one classified column.

    One place, so the wording stays consistent across agents. Rough shapes:

        SEMANTIC  how to write the vector predicate, naming the embed column
                  and a starting distance threshold
        ENUM      the allowed values, since there are few enough to list
        OPERATOR  that comparisons and aggregates apply
        DATETIME  that it casts to timestamp and takes ranges
        TEXT      no strong guidance; exact match or ILIKE

    TODO: implement.

    TODO: for ENUM, listing actual values is far more useful to the model
    than saying "few distinct values" - but it means reading them. Decide
    whether to fetch values alongside distinct_count at startup, and cap the
    list length so a mis-classified column cannot flood the prompt.
    """
    # TODO: implement
    ...
