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

from domain.entities.column_filter import FilterKind, ENUM_MAX_DISTINCT, ColumnFilter
from domain.entities.table_schema import EMBED_COLUMN_PREFIX

# pgvector's cosine distance. Mirrors config.DIST_OP's default; the
# composition root passes the configured operator through, so the guidance
# names the same operator the index was built for.
DEFAULT_DIST_OP = "<=>"

# The values information_schema.columns.data_type reports, spelled in full -
# `timestamptz` in the DDL comes back as `timestamp with time zone`.
#
# frozenset, not set: these are module-level constants read on every
# classification, and a single `.add()` anywhere would silently reclassify
# every column in every table.
NUMERIC_TYPES: frozenset[str] = frozenset({
    "integer", "bigint", "smallint", "numeric", "real", "double precision",
})

DATETIME_TYPES: frozenset[str] = frozenset({
    "date",
    "timestamp with time zone", "timestamp without time zone",
    "time with time zone", "time without time zone",
})


def classify_column(table, column, distinct_count: int | None) -> FilterKind:
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

    `distinct_count` is None when it was not measured, and ENUM simply does
    not apply - the column falls through to TEXT rather than raising. Zero is
    not None but is treated the same way: an empty column has no short list
    for the model to choose from.

    Note what is NOT here: the old code had two unconditional overrides, one
    for name/shortname and one for address/location, that fired regardless of
    which list the column came from. Those encoded one deployment's column
    names into shared logic. Their legitimate purpose - "search these two
    columns together" - belongs in per-deployment annotation, not here.
    """
    if column.is_vector:
        return FilterKind.VECTOR_STORAGE

    elif table.embedding_partner(column.name) is not None:
        return FilterKind.SEMANTIC

    elif column.sql_type in NUMERIC_TYPES:
        return FilterKind.OPERATOR

    elif column.sql_type in DATETIME_TYPES:
        return FilterKind.DATETIME

    elif distinct_count is not None and 0 < distinct_count <= ENUM_MAX_DISTINCT:
        return FilterKind.ENUM

    else:
        return FilterKind.TEXT


def classify_table(
    table,
    distinct_counts: dict[str, int] | None = None,
    dist_op: str = DEFAULT_DIST_OP,
) -> dict[str, ColumnFilter]:
    """Classify every column in a TableSchema.

    Takes the table schema plus, optionally, distinct counts keyed by column
    name. Without them, step 5 cannot run and text columns fall through to
    TEXT - which should degrade quality, never raise. ENUM is an
    optimisation, and the classifier must work without it.

    `dist_op` is threaded through to the guidance so the sentence the model
    reads names the same operator SqlToolAdapter was configured with. The
    default matches config.DIST_OP's, but a deployment that changes one and
    not the other would tell the model to write a query the index cannot
    serve.

    Vector columns are kept in the result rather than filtered out. They
    classify as VECTOR_STORAGE, so a model that asks about `embed_summary`
    gets told what it is instead of "column not found".
    """
    # The caller may pass nothing at all. An empty dict makes every .get
    # below return None, which classify_column reads as "no cardinality
    # known" and sends to TEXT.
    distinct_counts = distinct_counts or {}

    table_filter: dict[str, ColumnFilter] = {}

    for column in table.columns:
        kind = classify_column(table, column, distinct_counts.get(column.name))
        guidance = build_guidance(table, column, kind, dist_op)
        table_filter[column.name] = ColumnFilter(column.name, kind, guidance)

    return table_filter


def build_guidance(table, column, kind, dist_op: str = DEFAULT_DIST_OP,
                   enum_values: list | None = None) -> str:
    """Render the sentence the model reads for one classified column.

    One place, so the wording stays consistent across agents. The sentences
    name the actual tools and operators the adapter serves - guidance that
    says "use a distance operator" without saying which one invites `<->`
    against an index built for `<=>`.

    `enum_values` is unused for now. Reading a column's values is a second
    query per ENUM column at startup; see docs/deferred.md.
    """
    if kind is FilterKind.VECTOR_STORAGE:
        source = column.name.removeprefix(EMBED_COLUMN_PREFIX)
        return (
            f"Storage for {source}'s embedding, not something to filter on "
            f"directly. Filter on {source} and this column is used for you."
        )

    if kind is FilterKind.SEMANTIC:
        partner = table.embedding_partner(column.name)
        if partner is None:
            # Unreachable through classify_table, which only reaches SEMANTIC
            # when the partner exists. Reachable by a direct call, and worth
            # saying plainly: the AttributeError one line down names neither
            # the column nor the reason.
            raise ValueError(
                f"{table.name}.{column.name} was classified SEMANTIC but has no "
                f"{EMBED_COLUMN_PREFIX}{column.name} column to search against"
            )

        threshold = (
            " Lower is closer; start around 0.35 and widen it if too few rows "
            "come back."
            if dist_op == "<=>"
            else " Lower is closer."
        )
        return (
            f"Searchable by meaning. Call embed_query_tool with the search "
            f"phrase, then put `{partner.name} {dist_op} $n` in the WHERE "
            f"clause and pass the returned token as $n.{threshold} Order by the "
            f"same expression. Exact matching on {column.name} with = or ILIKE "
            f"still works - use that when the question names a literal value."
        )

    if kind is FilterKind.OPERATOR:
        return (
            f"Numeric ({column.sql_type}). Compare with =, <, >, <=, >= or "
            f"BETWEEN, and aggregate with count, sum, avg, min, max. No tool "
            f"call is needed before using it."
        )

    if kind is FilterKind.DATETIME:
        return (
            f"Date/time ({column.sql_type}). Filter with a half-open range - "
            f"`{column.name} >= $n AND {column.name} < $n` - rather than "
            f"equality, and pass both bounds as parameters. now() and interval "
            f"arithmetic are available for relative dates."
        )

    if kind is FilterKind.ENUM:
        if enum_values:
            listed = ", ".join(str(v) for v in enum_values)
            return (
                f"One of a short list of values: {listed}. Match exactly with = "
                f"against one of those - do not invent a spelling."
            )
        return (
            f"A short list of distinct values. Call "
            f"get_lsit_values('{table.name}', '{column.name}') before "
            f"filtering, then match exactly with = against a value it returns, "
            f"so you match what is stored rather than what the question called "
            f"it."
        )

    if kind is FilterKind.TEXT:
        return (
            f"Free text with many distinct values. Match with = for a whole "
            f"value or ILIKE '%...%' for a fragment. Do not call "
            f"get_lsit_values on it - there are too many values for the list to "
            f"be useful."
        )

    raise ValueError(f"No guidance defined for {kind}")