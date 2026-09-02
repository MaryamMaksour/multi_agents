"""Startup: turn a connected database into everything SqlToolAdapter needs.

The classifier is a pure function over an already-introspected schema. This
is the half that talks to the database - it introspects, decides which
columns are worth a cardinality probe, runs those probes, and hands back the
three arguments the adapter takes.

Kept separate from filter_classifier.py so that file stays pure and testable
without a database, and separate from the adapter so the adapter keeps taking
plain dictionaries and never learns where they came from.

Two things this module decides, and both are about cost:

`SELECT count(DISTINCT col)` is the only call here that reads data rather
than the catalogue, and on a large table it is a sequential scan. So it is
run for the smallest possible set of columns: the ones whose classification
could actually change because of it. That set is not restated here - it is
derived by asking the classifier what a column would be *without* a count.
Only TEXT can become ENUM, so only TEXT is probed.

The columns that turn out to be enums get one more query each, to read the
values themselves. That is the difference between telling a model a column
has a short list and telling it which values are on it - and in practice the
difference between a model filtering on `genre` and silently leaving it out.
Same derivation: a column is read when the classifier, given the count just
measured, calls it ENUM.

And a probe that fails is not a startup failure. ENUM is an optimisation; a
column whose count could not be read is classified TEXT, which is what it
would have been anyway had nobody asked. The names are reported rather than
swallowed, so a caller can say why the guidance is thinner than expected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from domain.entities.column_filter import ENUM_MAX_DISTINCT, ColumnFilter, FilterKind
from domain.entities.table_schema import TableSchema
from domain.exceptions import DatabaseError
from domain.ports.schema_port import SchemaPort
from libs.agent_core.filter_classifier import (
    DEFAULT_DIST_OP,
    classify_column,
    classify_table,
)
from libs.agent_core.logging_setup import log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentSchema:
    """One agent's view of the database, ready to hand to SqlToolAdapter.

    `tables`, `schema` and `filters` map onto the adapter's constructor
    arguments of the same names. `classified` is the same information as
    `filters` before it was flattened to strings - nothing in the adapter
    wants it, but a console that shows a column's kind does, and recomputing
    it would mean probing again.
    """

    tables: tuple[str, ...]
    schema: dict[str, dict[str, str]]
    filters: dict[str, dict[str, str]]
    classified: dict[str, dict[str, ColumnFilter]]
    unprobed: tuple[str, ...] = ()


def render_columns(
    table: TableSchema, enum_values: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """The column block the model reads, in the shape the adapter parses.

    `_extract_column_names` reads this back with a `name type` regex when it
    validates a column in get_list_values, so the first two fields on each
    line are load-bearing and anything after them is for the model.

    The enum values go here as well as in the filter guidance, and the
    duplication is the point. `get_filter` is pull-based: a model asks about
    the columns it already intends to filter on, so guidance placed only
    there never reaches the decision about *whether* to use a column. Asked
    for novels, a model filtered on language and left `genre` out entirely,
    because at the moment it was choosing columns all it had been shown was
    `genre text`.

    The schema is the surface it always reads, and reads first. A column that
    says what it contains is a column that can be chosen.
    """
    enum_values = enum_values or {}
    lines = []
    for column in table.columns:
        line = f"{column.name} {column.sql_type}"
        if not column.nullable:
            line += " NOT NULL"
        notes = []
        if column.references:
            notes.append(f"joins to {column.references}")
        values = enum_values.get(column.name)
        if values:
            notes.append("one of: " + ", ".join(values))
        if notes:
            line += "  -- " + "; ".join(notes)
        lines.append(line)
    return "\n".join(lines)


def columns_needing_a_count(table: TableSchema) -> tuple[str, ...]:
    """Which columns are worth a distinct-count probe.

    Asks the classifier what each column would be with no count at all. Only
    TEXT can be moved by one - a vector column, a semantic column, a numeric
    and a date are all settled by earlier rules - so only TEXT is probed.

    Derived rather than restated on purpose: written out as "text columns
    without an embedding partner" this would be a second copy of the
    precedence chain, free to drift from the real one.
    """
    return tuple(
        c.name for c in table.columns
        if classify_column(table, c, None) is FilterKind.TEXT
    )


async def read_enum_values(
    schema_port: SchemaPort, table: TableSchema, counts: dict[str, int],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Read the values of the columns that turned out to be enums.

    A second pass rather than part of the first, and derived rather than
    restated: a column is worth reading when classify_column, given the count
    just measured, calls it ENUM. So the rule for "few enough to list" stays
    in one place, and the cap is the classifier's cutoff rather than a second
    number that could drift from it.

    Costs one query per enum column at startup - `genre` and `language` on
    the development schema, two queries. Failures are per column and never
    raised: a column whose values could not be read still classifies as ENUM,
    and its guidance falls back to naming the tool that lists them.
    """
    values: dict[str, tuple[str, ...]] = {}
    unread: list[str] = []

    for column in table.columns:
        if classify_column(table, column, counts.get(column.name)) is not FilterKind.ENUM:
            continue
        try:
            values[column.name] = await schema_port.distinct_values(
                table.name, column.name, ENUM_MAX_DISTINCT
            )
        except DatabaseError:
            # Per column and never raised - but a column that classifies as
            # ENUM and whose values could not be read is a column the model
            # will be told to call get_list_values on, and that call will fail
            # the same way. Worth a line.
            unread.append(f"{table.name}.{column.name}")
            log_event(logger, "startup.enum_unreadable", level=logging.WARNING,
                      table=table.name, column=column.name)

    if values:
        log_event(logger, "startup.enums", level=logging.DEBUG, table=table.name,
                  columns={name: list(vals) for name, vals in values.items()})
    return values, tuple(unread)


async def find_empty_vector_columns(
    schema_port: SchemaPort, table: TableSchema,
) -> frozenset[str]:
    """Which of this table's vector columns hold nothing at all.

    The failure this exists for produces a confident wrong answer and no error
    anywhere. A vector column that was created and never filled makes its
    partner classify as SEMANTIC, so the model is told it may search it, and

        ORDER BY embed_title_en <=> $1 LIMIT 10

    comes back with zero rows - because a pgvector index does not index NULL.
    The model reports that there are none. On this development database every
    vector column is empty, so every semantic search returns nothing and looks
    like an answer.

    One LIMIT 1 query per vector column, which stops at the first row it
    finds, so this costs nothing on a filled column and nothing on an empty
    one either.

    A probe that fails is treated as filled. Being unable to check is not
    evidence of emptiness, and downgrading a working column on a transient
    error would quietly remove the feature.
    """
    empty: set[str] = set()

    for column in table.columns:
        if not column.is_vector:
            continue
        try:
            if not await schema_port.has_any_value(table.name, column.name):
                empty.add(column.name)
                log_event(logger, "startup.vector_empty", level=logging.WARNING,
                          table=table.name, column=column.name)
        except DatabaseError:
            log_event(logger, "startup.vector_uncheckable", level=logging.WARNING,
                      table=table.name, column=column.name)

    return frozenset(empty)


async def count_distinct_values(
    schema_port: SchemaPort, table: TableSchema,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Probe the columns that need it. Returns the counts and the failures.

    A failure is per column and never raised: one unreadable column must not
    stop an agent from starting, it just leaves that column as TEXT.
    """
    counts: dict[str, int] = {}
    unprobed: list[str] = []

    for column in columns_needing_a_count(table):
        try:
            counts[column] = await schema_port.distinct_count(table.name, column)
        except DatabaseError:
            # The column stays TEXT, which is a silent downgrade: its guidance
            # stops naming the values tool, so the model filters on it by
            # guessing at spellings instead of matching one it was shown.
            unprobed.append(f"{table.name}.{column}")
            log_event(logger, "startup.probe_failed", level=logging.WARNING,
                      table=table.name, column=column)

    return counts, tuple(unprobed)


async def load_agent_schema(
    schema_port: SchemaPort,
    *,
    tables: tuple[str, ...] | None = None,
    dist_op: str = DEFAULT_DIST_OP,
    probe_cardinality: bool = True,
) -> AgentSchema:
    """Introspect, classify, and return the adapter's three arguments.

    `tables` defaults to every table the connection can read. For an agent
    connecting as its own Postgres role that is already its allowlist, which
    is the point of the design: the GRANTs are the single source of truth and
    there is no table list in code to drift from them. Passing it explicitly
    is for a caller that wants a subset, not for widening one.

    `probe_cardinality=False` skips every distinct-count call. Everything
    still works; ENUM columns come back as TEXT and their guidance stops
    naming the values tool. Useful against a large database where the probes
    are the slow part of startup.
    """
    names = tables if tables is not None else tuple(await schema_port.list_tables())
    described = await schema_port.describe(tuple(names))

    schema: dict[str, dict[str, str]] = {}
    filters: dict[str, dict[str, str]] = {}
    classified: dict[str, dict[str, ColumnFilter]] = {}
    unprobed: list[str] = []
    unfilled: list[str] = []

    for table in described.values():
        key = table.name.lower()

        if probe_cardinality:
            counts, failed = await count_distinct_values(schema_port, table)
            unprobed.extend(failed)
            values, unread = await read_enum_values(schema_port, table, counts)
            unprobed.extend(unread)
            empty_vectors = await find_empty_vector_columns(schema_port, table)
            unfilled.extend(f"{table.name}.{name}" for name in sorted(empty_vectors))
        else:
            counts, values, empty_vectors = {}, {}, frozenset()

        column_filters = classify_table(table, counts, dist_op, values, empty_vectors)

        schema[key] = {"columns": render_columns(table, values)}
        classified[key] = column_filters
        filters[key] = {name: cf.guidance for name, cf in column_filters.items()}

    if unfilled:
        # The loudest line this system writes at startup, and it earns it.
        # Every semantic search against these columns returns nothing and
        # reads as an answer. The columns still work as text - they classify
        # as TEXT now, where ILIKE finds what a vector search could not - but
        # the feature is off until something fills them.
        log_event(logger, "startup.vectors_unfilled", level=logging.WARNING,
                  count=len(unfilled), columns=unfilled[:20],
                  note="semantic search is disabled on these; they are "
                       "searchable as text. Run scripts/backfill_embeddings.py.")

    if unprobed:
        # One summary line as well as the per-column ones, because this is the
        # number that says how much of the schema the agent is working blind
        # on - and a startup with twenty of these is a permissions problem,
        # not twenty separate accidents.
        log_event(logger, "startup.unprobed", level=logging.WARNING,
                  count=len(unprobed), columns=unprobed[:20])

    return AgentSchema(
        tables=tuple(sorted(schema)),
        schema=schema,
        filters=filters,
        classified=classified,
        unprobed=tuple(unprobed),
    )
