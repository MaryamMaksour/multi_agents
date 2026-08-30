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

And a probe that fails is not a startup failure. ENUM is an optimisation; a
column whose count could not be read is classified TEXT, which is what it
would have been anyway had nobody asked. The names are reported rather than
swallowed, so a caller can say why the guidance is thinner than expected.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.entities.column_filter import ColumnFilter, FilterKind
from domain.entities.table_schema import TableSchema
from domain.exceptions import DatabaseError
from domain.ports.schema_port import SchemaPort
from libs.agent_core.filter_classifier import (
    DEFAULT_DIST_OP,
    classify_column,
    classify_table,
)


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


def render_columns(table: TableSchema) -> str:
    """The column block the model reads, in the shape the adapter parses.

    `_extract_column_names` reads this back with a `name type` regex when it
    validates a column in get_lsit_values, so the first two fields on each
    line are load-bearing and anything after them is for the model.
    """
    return "\n".join(
        f"{c.name} {c.sql_type}" + ("" if c.nullable else " NOT NULL")
        for c in table.columns
    )


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
            unprobed.append(f"{table.name}.{column}")

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

    for table in described.values():
        key = table.name.lower()

        if probe_cardinality:
            counts, failed = await count_distinct_values(schema_port, table)
            unprobed.extend(failed)
        else:
            counts = {}

        column_filters = classify_table(table, counts, dist_op)

        schema[key] = {"columns": render_columns(table)}
        classified[key] = column_filters
        filters[key] = {name: cf.guidance for name, cf in column_filters.items()}

    return AgentSchema(
        tables=tuple(sorted(schema)),
        schema=schema,
        filters=filters,
        classified=classified,
        unprobed=tuple(unprobed),
    )
