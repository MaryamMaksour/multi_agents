"""SchemaPort over Postgres catalogue tables.

Implements SchemaPort by reading information_schema through whatever
connection it is given. That last part carries the security property: because
information_schema only reports objects the current user holds a privilege
on, connecting as the agent's own role makes the result self-scoping.

The queries below were run against seeds/001_schema.sql and produce the
intended classification - they are here so they do not have to be re-derived,
not as a finished implementation.

    -- list_tables()
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;

    -- describe(): columns plus the embed_<col> pairing that marks a text
    -- column as semantically searchable. Note udt_name, not data_type:
    -- pgvector columns report data_type = 'USER-DEFINED'.
    WITH cols AS (
        SELECT table_name, column_name, data_type, udt_name, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = ANY($1)
    ),
    vec AS (
        SELECT table_name, substring(column_name from 7) AS base
        FROM cols
        WHERE column_name LIKE 'embed\\_%' AND udt_name = 'vector'
    )
    SELECT c.table_name, c.column_name, c.data_type, c.udt_name,
           c.is_nullable, (v.base IS NOT NULL) AS has_embedding
    FROM cols c
    LEFT JOIN vec v
      ON v.table_name = c.table_name AND v.base = c.column_name
    ORDER BY c.table_name, c.column_name;

    -- distinct_count(): the only call that reads data. Identifiers cannot be
    -- parameterised, so both names MUST go through the same identifier
    -- validation the SQL tool uses before being interpolated.
    SELECT count(DISTINCT {column}) FROM {table};
"""

from __future__ import annotations


class PostgresIntrospectionAdapter:
    """Reads table and column structure from the Postgres catalogue.

    Constructor takes:
        db      DatabasePort  - reuse it rather than a raw pool, so this
                                adapter stays testable and does not open a
                                second connection path
        schema  str           - which Postgres schema to inspect, default
                                'public'. Not hardcoded: a deployment may
                                keep its tables elsewhere.

    TODO: implement list_tables / describe / distinct_count.

    TODO: distinct_count interpolates identifiers into SQL. Validate both
    against the identifier pattern first and reject anything that fails -
    this is the one place in the adapter where injection is possible.

    TODO: on a large table, count(DISTINCT col) is a full scan. Either sample
    (TABLESAMPLE, or a LIMIT subquery) or read reltuples/n_distinct from
    pg_stats, which is free but only as fresh as the last ANALYZE. Decide
    which, and say so here - a wrong ENUM classification degrades filtering
    quality silently.
    """

    # TODO: implement
    ...
