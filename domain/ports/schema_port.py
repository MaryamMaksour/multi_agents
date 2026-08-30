"""Where schema knowledge comes from.

The core does not ship a schema. It asks for one. Any implementation that can
answer these three questions works - Postgres introspection is the one we
build, but a test can hand back a fixed schema with no database at all.
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.table_schema import TableSchema


class SchemaPort(Protocol):
    """Discovers database structure at runtime.

    Methods to define:

        async def list_tables() -> tuple[str, ...]
            Every table the current connection can read. Scoped by the
            connection's own privileges, so this doubles as the agent's
            allowlist rather than needing one.

        async def describe(table: str) -> TableSchema
            Columns, types, and which of them are vector columns.

        async def distinct_count(table: str, column: str) -> int
            Backs the ENUM decision in the classifier. Separate from
            describe() because it reads data, not catalogue - it is the one
            expensive call here and callers should be able to skip it.

    TODO: implement as a Protocol - structural typing, no inheritance needed
    in adapters, same as the other ports.

    TODO: decide whether describe() takes one table or a tuple. One round
    trip for all tables is meaningfully faster at agent startup, and startup
    is the only place this is called if the result is cached.
    """

    ...
