"""Where schema knowledge comes from.

The core does not ship a schema. It asks for one. Any implementation that can
answer these three questions works - Postgres introspection is the one built
here, but a test hands back a fixed schema with no database at all.
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.table_schema import TableSchema


class SchemaPort(Protocol):

    async def list_tables(self) -> tuple[str, ...]:
        """Every table the current connection can read.

        Scoped by the connection's own privileges, so for an agent connecting
        as its own role this is already that agent's allowlist - there is no
        separate list to keep in step with the GRANTs.

        Raises:
            DatabaseError: if the database call fails.
        """
        ...

    async def describe(self, tables: tuple[str, ...]) -> dict[str, TableSchema]:
        """Columns and types for the named tables.

        Takes several at once because this runs at agent startup and one
        round trip for the whole schema is meaningfully faster than one per
        table. Tables the connection cannot read are absent from the result
        rather than raising: not being able to see a table is an answer.

        Raises:
            DatabaseError: if the database call fails.
        """
        ...

    async def distinct_count(self, table: str, column: str) -> int:
        """How many distinct values a column holds.

        Separate from describe() because it reads data rather than the
        catalogue - it is the one expensive call here, and callers should be
        able to skip it. Backs the ENUM decision in the classifier.

        Raises:
            DatabaseError: if the database call fails.
        """
        ...

    async def distinct_values(self, table: str, column: str, limit: int) -> tuple[str, ...]:
        """The values a column actually holds, up to `limit` of them.

        Only worth asking once distinct_count says there are few, which is
        why it is a second call rather than part of the first: reading the
        values of a column with four hundred of them is a cost with no use.

        What it buys is the difference between telling a model that a column
        has few distinct values and telling it which ones. A question about
        "روايات" cannot be turned into `genre = 'novel'` by a model that has
        never been shown that 'novel' is one of the ten things that column
        contains.

        Raises:
            DatabaseError: if the database call fails.
        """
        ...
