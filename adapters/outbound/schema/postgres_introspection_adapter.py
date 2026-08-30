"""SchemaPort over the Postgres catalogue.

Reads information_schema through whatever connection it is given, and that
last part carries the security property this design rests on:
information_schema only reports objects the current user holds a privilege
on, so connecting as an agent's own role makes the result self-scoping. The
GRANTs are the only place the agent-to-table mapping is written down.

Verified against seeds/001_schema.sql: introspecting as app_catalog reports
authors, books and publishers; as app_circulation, books, branches, loans and
members. Neither sees the other's tables, and neither needed a list in code.
"""

from __future__ import annotations

from domain.entities.table_schema import ColumnSchema, TableSchema
from domain.exceptions import DatabaseError
from domain.ports.database_port import DatabasePort
from libs.agent_core.sql_validation import validate_identifier

# Extension types all report data_type = 'USER-DEFINED'; udt_name is what
# actually identifies them. Recognising pgvector is a Postgres detail, so it
# is decided here and reaches the domain as ColumnSchema.is_vector.
_VECTOR_UDT = "vector"


class PostgresIntrospectionAdapter:
    """Reads table and column structure from the Postgres catalogue.

    Takes a DatabasePort rather than a raw pool: it needs no capability that
    port lacks, and depending on the port keeps it testable and stops a second
    connection path from appearing.
    """

    def __init__(self, db: DatabasePort, schema: str = "public"):
        self._db = db
        # Not hardcoded to public - a deployment may keep its tables
        # elsewhere, and this is the kind of assumption that is invisible
        # until it is wrong.
        self._schema = schema

    async def list_tables(self) -> tuple[str, ...]:
        rows = await self._db.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = $1
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            self._schema,
        )
        return tuple(row["table_name"] for row in rows)

    async def describe(self, tables: tuple[str, ...]) -> dict[str, TableSchema]:
        """Columns and types for the named tables, in one round trip.

        Tables the connection cannot read simply do not come back, so they
        are absent from the result rather than present and empty - an empty
        TableSchema would read as "a table with no columns", which is a
        different and misleading claim.
        """
        wanted = [t.lower() for t in tables]
        if not wanted:
            return {}

        rows = await self._db.fetch(
            """
            SELECT table_name, column_name, data_type, udt_name, is_nullable
            FROM information_schema.columns
            WHERE table_schema = $1
              AND table_name = ANY($2::text[])
            ORDER BY table_name, ordinal_position
            """,
            self._schema,
            wanted,
        )

        grouped: dict[str, list[ColumnSchema]] = {}
        for row in rows:
            grouped.setdefault(row["table_name"], []).append(
                ColumnSchema(
                    name=row["column_name"],
                    # For a vector column data_type is the unhelpful
                    # 'USER-DEFINED', so report udt_name instead - the
                    # classifier and any error message want the real name.
                    sql_type=(
                        row["udt_name"]
                        if row["data_type"] == "USER-DEFINED"
                        else row["data_type"]
                    ),
                    is_vector=row["udt_name"] == _VECTOR_UDT,
                    nullable=row["is_nullable"] == "YES",
                )
            )

        return {
            name: TableSchema(name=name, columns=tuple(columns))
            for name, columns in grouped.items()
        }

    async def distinct_count(self, table: str, column: str) -> int:
        """How many distinct values a column holds.

        Identifiers cannot be parameterised, so both names go through
        validate_identifier before being interpolated. This is the only place
        in the adapter where injection is possible, and the check raises
        rather than returning a flag so it cannot be ignored by accident.
        """
        try:
            table_id = validate_identifier(table)
            column_id = validate_identifier(column)
        except ValueError as e:
            raise DatabaseError(f"Refusing to introspect: {e}") from e

        rows = await self._db.fetch(
            f'SELECT count(DISTINCT "{column_id}") AS n FROM "{self._schema}"."{table_id}"'
        )
        return int(rows[0]["n"]) if rows else 0
