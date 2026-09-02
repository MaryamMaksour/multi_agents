from typing import Any

import asyncpg

from domain.exceptions import DatabaseError


class PostgresDatabaseAdapter():

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def fetch(self, query: str, *params: Any) -> list[dict]:
        """Execute a SELECT query and return the results as a list of dictionaries.

        The query runs inside a READ ONLY transaction that is always rolled
        back. A read needs no commit, and the rollback is what makes the
        connection safe to return to the pool: every GUC change made during
        the transaction is discarded with it, including `role`. Without this,
        a model-written `SELECT set_config('role', <other agent>, false)`
        passes the validator (it names only allowed tables) and leaves the
        pooled connection running as another agent for every later query -
        `RESET ALL` on release does not touch `role`.

        Raises:
            DatabaseError: if the database call fails.
        """
        try:
            async with self._pool.acquire() as conn:
                transaction = conn.transaction(readonly=True)
                await transaction.start()
                try:
                    records = await conn.fetch(query, *params)
                finally:
                    await transaction.rollback()
                return [dict(record) for record in records]

        except Exception as e:
            raise DatabaseError(f"Error {e}  while executing query: {query}") from e

    async def execute(self, query: str, *params: Any) -> None:
        """Execute an INSERT, UPDATE, or DELETE statement.
            Used for History Port
        Raises:
            DatabaseError: if the database call fails.
        """
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(query, *params)

        except Exception as e:
            raise DatabaseError(f"Error {e}  while executing query: {query}") from e
