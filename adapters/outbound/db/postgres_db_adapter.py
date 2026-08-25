from typing import Any

import asyncpg

from domain.exceptions import DatabaseError


class PostgresDatabaseAdapter():

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def fetch(self, query: str, *params: Any) -> list[dict]:
        """Execute a SELECT query and return the results as a list of dictionaries.

        Raises:
            DatabaseError: if the database call fails.
        """
        
        try:
            async with self._pool.acquire() as conn:
                records = await conn.fetch(query, *params)  # return -> asyncpg.Record objects
                dict_records = [dict(record) for record in records]
                return dict_records

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
        