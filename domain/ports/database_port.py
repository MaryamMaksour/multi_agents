from typing import Any, Protocol


class DatabasePort(Protocol):

    async def fetch(self, query: str, *params: Any) -> list[dict]:
        """Execute a SELECT query and return the results as a list of dictionaries.

        Raises:
            DatabaseError: if the database call fails.
        """
        ...

    async def execute(self, query: str, *params: Any) -> None:
        """Execute an INSERT, UPDATE, or DELETE query.

        Raises:
            DatabaseError: if the database call fails.
        """
        ...