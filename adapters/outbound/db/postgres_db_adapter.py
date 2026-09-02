"""The database, behind DatabasePort, with every statement accounted for.

Two pools use this adapter and they are not the same thing. An agent's pool
has already run SET ROLE and can read only that agent's tables; the service's
pool is the authenticator and can write history and nothing else. Neither
distinction is visible from here, which is the point - the privilege lives in
the GRANT, not in this code.

What is visible from here is timing, and it is the timing that explains the
turns that feel slow. A question is roughly seven model calls and a handful
of queries; when one of those queries is a sequential scan over an unindexed
column, nothing else in the system says so.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg

from domain.exceptions import DatabaseError
from libs.agent_core.logging_setup import Timer, log_event

logger = logging.getLogger(__name__)

# Above this, a query is logged at WARNING with its text. Not a timeout and
# not a limit - just the line that turns "the agent is slow" into a specific
# statement to look at. Well under DB_COMMAND_TIMEOUT, which is the real
# ceiling and kills the query rather than describing it.
SLOW_QUERY_MS = 1000.0


def _one_line(query: str, limit: int = 400) -> str:
    """A statement as a single log field.

    The SQL in this system is written across several lines, and a log line
    that becomes eight breaks every tool that reads logs by the line - and
    reads, to a person scanning, as eight events rather than one.
    """
    collapsed = " ".join(query.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + " ..."


class PostgresDatabaseAdapter:

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def fetch(self, query: str, *params: Any) -> list[dict]:
        """Execute a SELECT query and return the results as a list of dictionaries.

        Raises:
            DatabaseError: if the database call fails.
        """
        # Parameter values at DEBUG only. They are not secrets - they came
        # from the model, which got them from the question - but they are the
        # user's words, and a log kept for a month is a different place for
        # those than a log read during a run.
        log_event(logger, "db.query", level=logging.DEBUG,
                  sql=_one_line(query), params=len(params))

        try:
            with Timer() as timer:
                async with self._pool.acquire() as conn:
                    records = await conn.fetch(query, *params)
                    dict_records = [dict(record) for record in records]

        except Exception as e:
            # The event line names the error type for counting; the exc_info
            # line carries the frame. asyncpg's message says which constraint
            # or which permission, and that is usually the whole diagnosis -
            # "permission denied for table loans" is an agent reaching outside
            # its GRANT, which is a registry mistake, not a database problem.
            log_event(logger, "db.error", level=logging.ERROR,
                      sql=_one_line(query), error=type(e).__name__, detail=str(e))
            logger.error("query failed", exc_info=True)
            raise DatabaseError(f"Error {e}  while executing query: {query}") from e

        log_event(
            logger, "db.rows",
            level=logging.WARNING if timer.ms >= SLOW_QUERY_MS else logging.DEBUG,
            rows=len(dict_records), ms=timer.ms,
            # Only when it is slow: at DEBUG the statement is already on the
            # db.query line above, and repeating it doubles every query's
            # output for no information.
            **({"sql": _one_line(query)} if timer.ms >= SLOW_QUERY_MS else {}),
        )
        return dict_records

    async def execute(self, query: str, *params: Any) -> None:
        """Execute an INSERT, UPDATE, or DELETE statement.
            Used for History Port

        Raises:
            DatabaseError: if the database call fails.
        """
        try:
            with Timer() as timer:
                async with self._pool.acquire() as conn:
                    await conn.execute(query, *params)

        except Exception as e:
            log_event(logger, "db.error", level=logging.ERROR,
                      sql=_one_line(query), error=type(e).__name__, detail=str(e))
            logger.error("statement failed", exc_info=True)
            raise DatabaseError(f"Error {e}  while executing query: {query}") from e

        log_event(
            logger, "db.execute",
            level=logging.WARNING if timer.ms >= SLOW_QUERY_MS else logging.DEBUG,
            sql=_one_line(query), ms=timer.ms,
        )
