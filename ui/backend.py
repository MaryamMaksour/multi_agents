"""Everything the console asks the system to do, in one file.

This is the seam. app.py draws screens and calls these functions; nothing in
app.py imports from domain/ or adapters/ directly. So wiring a screen to real
code means replacing a function body here, and never touching the UI.

Each function says whether it is REAL or a STUB, and a stub names the feature
that will make it real. That labelling is not decoration - a console that
quietly fakes an answer is worse than no console, because it teaches you the
system works when it does not.

The console is not part of the architecture. It is a client, like any other
caller of the orchestrator, and lives outside the hexagon on purpose.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Streamlit is launched from wherever the user happens to be, so the repo
# root has to be put on the path explicitly for `domain` and `adapters` to
# import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domain.entities.table_schema import TableSchema  # noqa: E402


@dataclass
class Connection:
    """Where the console is pointed. Matches deploy/docker-compose.dev.yml."""

    host: str = "localhost"
    port: int = 55432
    database: str = "library_dev"
    user: str = "dev"
    password: str = "dev"


@dataclass
class AgentDraft:
    """An agent as the console collects it, before any registry exists.

    Deliberately the same shape as ProviderSpec's new fields, so wiring this
    to the registry later is a mapping and not a redesign.
    """

    key: str
    display_name: str
    description: str
    prompt: str
    tables: list[str] = field(default_factory=list)
    db_role: str = ""
    status: str = "pending"


def _run(coro):
    """Run one coroutine to completion, on a fresh event loop.

    Streamlit is synchronous and every adapter here is async, so each action
    gets its own loop. That also settles a subtler problem: an asyncpg pool
    belongs to the loop that created it, so a pool cached between reruns
    would break the moment Streamlit called it from a new one. Connecting per
    action is slower and correct, which for a development console is the
    right trade.
    """
    return asyncio.run(coro)


# ==========================================================================
# REAL - schema introspection
#
# Uses PostgresIntrospectionAdapter, the same code the agents will use. What
# appears on the Tables screen is read out of the database, not described to
# it: the column list, the types, and which columns are semantically
# searchable all come from the catalogue.
# ==========================================================================


async def _introspect(conn: Connection) -> dict[str, TableSchema]:
    import asyncpg

    from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter
    from adapters.outbound.schema.postgres_introspection_adapter import (
        PostgresIntrospectionAdapter,
    )

    pool = await asyncpg.create_pool(
        host=conn.host, port=conn.port, database=conn.database,
        user=conn.user, password=conn.password,
        min_size=1, max_size=2, timeout=5,
    )
    try:
        schema = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))
        tables = await schema.list_tables()
        return await schema.describe(tables)
    finally:
        await pool.close()


def introspect(conn: Connection) -> dict[str, TableSchema]:
    """Every table this connection can read, with its columns. REAL.

    Scoped by the connecting role's own privileges - connect as app_catalog
    and only that agent's tables come back. That is the whole design in one
    observable behaviour, so it is worth trying both roles from the sidebar.
    """
    return _run(_introspect(conn))


async def _distinct_count(conn: Connection, table: str, column: str) -> int:
    import asyncpg

    from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter
    from adapters.outbound.schema.postgres_introspection_adapter import (
        PostgresIntrospectionAdapter,
    )

    pool = await asyncpg.create_pool(
        host=conn.host, port=conn.port, database=conn.database,
        user=conn.user, password=conn.password, min_size=1, max_size=2, timeout=5,
    )
    try:
        schema = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))
        return await schema.distinct_count(table, column)
    finally:
        await pool.close()


def distinct_count(conn: Connection, table: str, column: str) -> int:
    """How many distinct values a column holds. REAL.

    This is the one number the ENUM half of the classifier turns on, which is
    why the Tables screen offers it per column: the cutoff is easier to
    choose after seeing the real spread.
    """
    return _run(_distinct_count(conn, table, column))


# ==========================================================================
# STUB - column classification
#
# Becomes real with: feature 2 (filter_classifier).
#
# The rules below are the intended precedence, applied here only so the
# Tables screen can show the shape of the answer. Real classification also
# needs the distinct counts, which this does not fetch.
# ==========================================================================

_NUMERIC = {"integer", "bigint", "smallint", "numeric", "real", "double precision"}
_DATETIME = {"date", "timestamp with time zone", "timestamp without time zone", "time"}


def classify_preview(table: TableSchema, column_name: str) -> str:
    """A provisional filter kind for one column. STUB.

    Follows the intended order - vector, then semantic, then type, then text -
    but stops short of ENUM, which needs a count this does not take. Feature 2
    replaces this with the real classifier and the guidance text the model
    actually reads.
    """
    column = table.column(column_name)
    if column is None:
        return "—"
    if column.is_vector:
        return "VECTOR_STORAGE"
    if table.embedding_partner(column.name) is not None:
        return "SEMANTIC"
    if column.sql_type in _NUMERIC:
        return "OPERATOR"
    if column.sql_type in _DATETIME:
        return "DATETIME"
    return "TEXT or ENUM"


# ==========================================================================
# STUB - the agent registry
#
# Becomes real with: feature 3 (FileAgentRegistryAdapter), then a table in
# phase 4. Drafts live in Streamlit's session state, so they are gone when
# the browser tab closes - nothing here is persisted yet.
# ==========================================================================


def save_agent(drafts: list[AgentDraft], draft: AgentDraft) -> list[AgentDraft]:
    """Add or replace an agent draft. STUB - in memory only.

    Feature 3 writes these to seeds/agents.example.json through
    FileAgentRegistryAdapter; phase 4 moves them to a table and has the
    provisioner create the matching Postgres role.
    """
    remaining = [d for d in drafts if d.key != draft.key]
    return remaining + [draft]


def delete_agent(drafts: list[AgentDraft], key: str) -> list[AgentDraft]:
    """Remove an agent draft. STUB - in memory only.

    One line here, and genuinely hard later. In phase 4 an agent owns a
    Postgres role, and DROP ROLE fails while that role still holds a
    privilege or owns an object - so the real path is REASSIGN OWNED, then
    DROP OWNED, then DROP ROLE, run by the provisioner with credentials the
    request path never has.

    Two things will also have to happen before the role goes: the agent stops
    being routable, so the orchestrator does not offer a tool it can no
    longer call, and its history is dealt with deliberately rather than
    cascading away with the role.
    """
    return [d for d in drafts if d.key != key]


def suggested_role_name(agent_key: str) -> str:
    """The Postgres role an agent would connect as. STUB - naming only.

    Nothing creates this role. In phase 4 the provisioner does, out of the
    request path, because CREATE ROLE and GRANT need privileges no service
    handling requests should hold.
    """
    return f"app_{agent_key}" if agent_key else ""


# ==========================================================================
# STUB - asking a question
#
# Becomes real with: feature 4 and phase 2 (the orchestrator's inbound
# adapter and composition root). This returns a fixed trace so the delegation
# flow is visible before any of it runs: which agent the orchestrator picks,
# what the sub-agent is asked, what comes back, and what the user finally
# sees.
#
# No model is called. No SQL runs. Every value below is written by hand.
# ==========================================================================


def ask(question: str, agents: list[AgentDraft]) -> dict[str, Any]:
    """Answer a question by delegating to a sub-agent. STUB.

    The shape is the real one - RunAgentTurn returns messages plus pagination
    state per tool, and the orchestrator's tools are one per registered agent -
    so wiring this later means replacing the body, not the caller.
    """
    target = agents[0] if agents else None

    if target is None:
        return {
            "steps": [],
            "answer": "No agents registered yet. Define one on the Agents tab first.",
            "stub": True,
        }

    return {
        "steps": [
            {
                "kind": "route",
                "detail": (
                    f"Orchestrator read the descriptions of {len(agents)} agent(s) "
                    f"and chose **{target.key}**."
                ),
            },
            {
                "kind": "delegate",
                "detail": (
                    f"Called `{target.key}` with a self-contained question. "
                    "Sub-agents keep no history, so every reference is resolved first."
                ),
                "payload": {"query": question, "cursor": None},
            },
            {
                "kind": "sql",
                "detail": (
                    f"`{target.key}` asked for the schema and the filter kinds, then "
                    "wrote one query combining every condition."
                ),
                "payload": {
                    "query": "SELECT title_en, page_count\n"
                             "FROM books\n"
                             "WHERE page_count < $1\n"
                             "  AND language = $2\n"
                             "LIMIT $3 OFFSET $4",
                    "params": [300, "Arabic", 10, 0],
                    "note": "hand-written placeholder - no model wrote this and it never ran",
                },
            },
            {
                "kind": "result",
                "detail": "Sub-agent returned one page.",
                "payload": {"rows": 10, "has_more": True, "next_cursor": "…"},
            },
        ],
        "answer": (
            "This is a placeholder answer. The trace above shows the path a real "
            "question will take once the orchestrator is wired up - it is not the "
            "result of running anything."
        ),
        "stub": True,
    }
