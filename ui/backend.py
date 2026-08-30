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
import os
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
# REAL - column classification
#
# Uses load_agent_schema, the same call the composition root will make. What
# the Tables screen shows is what the model will be told: the same kinds, the
# same guidance sentences, produced by the same code.
#
# Probing is on, so this reads data - a count(DISTINCT) per text column. On
# the development database that is milliseconds; on a large one it is the
# slow part, which is what probe_cardinality=False exists for.
# ==========================================================================


async def _classify(conn: Connection, dist_op: str = "<=>"):
    import asyncpg

    from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter
    from adapters.outbound.schema.postgres_introspection_adapter import (
        PostgresIntrospectionAdapter,
    )
    from libs.agent_core.schema_bootstrap import load_agent_schema

    pool = await asyncpg.create_pool(
        host=conn.host, port=conn.port, database=conn.database,
        user=conn.user, password=conn.password, min_size=1, max_size=2, timeout=5,
    )
    try:
        schema = PostgresIntrospectionAdapter(PostgresDatabaseAdapter(pool))
        return await load_agent_schema(schema, dist_op=dist_op)
    finally:
        await pool.close()


def classify(conn: Connection, dist_op: str = "<=>"):
    """Every table's columns, classified, with the guidance text. REAL.

    Returns an AgentSchema: `.classified[table][column]` is a ColumnFilter,
    and `.unprobed` names any column whose distinct count could not be read -
    those fall back to TEXT rather than failing the load.
    """
    return _run(_classify(conn, dist_op))


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


def is_provisioned(draft: AgentDraft) -> bool:
    """Whether this agent has a Postgres role behind it.

    The line the console draws everywhere else. Before it, an agent is text
    in a form and anything can change. After it, a role exists with grants
    on real tables, and the fields naming those - key, db_role, tables -
    stop being editable, because changing them is revoking and re-granting
    rather than saving a form.

    Description and prompt stay editable on either side: they are text, and
    the GRANT is the boundary, so no wording can widen what the agent reads.
    That matters more than it sounds - a prompt is never right the first
    time, and freezing it would leave `catalog_v2` as the only way to fix a
    sentence.
    """
    return draft.status in ("active", "disabled")


LOCKED_AFTER_RUN = ("key", "db_role", "tables")
EDITABLE_ALWAYS = ("display_name", "description", "prompt")


def run_agent(drafts: list[AgentDraft], key: str) -> list[AgentDraft]:
    """Provision an agent and make it routable. STUB - flips a string.

    Becomes real with: phase 4, the provisioner.

    What it will do, and why none of it belongs here: CREATE ROLE, GRANT
    SELECT on the agent's tables, then set status to active. Those need
    privileges no service handling requests should hold, so they run in a
    separate component - and asynchronously, which is why status exists at
    all rather than an agent simply being present or absent.

    The orchestrator must not offer a tool for an agent whose role does not
    exist yet, so nothing is routable until this completes.
    """
    return [
        AgentDraft(**{**vars(d), "status": "active"}) if d.key == key else d
        for d in drafts
    ]


def disable_agent(drafts: list[AgentDraft], key: str) -> list[AgentDraft]:
    """Stop routing to an agent, without removing it. STUB.

    This is what deleting a provisioned agent should mean. It takes effect
    immediately - the orchestrator builds its tool list from active agents
    only - while the role and the agent's history stay untouched. Dropping
    the role is a separate, deliberate act, because it cannot be undone and
    because DROP ROLE fails while the role still holds a privilege.
    """
    return [
        AgentDraft(**{**vars(d), "status": "disabled"}) if d.key == key else d
        for d in drafts
    ]


def enable_agent(drafts: list[AgentDraft], key: str) -> list[AgentDraft]:
    """Route to a disabled agent again. STUB.

    Safe and cheap: the role and its grants were never removed, so this only
    puts the agent back in the orchestrator's tool list.
    """
    return [
        AgentDraft(**{**vars(d), "status": "active"}) if d.key == key else d
        for d in drafts
    ]


def delete_agent(drafts: list[AgentDraft], key: str) -> list[AgentDraft]:
    """Forget an agent that was never provisioned. STUB - in memory only.

    Only offered before Run, where it really is this cheap: nothing exists
    outside the form yet. Once a role has been created, the equivalent is
    disable_agent, and actually dropping the role is a deliberate step of its
    own - REASSIGN OWNED, then DROP OWNED, then DROP ROLE, run by the
    provisioner, with the agent's history dealt with rather than cascading
    away underneath it.
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
# REAL - asking a question, when an orchestrator is running
#
# Posts to the orchestrator's /ask, which is the same endpoint any other
# client uses. There is no console-specific path into the system: if this
# works, the deployment works.
#
# It needs the orchestrator up (deploy/docker-compose.yml). When it is not,
# this says so plainly instead of returning a fabricated trace - a console
# that invents an answer teaches you the system works when it does not.
# ==========================================================================

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")


async def _ask(question: str, session_id: str, url: str) -> dict[str, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{url.rstrip('/')}/ask",
            json={"question": question, "session_id": session_id},
        )
        response.raise_for_status()
        return response.json()


async def _agents(url: str) -> list[dict[str, Any]]:
    import httpx

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{url.rstrip('/')}/agents")
        response.raise_for_status()
        return response.json()


def orchestrator_status(url: str = ORCHESTRATOR_URL) -> dict[str, Any]:
    """Whether an orchestrator is reachable, and who it routes to. REAL.

    Checked before offering the question box, so "nothing happens when I press
    ask" becomes "the orchestrator is not running" - which is a much shorter
    thing to fix.
    """
    async def check():
        import httpx

        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{url.rstrip('/')}/health")
            response.raise_for_status()
            return response.json()

    try:
        return {"reachable": True, **_run(check())}
    except Exception as e:
        return {"reachable": False, "error": str(e)}


def ask(question: str, session_id: str, url: str = ORCHESTRATOR_URL) -> dict[str, Any]:
    """Answer a question by delegating to a sub-agent. REAL.

    The orchestrator picks the agent, sends it a self-contained question, and
    combines what comes back. Everything factual in the answer came from an
    agent's SQL - the orchestrator has no database access of its own.
    """
    try:
        body = _run(_ask(question, session_id, url))
    except Exception as e:
        return {
            "ok": False,
            "answer": "",
            "error": (
                f"Could not reach the orchestrator at {url}: {e}\n\n"
                "Start it with:  docker compose -f deploy/docker-compose.yml up"
            ),
        }
    return {"ok": True, **body}


def registered_agents(url: str = ORCHESTRATOR_URL) -> list[dict[str, Any]]:
    """Who the running orchestrator can route to. REAL.

    Read from the orchestrator rather than from the registry file, so what
    the console shows is what the running system will actually use - the two
    differ whenever the file has been edited since the process started.
    """
    try:
        return _run(_agents(url))
    except Exception:
        return []
