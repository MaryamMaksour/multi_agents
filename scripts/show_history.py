"""Print what actually happened inside a turn, from the history tables.

Every turn already records its own full trace. `run_agent_turn` hands
`log_assistant_final` the whole message list the loop produced - the
assistant's tool calls with their arguments, every tool result, and the final
answer - and that list is stored as JSONB. Nothing new has to be instrumented
to see how a model reached an answer; it is already written down.

What that buys, concretely: when the orchestrator answers "129" to a question
about Arabic novels, the trace says which of two things went wrong. Either the
orchestrator delegated a question that had already lost the word "novel":

    -> catalog {"query": "How many Arabic books are under 300 pages?"}

or it delegated correctly and the sub-agent ignored the genre column:

    -> db_execute {"sql": "SELECT count(*) FROM books WHERE language='ar' ..."}

Those are different bugs in different processes, and no amount of re-reading
the final number distinguishes them.

    python3 scripts/show_history.py                     last 5 turns, every agent
    python3 scripts/show_history.py --agent catalog     one agent's table
    python3 scripts/show_history.py --turns 20
    python3 scripts/show_history.py --session s1
    python3 scripts/show_history.py --turn <uuid>       one turn everywhere
    python3 scripts/show_history.py --full              no truncation
    python3 scripts/show_history.py --json              the rows, unrendered

Reads. Writes nothing, and holds no privilege to.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap

import asyncpg

# The dev compose file publishes Postgres on 55432, so running this on the
# host with nothing exported works. PG_* still wins where it is set, which is
# what makes the same script usable against a deployed database.
PG_HOST = os.getenv("PG_HOST") or "localhost"
PG_PORT = int(os.getenv("PG_PORT") or "55432")
PG_DBNAME = os.getenv("PG_DBNAME") or "library_dev"
# Not the authenticator: it holds SELECT on the history tables, but the
# password lives in the compose file rather than in a person's shell.
PG_USER = os.getenv("HISTORY_PG_USER") or "dev"
PG_PASSWORD = os.getenv("HISTORY_PG_PASSWORD") or "dev"

RESULT_WIDTH = 600      # per tool result, unless --full
ARGS_WIDTH = 400        # per tool call's arguments


def _c(text: str, code: str) -> str:
    """Colour, when the output is a terminal and not a pipe."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _clip(text: str, width: int, full: bool) -> str:
    if full or len(text) <= width:
        return text
    return text[:width] + _c(f" ... [{len(text) - width} more chars]", "2")


async def history_tables(conn: asyncpg.Connection, agent: str | None) -> list[str]:
    """Whichever history tables exist, rather than a list to keep in step.

    Registering an agent creates history_<name>; this then shows it without
    being edited, which is the same reason the registry is data and not code.
    """
    rows = await conn.fetch("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name LIKE 'history\\_%'
        ORDER BY table_name
    """)
    tables = [r["table_name"] for r in rows]
    if agent:
        wanted = f"history_{agent}"
        tables = [t for t in tables if t == wanted]
        if not tables:
            raise SystemExit(f"No history table for agent {agent!r}.")
    return tables


async def read_turns(conn: asyncpg.Connection, table: str, limit: int,
                     session: str | None, turn: str | None) -> list[dict]:
    """One row per turn: the question, the trace, and how long it took.

    A turn is two rows - 'user' written before the loop runs and
    'assistant_final' written after - joined on turn_id. LEFT JOIN, not INNER:
    a turn that raised has a question and no trace, and that is the turn most
    worth looking at.
    """
    where, params = [], []

    if session:
        params.append(session)
        where.append(f"u.session_id = ${len(params)}")
    if turn:
        params.append(turn)
        where.append(f"u.turn_id = ${len(params)}::uuid")

    clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)

    rows = await conn.fetch(f"""
        SELECT u.session_id,
               u.turn_id,
               u.created_at,
               u.payload      AS question,
               a.payload      AS trace,
               a.time         AS elapsed,
               (u.user_message_embed IS NOT NULL) AS embedded
        FROM {table} u
        LEFT JOIN {table} a
               ON a.turn_id = u.turn_id AND a.event_type = 'assistant_final'
        WHERE u.event_type = 'user'
        {clause.replace('WHERE', 'AND', 1) if clause else ''}
        ORDER BY u.created_at DESC
        LIMIT ${len(params)}
    """, *params)

    # Read newest-first so the LIMIT keeps the recent turns, print oldest-first
    # so a session reads in the order it happened.
    return [dict(r) for r in reversed(rows)]


def render_turn(table: str, row: dict, full: bool) -> None:
    agent = table.removeprefix("history_")
    when = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    elapsed = f"{row['elapsed']:.1f}s" if row["elapsed"] is not None else "-"

    print()
    print(_c("=" * 78, "36"))
    print(_c(f"{agent:<14}", "1;36") + f"{when}   {elapsed}   session={row['session_id']}")
    print(_c(f"turn {row['turn_id']}", "2"))
    print(_c("=" * 78, "36"))

    question = row["question"]
    if isinstance(question, str):
        try:
            question = json.loads(question)
        except json.JSONDecodeError:
            pass
    print(_c("QUESTION  ", "1;33") + str(question))

    if not row["embedded"]:
        # No embedding means get_memory can never retrieve this turn: it is
        # recorded for audit but invisible as a worked example.
        print(_c("          (not embedded - unreachable by get_memory)", "2"))

    if row["trace"] is None:
        print(_c("\nNO TRACE - the turn did not finish. Check the service logs "
                 "for this turn_id.", "1;31"))
        return

    trace = json.loads(row["trace"]) if isinstance(row["trace"], str) else row["trace"]
    render_trace(trace, full)


def render_trace(messages: list[dict], full: bool) -> None:
    """Walk the message list in order, which is the order it happened in."""
    # Tool results are matched back to the call that produced them, so a
    # result prints under its own call even when the model issued several in
    # one step and they returned out of order.
    call_names = {
        call["id"]: call["name"]
        for message in messages
        for call in (message.get("tool_calls") or [])
    }

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role == "user":
            continue  # already printed as QUESTION

        if role == "assistant":
            # Text alongside tool calls is the model narrating its plan; text
            # on its own, at the end, is the answer.
            if content and str(content).strip():
                label = "SAYS      " if message.get("tool_calls") else "ANSWER    "
                colour = "0" if message.get("tool_calls") else "1;32"
                body = textwrap.indent(str(content).strip(), " " * 10).lstrip()
                print(_c(label, "1;32" if label.startswith("ANSWER") else "1") + _c(body, colour))

            for call in message.get("tool_calls") or []:
                args = json.dumps(call.get("args", {}), ensure_ascii=False)
                print(_c("  -> ", "1;35") + _c(call["name"], "1;35") + " "
                      + _clip(args, ARGS_WIDTH, full))

        elif role == "tool":
            name = call_names.get(message.get("tool_call_id"), message.get("name") or "?")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            # An error came back as a result rather than as an exception, so
            # it is easy to miss in a wall of rows. Mark it.
            failed = '"error"' in text[:200]
            print(_c("  <- ", "1;31" if failed else "2") + _c(name, "2") + " "
                  + _clip(text, RESULT_WIDTH, full))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", help="one agent (catalog, circulation, orchestrator)")
    parser.add_argument("--turns", type=int, default=5, help="how many recent turns per agent")
    parser.add_argument("--session", help="only this session_id")
    parser.add_argument("--turn", help="only this turn_id, across every agent")
    parser.add_argument("--full", action="store_true", help="do not truncate")
    parser.add_argument("--json", action="store_true", help="print the rows, unrendered")
    args = parser.parse_args()

    try:
        conn = await asyncpg.connect(host=PG_HOST, port=PG_PORT, database=PG_DBNAME,
                                     user=PG_USER, password=PG_PASSWORD)
    except Exception as e:
        print(f"Cannot reach Postgres at {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}: {e}",
              file=sys.stderr)
        print("Set PG_HOST/PG_PORT/PG_DBNAME, or HISTORY_PG_USER/HISTORY_PG_PASSWORD.",
              file=sys.stderr)
        return 1

    try:
        tables = await history_tables(conn, args.agent)
        # A turn_id is unique across the system, so --turn wants every process
        # that touched it - the orchestrator's turn and the sub-agent's are
        # the same turn seen from two sides.
        turns_each = args.turns if not args.turn else 50

        found = 0
        for table in tables:
            rows = await read_turns(conn, table, turns_each, args.session, args.turn)
            found += len(rows)
            for row in rows:
                if args.json:
                    print(json.dumps(row, default=str, ensure_ascii=False))
                else:
                    render_turn(table, row, args.full)

        if not found:
            print("No turns recorded yet for that filter.")
            print(f"Tables read: {', '.join(tables) or 'none'}")
        elif not args.json:
            print()
    finally:
        await conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
