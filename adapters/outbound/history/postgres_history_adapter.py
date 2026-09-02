# adapters/outbound/history/postgres_history_adapter.py
from typing import Any
import json
import re

from domain.ports.database_port import DatabasePort
from domain.ports.embedding_port import EmbeddingPort
from domain.entities.chat_message import ChatMessage, to_plain
from domain.exceptions import HistoryError
from libs.agent_core.pgvector import to_vector_literal

_VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def judge(messages: Any) -> tuple[bool, str]:
    """Was this turn worth learning from, and what shape was it?

    The `valid` and `reason` columns existed, `get_memory` filtered on them,
    and nothing ever wrote them - so every row had valid NULL, `valid = true`
    matched nothing, `valid = false` matched nothing, and get_memory could not
    return an example however many turns had been recorded. It cost an
    embedding call per turn to return an empty list. The docstring in
    RunAgentTurn describes this behaviour as though it were implemented; this
    is that implementation.

    A turn is valid when it produced a real answer and no tool failed on the
    way. Both halves matter. A turn that errored is not a worked example, and
    a turn that ended without an answer - the loop budget stopping it - is a
    worse one, because the model would be shown a pattern that leads nowhere.

    `reason` is the deduplication key, and making it the sequence of tools is
    what earns it. get_memory takes DISTINCT ON (reason), so a key that is the
    approach means three examples of three different approaches rather than
    three of the same one. On failure it is the error instead, which groups
    the failures by what went wrong.
    """
    from domain.entities.chat_message import ChatMessage, Role

    def role_of(message):
        role = message.role if isinstance(message, ChatMessage) else message.get("role")
        return role.value if isinstance(role, Role) else role

    def content_of(message):
        return (message.content if isinstance(message, ChatMessage)
                else message.get("content")) or ""

    def calls_of(message):
        if isinstance(message, ChatMessage):
            return [c.name for c in message.tool_calls or []]
        return [c.get("name") for c in (message.get("tool_calls") or [])]

    if not isinstance(messages, list):
        return False, "no trace"

    tools, failure, answered = [], "", False
    for message in messages:
        role = role_of(message)
        if role == "assistant":
            tools.extend(calls_of(message))
            if str(content_of(message)).strip():
                answered = True
        elif role == "tool":
            text = str(content_of(message))
            # The loop hands tool failures back to the model as a result
            # rather than raising, so this is where they are visible at all.
            if '"error"' in text[:400]:
                failure = failure or text[:200]

    if failure:
        return False, failure
    if not answered:
        return False, "no answer"
    return True, ">".join(tools) if tools else "answered directly"


def _as_json(payload: Any) -> str:
    """Encode a payload that may contain ChatMessages.

    log_assistant_final is handed the turn's messages, which are entities and
    not JSON. Every successful turn used to fail to record its own answer for
    exactly this reason - non-fatally now, which means silently.
    """
    if isinstance(payload, list) and all(isinstance(m, ChatMessage) for m in payload):
        return json.dumps([to_plain(m) for m in payload], ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False)


class PostgresHistoryAdapter:
    def __init__(self, db: DatabasePort, embeddings: EmbeddingPort,
                 table_name: str, embedding_dim: int):
        if not _VALID_TABLE_NAME.match(table_name):
            raise ValueError(f"Invalid table name {table_name!r}")

        self._db = db
        self._embeddings = embeddings
        self._table = table_name
        self._embedding_dim = embedding_dim

    async def ensure_schema(self) -> None:
        """Ensure the history schema is set up correctly.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            sql = f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    turn_id UUID NOT NULL,
                    event_type TEXT NOT NULL,
                    payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    user_message_embed vector({self._embedding_dim}),
                    valid BOOLEAN,
                    reason TEXT,
                    time DOUBLE PRECISION,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """
            await self._db.execute(sql)
        except Exception as e:
            raise HistoryError(f"Failed to create table {self._table}. Error {e}") from e

    async def log_user_message(self, session_id: str, turn_id: str, message: str) -> None:
        """Log a user message to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            embed = await self._embeddings.embed(message)
            sql = f"""
                INSERT INTO {self._table}
                (session_id, turn_id, event_type, payload, user_message_embed)
                VALUES ($1, $2, 'user', $3::jsonb, $4::vector)
            """
            await self._db.execute(sql, session_id, turn_id, json.dumps(message), to_vector_literal(embed))
        except Exception as e:
            raise HistoryError(f"Error {e} while logging user message to {self._table}") from e

    async def log_assistant_final(self, session_id: str, turn_id: str, final_answer: Any, elapsed: float) -> None:
        """Log the final assistant message to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            valid, reason = judge(final_answer)
            sql = f"""
                INSERT INTO {self._table}
                (session_id, turn_id, event_type, payload, time, valid, reason)
                VALUES ($1, $2, 'assistant_final', $3::jsonb, $4, $5, $6)
            """
            await self._db.execute(sql, session_id, turn_id, _as_json(final_answer),
                                   elapsed, valid, reason)
        except Exception as e:
            raise HistoryError(f"Error {e} while logging assistant final answer to {self._table}") from e

    # log_tool_call and log_sql_query were here, and nothing ever called
    # them. They are not merely unused: they duplicate what is already
    # recorded. log_assistant_final is handed the turn's whole message list -
    # every tool call with its arguments and every result, the SQL among them
    # - and stores it as one row, which is what get_memory reads back and
    # what scripts/show_history.py renders. A second, per-call record of the
    # same facts would be a second thing to keep in step, writing rows nothing
    # reads, on a table the service is deliberately not allowed to UPDATE.
    #
    # Removed rather than left in place, because a method on a port
    # implementation reads as something a caller might use.

    async def get_memory(self, query: str) -> list[dict]:
        """Retrieve semantically similar past turns from the history.

        Returns up to 3 valid examples (question + reasoning trace) and up to
        3 invalid examples (question + failure reason), deduplicated by the
        `reason` tag so repeated question patterns don't crowd out other
        examples.

        The verdict is read from the assistant_final row, not the user row.
        It used to filter on `u.valid`, which is the row written *before* the
        turn runs - so it could only ever have been NULL, and it was: `valid =
        true` matched nothing, `valid = false` matched nothing, and this
        method returned an empty list for every question while charging an
        embedding call to do it. The verdict belongs with the outcome, which
        is the row that knows how the turn went.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            vec = to_vector_literal(await self._embeddings.embed(query))

            good_sql = f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (f.reason)
                        u.payload, f.valid, f.reason, f.time, u.event_type, f.payload AS trace,
                        u.user_message_embed <=> $1::vector AS distance
                    FROM {self._table} u
                    JOIN {self._table} f ON f.turn_id = u.turn_id AND f.event_type = 'assistant_final'
                    WHERE u.event_type = 'user'
                      AND f.valid = true
                      AND u.created_at >= NOW() - INTERVAL '3 days'
                    ORDER BY f.reason, u.user_message_embed <=> $1::vector ASC
                ) deduped
                ORDER BY distance ASC
                LIMIT 3
            """
            good_rows = await self._db.fetch(good_sql, vec)

            bad_sql = f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (f.reason)
                        u.payload, f.valid, f.reason, f.time, u.event_type,
                        u.user_message_embed <=> $1::vector AS distance
                    FROM {self._table} u
                    JOIN {self._table} f ON f.turn_id = u.turn_id AND f.event_type = 'assistant_final'
                    WHERE u.event_type = 'user'
                      AND f.valid = false
                      AND u.created_at >= NOW() - INTERVAL '3 days'
                    ORDER BY f.reason, u.user_message_embed <=> $1::vector ASC
                ) deduped
                ORDER BY distance ASC
                LIMIT 3
            """
            bad_rows = await self._db.fetch(bad_sql, vec)

            examples = []
            if good_rows:
                examples.append({"valid_examples": good_rows})
            if bad_rows:
                examples.append({"invalid_examples": bad_rows})
            return examples

        except Exception as e:
            raise HistoryError(f"Error {e} while getting memory for query") from e