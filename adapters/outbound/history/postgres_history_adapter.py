# adapters/outbound/history/postgres_history_adapter.py
from typing import Any
import json
import re

from domain.ports.database_port import DatabasePort
from domain.ports.embedding_port import EmbeddingPort
from domain.entities.chat_message import ChatMessage, Role, to_plain
from domain.exceptions import HistoryError
from libs.agent_core.pgvector import to_vector_literal

_VALID_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _as_json(payload: Any) -> str:
    """Encode a payload that may contain ChatMessages.

    log_assistant_final is handed the turn's messages, which are entities and
    not JSON. Every successful turn used to fail to record its own answer for
    exactly this reason - non-fatally now, which means silently.
    """
    if isinstance(payload, list) and all(isinstance(m, ChatMessage) for m in payload):
        return json.dumps([to_plain(m) for m in payload], ensure_ascii=False)
    return json.dumps(payload, ensure_ascii=False)


def _trace_is_clean(final_answer: Any) -> bool:
    """Whether a turn's trace holds no failed tool call.

    Tool results are JSON text on TOOL messages; a failure is a dict with an
    `error` key, which is the one shape every adapter uses for one. A turn
    with any such result is not an example worth imitating, whatever the
    model said at the end.
    """
    if not isinstance(final_answer, list):
        return True
    for message in final_answer:
        if not isinstance(message, ChatMessage) or message.role is not Role.TOOL:
            continue
        try:
            result = json.loads(message.content or "")
        except (TypeError, ValueError):
            continue
        if isinstance(result, dict) and "error" in result:
            return False
    return True


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

        `valid` is written here, on the row that has the trace to judge: true
        when no tool call in the turn failed. get_memory reads it from this
        row. The `user` row cannot carry it - it is written before the answer
        exists, and the service holds no UPDATE on history.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            sql = f"""
                INSERT INTO {self._table}
                (session_id, turn_id, event_type, payload, time, valid)
                VALUES ($1, $2, 'assistant_final', $3::jsonb, $4, $5)
            """
            await self._db.execute(
                sql, session_id, turn_id, _as_json(final_answer), elapsed,
                _trace_is_clean(final_answer),
            )
        except Exception as e:
            raise HistoryError(f"Error {e} while logging assistant final answer to {self._table}") from e

    async def log_tool_call(self, session_id: str, turn_id: str, tool_name: str, input_data: Any, output_data: Any) -> None:
        """Log a tool call to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            payload_data = {
                "tool_name": tool_name,
                "input_data": input_data,
                "output_data": output_data,
            }
            sql = f"""
                INSERT INTO {self._table}
                (session_id, turn_id, event_type, payload)
                VALUES ($1, $2, 'tool', $3::jsonb)
            """
            await self._db.execute(sql, session_id, turn_id, json.dumps(payload_data))
        except Exception as e:
            raise HistoryError(f"Error {e} while logging tool call to {self._table}") from e

    async def log_sql_query(self, session_id: str, turn_id: str, query: str, params: Any) -> None:
        """Log an SQL query to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            payload_data = {"query": query, "params": params}
            sql = f"""
                INSERT INTO {self._table}
                (session_id, turn_id, event_type, payload)
                VALUES ($1, $2, 'sql', $3::jsonb)
            """
            await self._db.execute(sql, session_id, turn_id, json.dumps(payload_data))
        except Exception as e:
            raise HistoryError(f"Error {e} while logging sql query to {self._table}") from e

    async def get_memory(self, query: str) -> list[dict]:
        """Retrieve semantically similar past turns from the history.

        Returns up to 3 valid examples (question + reasoning trace) and up to
        3 invalid examples (question + failure reason). Validity is the
        `assistant_final` row's `valid`, written by log_assistant_final.
        Examples are deduplicated by `reason` when one has been assigned, and
        otherwise by turn - a NULL reason must not collapse every untagged
        turn into a single example.

        Raises:
            HistoryError: if the history call fails.
        """
        try:
            vec = to_vector_literal(await self._embeddings.embed(query))

            good_sql = f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (COALESCE(u.reason, u.turn_id::text))
                        u.payload, f.valid, u.reason, f.time, u.event_type, f.payload AS trace,
                        u.user_message_embed <=> $1::vector AS distance
                    FROM {self._table} u
                    JOIN {self._table} f ON f.turn_id = u.turn_id AND f.event_type = 'assistant_final'
                    WHERE u.event_type = 'user'
                      AND f.valid = true
                      AND u.created_at >= NOW() - INTERVAL '3 days'
                    ORDER BY COALESCE(u.reason, u.turn_id::text), u.user_message_embed <=> $1::vector ASC
                ) deduped
                ORDER BY distance ASC
                LIMIT 3
            """
            good_rows = await self._db.fetch(good_sql, vec)

            bad_sql = f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (COALESCE(u.reason, u.turn_id::text))
                        u.payload, f.valid, u.reason, f.time, u.event_type,
                        u.user_message_embed <=> $1::vector AS distance
                    FROM {self._table} u
                    JOIN {self._table} f ON f.turn_id = u.turn_id AND f.event_type = 'assistant_final'
                    WHERE u.event_type = 'user'
                      AND f.valid = false
                      AND u.created_at >= NOW() - INTERVAL '3 days'
                    ORDER BY COALESCE(u.reason, u.turn_id::text), u.user_message_embed <=> $1::vector ASC
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