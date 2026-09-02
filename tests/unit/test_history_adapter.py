"""PostgresHistoryAdapter - what gets written, so get_memory has something to read.

get_memory selects on `valid`, which used to be written by nothing: the
column stayed NULL and both halves of the lookup matched no rows. It is now
decided from the turn's own trace at the moment the final answer is logged.
"""

from __future__ import annotations

import json

from adapters.outbound.history.postgres_history_adapter import PostgresHistoryAdapter, _trace_is_clean
from domain.entities.chat_message import ChatMessage, Role, ToolCall

from tests.conftest import FakeDatabase, FakeEmbeddings


def adapter(db=None) -> PostgresHistoryAdapter:
    return PostgresHistoryAdapter(
        db=db or FakeDatabase(), embeddings=FakeEmbeddings(),
        table_name="history_catalog", embedding_dim=3,
    )


def turn(*tool_results: dict) -> list[ChatMessage]:
    messages = [ChatMessage(role=Role.USER, content="how many?")]
    for i, result in enumerate(tool_results):
        messages.append(ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(id=f"c{i}", name="db_execute", args={}),
        ]))
        messages.append(ChatMessage(role=Role.TOOL, content=json.dumps(result), tool_call_id=f"c{i}"))
    messages.append(ChatMessage(role=Role.ASSISTANT, content="Twelve."))
    return messages


def test_a_trace_with_no_failed_tool_is_valid():
    assert _trace_is_clean(turn({"rows": [{"n": 12}], "has_more": False}))


def test_a_trace_with_a_refused_query_is_not():
    assert not _trace_is_clean(turn(
        {"error": "Table not allowed: members"},
        {"rows": [{"n": 12}], "has_more": False},
    ))


def test_non_json_tool_output_does_not_count_against_the_turn():
    messages = turn()
    messages.insert(1, ChatMessage(role=Role.TOOL, content="plain text", tool_call_id="c9"))
    assert _trace_is_clean(messages)


def test_a_payload_that_is_not_a_trace_is_taken_as_valid():
    assert _trace_is_clean("just a string")


async def test_log_assistant_final_writes_valid_on_the_row():
    db = FakeDatabase()
    await adapter(db).log_assistant_final("s1", "t1", turn({"error": "nope"}), elapsed=0.5)

    (sql, params), = db.queries
    assert "valid" in sql
    assert params[0:2] == ("s1", "t1")
    assert params[-1] is False


async def test_get_memory_reads_valid_from_the_final_row():
    """The `user` row is written before the answer exists and the service
    holds no UPDATE, so `valid` can only live on the assistant_final row."""
    db = FakeDatabase(rows=[[], []])
    await adapter(db).get_memory("anything")

    for sql, _ in db.queries:
        assert "f.valid" in sql
        assert "u.valid" not in sql
        assert "COALESCE(u.reason, u.turn_id::text)" in sql
