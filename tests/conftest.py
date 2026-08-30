"""Test doubles for the ports, and shared fixtures.

Everything the domain touches is a port, so a fake is a small class with the
right methods - no mocking library, no patching, no database. That is the
whole return on the ports: RunAgentTurn can be exercised completely in
memory, in milliseconds.

The fakes record what they were asked to do rather than only what they
returned. Most of the bugs found while reviewing this interactor were about
*sequence* - a lock released twice, history written before the lock was held,
the persisted messages differing from the ones sent to the model. A fake that
only returns values cannot catch any of those, so these keep a call log.
"""

from __future__ import annotations

import pytest

from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.entities.chat_message import ChatMessage, Role, ToolCall


class FakeCache:
    """CachePort with an in-memory store and a real single-holder lock.

    acquire_lock genuinely refuses a second acquisition, so the busy-session
    path can be tested without threads or Redis.
    """

    def __init__(self, *, lock_available: bool = True):
        self.store: dict = {}
        self.lock_available = lock_available
        self.locked: set[str] = set()
        self.calls: list[tuple] = []

    async def get(self, key: str):
        self.calls.append(("get", key))
        # Mirrors RedisCacheAdapter: a missing session is an empty window, not
        # None, so the interactor can concatenate without a None check.
        return self.store.get(key, [])

    async def set(self, key: str, value, ttl: int) -> None:
        self.calls.append(("set", key, ttl))
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.calls.append(("delete", key))
        self.store.pop(key, None)

    async def acquire_lock(self, key: str, timeout: int) -> bool:
        self.calls.append(("acquire_lock", key, timeout))
        if not self.lock_available or key in self.locked:
            return False
        self.locked.add(key)
        return True

    async def release_lock(self, key: str) -> None:
        self.calls.append(("release_lock", key))
        self.locked.discard(key)

    # -- helpers for assertions --
    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeHistory:
    """HistoryPort that records every write and returns canned memory."""

    def __init__(self, memory: list[dict] | None = None):
        self.memory = memory if memory is not None else []
        self.user_messages: list[dict] = []
        self.finals: list[dict] = []
        self.tool_calls: list[dict] = []
        self.sql_queries: list[dict] = []
        self.schema_ensured = False
        self.calls: list[str] = []

    async def log_user_message(self, session_id: str, turn_id: str, message: str) -> None:
        self.calls.append("log_user_message")
        self.user_messages.append(
            {"session_id": session_id, "turn_id": turn_id, "message": message}
        )

    async def log_assistant_final(self, session_id: str, turn_id: str, final_answer, elapsed: float) -> None:
        self.calls.append("log_assistant_final")
        self.finals.append(
            {"session_id": session_id, "turn_id": turn_id,
             "final_answer": final_answer, "elapsed": elapsed}
        )

    async def log_tool_call(self, session_id: str, turn_id: str, tool_name: str,
                            input_data, output_data) -> None:
        self.calls.append("log_tool_call")
        self.tool_calls.append({"tool_name": tool_name, "input": input_data, "output": output_data})

    async def log_sql_query(self, session_id: str, turn_id: str, query: str, params) -> None:
        self.calls.append("log_sql_query")
        self.sql_queries.append({"query": query, "params": params})

    async def get_memory(self, query: str) -> list[dict]:
        self.calls.append("get_memory")
        return self.memory

    async def ensure_schema(self) -> None:
        self.calls.append("ensure_schema")
        self.schema_ensured = True


class FakeAgentLoop:
    """AgentLoopPort returning a fixed result, or raising a fixed error.

    Captures the messages it was handed, which is how the tests check what the
    model would actually have seen.
    """

    def __init__(self, result: AgentTurnResult | None = None, error: Exception | None = None):
        self.result = result if result is not None else AgentTurnResult(
            messages=[ChatMessage(role=Role.ASSISTANT, content="answer")]
        )
        self.error = error
        self.received: list[list[ChatMessage]] = []

    async def run(self, messages: list[ChatMessage]) -> AgentTurnResult:
        self.received.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.result

    @property
    def last_messages(self) -> list[ChatMessage]:
        assert self.received, "the loop was never called"
        return self.received[-1]


class FakeLLM:
    """LLMPort returning queued replies in order."""

    def __init__(self, replies: list[ChatMessage] | None = None, error: Exception | None = None):
        self.replies = list(replies or [])
        self.error = error
        self.received: list[list[ChatMessage]] = []

    async def achat(self, messages: list[ChatMessage]) -> ChatMessage:
        self.received.append(list(messages))
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        return ChatMessage(role=Role.ASSISTANT, content="done")


class FakeTools:
    """ToolPort returning a canned result per tool name."""

    def __init__(self, results: dict | None = None, error: Exception | None = None):
        self.results = results or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name: str, args: dict):
        self.calls.append((tool_name, args))
        if self.error is not None:
            raise self.error
        return self.results.get(tool_name, {"ok": True})


class FakeEmbeddings:
    """EmbeddingPort returning a fixed-width vector."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.received: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.received.append(text)
        return [0.1] * self.dim


class FakeDatabase:
    """DatabasePort returning queued row batches."""

    def __init__(self, rows: list[list[dict]] | None = None):
        self.rows = list(rows or [])
        self.queries: list[tuple] = []

    async def fetch(self, query: str, *params):
        self.queries.append((query, params))
        return self.rows.pop(0) if self.rows else []

    async def execute(self, query: str, *params):
        self.queries.append((query, params))
        return None


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def cache():
    return FakeCache()


@pytest.fixture
def history():
    return FakeHistory()


@pytest.fixture
def agent_loop():
    return FakeAgentLoop()


def msg(role: Role, content, **kw) -> ChatMessage:
    """Shorthand for building a ChatMessage in a test."""
    return ChatMessage(role=role, content=content, **kw)


def window(n: int, prefix: str = "m") -> list[ChatMessage]:
    """A conversation window of n alternating messages, content m0, m1, ...

    Used to check the trimming rules, where what matters is which messages
    survive and in what order - so the content is just an index.
    """
    out = []
    for i in range(n):
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        out.append(ChatMessage(role=role, content=f"{prefix}{i}"))
    return out
