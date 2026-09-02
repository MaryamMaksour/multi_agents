"""The cache adapter, against a real Redis.

This file exists because of a bug that every unit test missed and that broke
the very first question of every new session.

RunAgentTurn locks a session and then reads that session's window, passing
the same key to both. That is the right shape for a port - "the lock for this
session" and "the window for this session" are two facts about one thing. But
redis-py's Lock writes its token at exactly the key it is given, so the lock
overwrote the window, and get() then tried to json.loads a uuid:

    CacheError: Extra data: line 1 column 9 (char 8)

The fake in conftest.py could not catch it, because a fake keeps its store
and its locks in separate Python attributes - it has no shared namespace to
collide in. The fake was, accidentally, a more correct implementation than
the real one. That is the general shape of what integration tests are for:
the fake encodes an assumption, and only the real thing can check it.

Requires Redis. Skipped when it is not reachable.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

redis = pytest.importorskip("redis.asyncio")

from adapters.outbound.cache.redis_cache_adapter import (  # noqa: E402
    LOCK_PREFIX,
    VALUE_PREFIX,
    RedisCacheAdapter,
)
from domain.entities.chat_message import ChatMessage, Role, ToolCall  # noqa: E402

pytestmark = pytest.mark.integration

# Database 9, not the one the services use, so a test run cannot flush a
# working session out from under a running container.
URL = os.getenv("TEST_REDIS_URL", "redis://localhost:56379/9")


@pytest.fixture
async def cache():
    try:
        client = redis.from_url(URL)
        await client.ping()
    except Exception as e:
        pytest.skip(f"Redis unavailable at {URL}: {e}")

    await client.flushdb()
    yield RedisCacheAdapter(client), client
    await client.flushdb()
    await client.aclose()


def window(*contents):
    return [ChatMessage(role=Role.USER, content=c) for c in contents]


# --------------------------------------------------------------------------
# the collision
# --------------------------------------------------------------------------


async def test_locking_a_session_does_not_destroy_its_window(cache):
    """The regression. RunAgentTurn does exactly this, in this order, on
    every single turn."""
    adapter, _ = cache
    await adapter.set("s1", window("earlier question"), ttl=60)

    assert await adapter.acquire_lock("s1", timeout=30) is True
    restored = await adapter.get("s1")

    assert [m.content for m in restored] == ["earlier question"]


async def test_a_new_session_reads_an_empty_window_while_locked(cache):
    """The exact failing path: a session with no history, locked first and
    read second. This is what raised on the first question every time."""
    adapter, _ = cache
    await adapter.acquire_lock("brand-new", timeout=30)

    assert await adapter.get("brand-new") == []


async def test_values_and_locks_use_separate_key_namespaces(cache):
    """Asserted on the keys themselves, not only on the behaviour above. A
    future change that reintroduces the collision fails here with a message
    about namespaces rather than about JSON."""
    adapter, client = cache
    await adapter.set("s1", window("hello"), ttl=60)
    await adapter.acquire_lock("s1", timeout=30)

    keys = {k.decode() for k in await client.keys("*")}
    assert f"{VALUE_PREFIX}s1" in keys
    assert f"{LOCK_PREFIX}s1" in keys
    assert "s1" not in keys


# --------------------------------------------------------------------------
# a missing session is an empty window, not None
# --------------------------------------------------------------------------


async def test_an_unknown_session_is_an_empty_window(cache):
    """The second bug, which would have fired the moment the first was
    fixed: RunAgentTurn slices what this returns, and None is not
    subscriptable. A session that has said nothing and a session that does
    not exist are the same thing to the caller."""
    adapter, _ = cache
    assert await adapter.get("never-seen") == []


async def test_what_comes_back_can_be_sliced_and_concatenated(cache):
    """Exactly what the interactor does with it."""
    adapter, _ = cache
    existing = await adapter.get("never-seen")

    assert existing[-20:] == []
    assert [m.content for m in existing + window("new")] == ["new"]


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


async def test_a_window_survives_a_round_trip(cache):
    adapter, _ = cache
    await adapter.set("s1", window("one", "two"), ttl=60)

    assert [m.content for m in await adapter.get("s1")] == ["one", "two"]


async def test_roles_survive_a_round_trip(cache):
    adapter, _ = cache
    await adapter.set("s1", [
        ChatMessage(role=Role.USER, content="q"),
        ChatMessage(role=Role.ASSISTANT, content="a"),
    ], ttl=60)

    assert [m.role for m in await adapter.get("s1")] == [Role.USER, Role.ASSISTANT]


async def test_tool_calls_survive_a_round_trip(cache):
    """The part with real structure. A window that loses its tool calls
    reads to the model as an assistant that asked for nothing and then got
    an answer from nowhere."""
    adapter, _ = cache
    await adapter.set("s1", [
        ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(id="c1", name="db_execute", args={"query": "SELECT 1"}),
        ]),
        ChatMessage(role=Role.TOOL, content='{"rows": []}', tool_call_id="c1"),
    ], ttl=60)

    restored = await adapter.get("s1")
    assert restored[0].tool_calls[0].name == "db_execute"
    assert restored[0].tool_calls[0].args == {"query": "SELECT 1"}
    assert restored[1].tool_call_id == "c1"


async def test_deleting_removes_the_window(cache):
    adapter, _ = cache
    await adapter.set("s1", window("one"), ttl=60)
    await adapter.delete("s1")

    assert await adapter.get("s1") == []


# --------------------------------------------------------------------------
# the lock actually locks
# --------------------------------------------------------------------------


async def test_a_second_acquisition_is_refused(cache):
    """What makes two overlapping requests for one session serialise rather
    than interleave and corrupt each other's history."""
    adapter, _ = cache
    assert await adapter.acquire_lock("s1", timeout=30) is True
    assert await adapter.acquire_lock("s1", timeout=30) is False


async def test_a_released_lock_can_be_taken_again(cache):
    adapter, _ = cache
    await adapter.acquire_lock("s1", timeout=30)
    await adapter.release_lock("s1")

    assert await adapter.acquire_lock("s1", timeout=30) is True


async def test_different_sessions_do_not_block_each_other(cache):
    adapter, _ = cache
    assert await adapter.acquire_lock("sa", timeout=30) is True
    assert await adapter.acquire_lock("sb", timeout=30) is True


async def test_releasing_a_lock_never_taken_is_harmless(cache):
    """RunAgentTurn releases in a finally, which runs even on the path where
    the lock was never acquired."""
    adapter, _ = cache
    await adapter.release_lock("never-locked")


@pytest.mark.asyncio
async def test_releasing_an_expired_lock_does_not_raise(cache):
    adapter, _ = cache
    """The failure this prevents cost a correct answer.

    RunAgentTurn releases the session lock in a `finally`, so anything raised
    here replaces the value the turn was about to return. And the likeliest
    cause is a turn that outlived its own lock timeout: the lock expired by
    itself, another turn may hold it now, and there is nothing to release.
    Losing the answer over that is the worst available outcome.
    """
    key = f"expiring-{uuid.uuid4()}"

    # A one-second lease, then wait past it: the lock is gone from Redis while
    # this adapter still holds the object it acquired.
    assert await adapter.acquire_lock(key=key, timeout=1)
    await asyncio.sleep(1.5)

    await adapter.release_lock(key=key)  # must not raise


@pytest.mark.asyncio
async def test_releasing_a_lock_that_was_never_acquired_is_a_no_op(cache):
    adapter, _ = cache
    await adapter.release_lock(key=f"never-held-{uuid.uuid4()}")
