"""RunAgentTurn - the core use case, exercised entirely with fakes.

No database, no Redis, no model. Every one of these runs in microseconds,
which is the point: this is the file that changes most often and it needs a
suite fast enough to run on every save.

Several of these cover bugs that were actually present during development -
a shared system_prompt being mutated, the persisted messages being the same
list as the ones sent to the model, a lock not released on failure. Those are
marked, because a test whose failure mode has been seen once is worth more
than one written speculatively.
"""

from __future__ import annotations

import pytest

from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.entities.chat_message import ChatMessage, Role
from domain.exceptions import SessionBusyError
from domain.interactors.run_agent_turn import RunAgentTurn

from tests.conftest import FakeAgentLoop, FakeCache, FakeHistory, window


def build(cache=None, history=None, agent_loop=None, **kw) -> RunAgentTurn:
    return RunAgentTurn(
        agent_loop=agent_loop or FakeAgentLoop(),
        history=history or FakeHistory(),
        cache=cache or FakeCache(),
        system_prompt=kw.pop("system_prompt", "BASE PROMPT"),
        **kw,
    )


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------


async def test_returns_loop_messages_and_pagination():
    pagination = {"catalog": PaginationState(has_more=True, next_cursor="c1", pages_fetched=1)}
    loop = FakeAgentLoop(AgentTurnResult(
        messages=[ChatMessage(role=Role.ASSISTANT, content="the answer")],
        pagination=pagination,
    ))
    result = await build(agent_loop=loop).run("s1", "t1", "hello")

    assert result.pagination == pagination
    assert result.messages[-1].content == "the answer"


async def test_model_sees_system_prompt_then_window_then_user():
    cache = FakeCache()
    cache.store["s1"] = window(2)
    loop = FakeAgentLoop()

    await build(cache=cache, agent_loop=loop).run("s1", "t1", "new question")

    sent = loop.last_messages
    assert sent[0].role is Role.SYSTEM
    assert [m.content for m in sent[1:3]] == ["m0", "m1"]
    assert sent[-1].role is Role.USER
    assert sent[-1].content == "new question"


async def test_memory_examples_reach_the_model():
    history = FakeHistory(memory=[{"q": "an earlier question", "a": "its answer"}])
    loop = FakeAgentLoop()

    await build(history=history, agent_loop=loop).run("s1", "t1", "hi")

    assert any("an earlier question" in (m.content or "") for m in loop.last_messages)


async def test_the_system_prompt_is_sent_unchanged():
    """It is the first thing in every request and it must be byte-identical
    every time. Providers cache by matching the prefix of a request, so
    anything volatile mixed in here moves the first difference to byte zero
    and nothing after it can be cached - including the tool schemas, which
    are the largest fixed cost in the loop."""
    history = FakeHistory(memory=[{"q": "an earlier question"}])
    loop = FakeAgentLoop()

    await build(history=history, agent_loop=loop).run("s1", "t1", "hi")

    assert loop.last_messages[0].content == "BASE PROMPT"


async def test_the_volatile_part_comes_after_the_stable_part():
    """Examples last, next to the question they are examples for. Putting
    them before the conversation window would make the window uncacheable
    too, since a prefix is only reusable up to its first difference."""
    history = FakeHistory(memory=[{"q": "an earlier question"}])
    loop = FakeAgentLoop()

    await build(history=history, agent_loop=loop).run("s1", "t1", "hi")

    contents = [m.content or "" for m in loop.last_messages]
    examples_at = next(i for i, c in enumerate(contents) if "an earlier question" in c)
    assert examples_at > 0
    assert contents[-1] == "hi"


async def test_no_examples_means_no_extra_message():
    """An empty memory must not add an empty block. It would still be a
    difference in the prefix, which is the thing being avoided."""
    loop = FakeAgentLoop()
    await build(history=FakeHistory(memory=[]), agent_loop=loop).run("s1", "t1", "hi")

    assert [m.content for m in loop.last_messages] == ["BASE PROMPT", "hi"]


# --------------------------------------------------------------------------
# the locking rules
# --------------------------------------------------------------------------


async def test_busy_session_raises_and_touches_nothing():
    cache = FakeCache(lock_available=False)
    history = FakeHistory()
    loop = FakeAgentLoop()

    with pytest.raises(SessionBusyError):
        await build(cache=cache, history=history, agent_loop=loop).run("s1", "t1", "hi")

    # Nothing may be written for a turn that never ran.
    assert history.calls == []
    assert loop.received == []


async def test_a_refused_lock_is_not_released():
    """Releasing a lock this turn never held would free another turn's lock."""
    cache = FakeCache(lock_available=False)

    with pytest.raises(SessionBusyError):
        await build(cache=cache).run("s1", "t1", "hi")

    assert "release_lock" not in cache.names()


async def test_lock_is_released_when_the_loop_fails():
    """Regression: an exception used to leave the session locked until TTL."""
    cache = FakeCache()
    loop = FakeAgentLoop(error=RuntimeError("model exploded"))

    with pytest.raises(RuntimeError):
        await build(cache=cache, agent_loop=loop).run("s1", "t1", "hi")

    assert cache.names().count("release_lock") == 1
    assert cache.locked == set()


async def test_lock_is_released_exactly_once_on_success():
    """Regression: it was once released in the body *and* in finally."""
    cache = FakeCache()
    await build(cache=cache).run("s1", "t1", "hi")

    assert cache.names().count("release_lock") == 1


async def test_lock_is_taken_before_any_history_write():
    cache = FakeCache()
    history = FakeHistory()
    await build(cache=cache, history=history).run("s1", "t1", "hi")

    assert cache.names()[0] == "acquire_lock"
    assert history.calls[0] == "get_memory"


# --------------------------------------------------------------------------
# what is persisted vs. what the model sees
# --------------------------------------------------------------------------


async def test_persisted_window_excludes_the_system_prompt():
    """The prompt is rebuilt every turn; storing it would compound it."""
    cache = FakeCache()
    await build(cache=cache).run("s1", "t1", "hi")

    stored = cache.store["s1"]
    assert all(m.role is not Role.SYSTEM for m in stored)


async def test_persisted_window_keeps_user_and_loop_messages_in_order():
    cache = FakeCache()
    loop = FakeAgentLoop(AgentTurnResult(
        messages=[ChatMessage(role=Role.ASSISTANT, content="reply")]
    ))
    await build(cache=cache, agent_loop=loop).run("s1", "t1", "question")

    assert [(m.role, m.content) for m in cache.store["s1"]] == [
        (Role.USER, "question"),
        (Role.ASSISTANT, "reply"),
    ]


async def test_returned_messages_include_the_system_prompt_but_stored_do_not():
    """Regression: these were once the same list, so persisting the window
    also persisted the system prompt and every later turn grew."""
    cache = FakeCache()
    result = await build(cache=cache).run("s1", "t1", "hi")

    assert any(m.role is Role.SYSTEM for m in result.messages)
    assert not any(m.role is Role.SYSTEM for m in cache.store["s1"])


async def test_the_final_log_records_only_this_turn():
    cache = FakeCache()
    cache.store["s1"] = window(4)
    history = FakeHistory()

    await build(cache=cache, history=history).run("s1", "t1", "question")

    logged = history.finals[0]["final_answer"]
    assert [m.content for m in logged] == ["question", "answer"]
    assert history.finals[0]["elapsed"] >= 0


# --------------------------------------------------------------------------
# the two message limits
# --------------------------------------------------------------------------


async def test_context_messages_sent_trims_what_the_model_receives():
    cache = FakeCache()
    cache.store["s1"] = window(10)
    loop = FakeAgentLoop()

    await build(cache=cache, agent_loop=loop, context_messages_sent=4).run("s1", "t1", "q")

    sent = loop.last_messages
    # system + 4 most recent + this turn's user message
    assert len(sent) == 6
    assert [m.content for m in sent[1:5]] == ["m6", "m7", "m8", "m9"]


async def test_max_session_messages_trims_what_is_retained():
    cache = FakeCache()
    cache.store["s1"] = window(10)

    await build(cache=cache, max_session_messages=5).run("s1", "t1", "q")

    stored = cache.store["s1"]
    assert len(stored) == 5
    # the newest survive: the tail of the old window, then this turn
    assert [m.content for m in stored[-2:]] == ["q", "answer"]


async def test_the_two_limits_are_independent():
    """Retaining more than is sent is the intended configuration, not a bug."""
    cache = FakeCache()
    cache.store["s1"] = window(20)
    loop = FakeAgentLoop()

    await build(cache=cache, agent_loop=loop,
                context_messages_sent=4, max_session_messages=12).run("s1", "t1", "q")

    assert len(loop.last_messages) == 6      # system + 4 + user
    assert len(cache.store["s1"]) == 12


# --------------------------------------------------------------------------
# stateless mode, for sub-agents
# --------------------------------------------------------------------------


async def test_stateless_mode_neither_reads_nor_writes_the_window():
    cache = FakeCache()
    cache.store["s1"] = window(4)

    await build(cache=cache, use_conversation_history=False).run("s1", "t1", "q")

    assert "get" not in cache.names()
    assert "set" not in cache.names()
    assert [m.content for m in cache.store["s1"]] == ["m0", "m1", "m2", "m3"]


async def test_stateless_mode_still_logs_history():
    """Sub-agents keep no window, but their turns are still auditable."""
    history = FakeHistory()
    await build(history=history, use_conversation_history=False).run("s1", "t1", "q")

    assert history.user_messages[0]["message"] == "q"
    assert len(history.finals) == 1


async def test_stateless_mode_still_locks():
    cache = FakeCache()
    await build(cache=cache, use_conversation_history=False).run("s1", "t1", "q")

    assert "acquire_lock" in cache.names()
    assert cache.names().count("release_lock") == 1


# --------------------------------------------------------------------------
# shared-instance safety
# --------------------------------------------------------------------------


async def test_system_prompt_is_never_mutated_across_turns():
    """Regression: memory examples were appended to self.system_prompt, so
    every turn on this shared instance inherited the previous turn's memory."""
    history = FakeHistory(memory=[{"q": "first"}])
    interactor = build(history=history)

    await interactor.run("s1", "t1", "one")
    assert interactor.system_prompt == "BASE PROMPT"

    history.memory = [{"q": "second"}]
    loop2 = FakeAgentLoop()
    interactor.agent_loop = loop2
    await interactor.run("s1", "t2", "two")

    sent = " ".join(m.content or "" for m in loop2.last_messages)
    assert "second" in sent
    assert "first" not in sent


async def test_separate_sessions_keep_separate_windows():
    cache = FakeCache()
    interactor = build(cache=cache)

    await interactor.run("sa", "t1", "question a")
    await interactor.run("sb", "t2", "question b")

    assert [m.content for m in cache.store["sa"]] == ["question a", "answer"]
    assert [m.content for m in cache.store["sb"]] == ["question b", "answer"]
