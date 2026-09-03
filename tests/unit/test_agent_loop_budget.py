"""The loop's iteration budget, and the decision log that goes with it.

Without a budget the loop runs until LangGraph's own recursion limit raises -
after paying for every call, and with nothing to return to the caller. With
one it stops at a number a deployment chose and answers with what the turn
has.

The logging is the other half of the same problem. "How many Arabic novels
under 300 pages" becoming a query with no genre filter is the single most
useful line in the system, and it is the only thing that separates a bad
delegation from a bad sub-agent - so the tool name and the model's arguments
in full have to be in the log, not only in the history row.
"""

from __future__ import annotations

import logging

import pytest

from adapters.outbound.agent_loop.langgraph_agent_loop_adapter import LangGraphAgentLoopAdapter
from domain.entities.chat_message import ChatMessage, Role, ToolCall

from tests.conftest import FakeLLM, FakeTools


def wants_a_tool(i: int) -> ChatMessage:
    return ChatMessage(
        role=Role.ASSISTANT,
        content=None,
        tool_calls=[ToolCall(id=f"c{i}", name="db_execute", args={"sql": f"SELECT {i}"})],
    )


class AlwaysCallsATool(FakeLLM):
    """A model that never stops asking for one more tool call - the loop's
    worst case, and the reason the budget exists."""

    async def achat(self, messages):
        self.received.append(list(messages))
        return wants_a_tool(len(self.received))


def loop(llm, max_steps: int) -> LangGraphAgentLoopAdapter:
    return LangGraphAgentLoopAdapter(llm=llm, tools=FakeTools(), max_steps=max_steps)


@pytest.mark.asyncio
async def test_the_loop_stops_at_the_budget():
    llm = AlwaysCallsATool()

    await loop(llm, max_steps=3).run([ChatMessage(role=Role.USER, content="hi")])

    assert len(llm.received) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [1, 2, 5])
async def test_the_budget_is_the_number_of_model_calls(budget):
    llm = AlwaysCallsATool()

    await loop(llm, max_steps=budget).run([ChatMessage(role=Role.USER, content="hi")])

    assert len(llm.received) == budget


@pytest.mark.asyncio
async def test_stopping_early_still_returns_the_turn():
    """Reaching the budget is not an error: the caller gets the messages the
    turn produced, so a partial answer beats an exception."""
    llm = AlwaysCallsATool()

    result = await loop(llm, max_steps=2).run([ChatMessage(role=Role.USER, content="hi")])

    assert result.messages, "the turn's messages are returned, not discarded"


@pytest.mark.asyncio
async def test_a_turn_that_finishes_early_is_untouched_by_the_budget():
    llm = FakeLLM([ChatMessage(role=Role.ASSISTANT, content="twelve")])

    result = await loop(llm, max_steps=12).run([ChatMessage(role=Role.USER, content="how many?")])

    assert len(llm.received) == 1
    assert result.messages[-1].content == "twelve"


@pytest.mark.asyncio
async def test_reaching_the_budget_says_so_in_the_log(caplog):
    llm = AlwaysCallsATool()

    with caplog.at_level(logging.WARNING):
        await loop(llm, max_steps=1).run([ChatMessage(role=Role.USER, content="hi")])

    assert any("budget" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_tool_call_is_logged_with_its_arguments_in_full(caplog):
    """Truncated arguments would hide the filter that was not applied, which
    is the whole reason for logging them."""
    llm = FakeLLM([
        wants_a_tool(1),
        ChatMessage(role=Role.ASSISTANT, content="done"),
    ])

    with caplog.at_level(logging.INFO):
        await LangGraphAgentLoopAdapter(llm=llm, tools=FakeTools()).run(
            [ChatMessage(role=Role.USER, content="hi")], turn_id="t-42",
        )

    logged = [r for r in caplog.records if r.message == "tool called"]
    assert len(logged) == 1
    assert logged[0].tool == "db_execute"
    assert logged[0].tool_args == {"sql": "SELECT 1"}
    assert logged[0].turn_id == "t-42"


@pytest.mark.asyncio
async def test_the_model_reply_is_logged_with_the_tools_it_asked_for(caplog):
    llm = FakeLLM([wants_a_tool(1), ChatMessage(role=Role.ASSISTANT, content="done")])

    with caplog.at_level(logging.INFO):
        await LangGraphAgentLoopAdapter(llm=llm, tools=FakeTools()).run(
            [ChatMessage(role=Role.USER, content="hi")], turn_id="t-42",
        )

    answered = [r for r in caplog.records if r.message == "model answered"]
    assert [r.tool_calls for r in answered] == [["db_execute"], []]
    assert all(r.turn_id == "t-42" for r in answered)


@pytest.mark.asyncio
async def test_a_turn_stopped_at_the_budget_leaves_no_unanswered_tool_call():
    """The orchestrator persists the turn. A trailing tool call with no
    result after it is a conversation an OpenAI-compatible provider rejects
    on the next question, so the budget must not create one."""
    llm = AlwaysCallsATool()

    result = await loop(llm, max_steps=2).run([ChatMessage(role=Role.USER, content="hi")])

    last = result.messages[-1]
    assert not last.tool_calls
    assert last.content, "the caller gets words, not an empty assistant message"


@pytest.mark.asyncio
async def test_a_normal_final_answer_is_left_alone():
    llm = FakeLLM([ChatMessage(role=Role.ASSISTANT, content="twelve")])

    result = await loop(llm, max_steps=4).run([ChatMessage(role=Role.USER, content="how many?")])

    assert result.messages[-1].content == "twelve"


@pytest.mark.asyncio
async def test_a_budget_above_langgraphs_recursion_limit_still_answers():
    """LangGraph's default limit is 25 and a tool-using turn spends two nodes
    per model call, so an accepted budget of 40 would raise there before the
    loop ever stopped itself - a 500 instead of the partial answer."""
    llm = AlwaysCallsATool()

    result = await loop(llm, max_steps=40).run([ChatMessage(role=Role.USER, content="hi")])

    assert len(llm.received) == 40
    assert result.messages
