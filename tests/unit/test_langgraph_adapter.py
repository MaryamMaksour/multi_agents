"""LangGraphAgentLoopAdapter - the boundary where LangChain is contained.

Two concerns.

Translation. ChatMessage and LangChain's BaseMessage cross in both
directions on every turn, and a field dropped in translation is invisible:
nothing raises, the model simply stops receiving something. That is exactly
how tool_calls went missing once - computed and then not passed to
AIMessage - so these tests assert on the round trip, not on either half.

Containment. Nothing LangChain-shaped may escape this adapter. run() takes
ChatMessage and returns AgentTurnResult; if a BaseMessage ever appears in
what it returns, LangGraph has stopped being one adapter's business.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from adapters.outbound.agent_loop.langgraph_agent_loop_adapter import LangGraphAgentLoopAdapter
from domain.entities.agent_turn import AgentTurnResult
from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.exceptions import LLMRequestError, ToolExecutionError, UnknownToolError

from tests.conftest import FakeLLM, FakeTools

to_lc = LangGraphAgentLoopAdapter._to_lc_message
to_domain = LangGraphAgentLoopAdapter._to_chat_message


def adapter(llm=None, tools=None, **kw) -> LangGraphAgentLoopAdapter:
    return LangGraphAgentLoopAdapter(
        llm=llm or FakeLLM(), tools=tools or FakeTools(), **kw
    )


# --------------------------------------------------------------------------
# the class survived its own repair
# --------------------------------------------------------------------------


def test_every_method_is_on_the_class():
    """Regression: a stray dedent once ended the class body early, leaving
    _take_action at module level and swallowing _should_continue,
    _build_graph and run into its body as unreachable nested functions. The
    file still compiled; the class simply had no run()."""
    for name in ("_to_chat_message", "_to_lc_message", "_call_llm", "_invoke_one_tool",
                 "_take_action", "_should_continue", "_build_graph", "run"):
        assert callable(getattr(LangGraphAgentLoopAdapter, name, None)), f"missing {name}"


def test_constructing_builds_the_graph():
    assert adapter()._graph is not None


# --------------------------------------------------------------------------
# domain -> LangChain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role, expected", [
    (Role.SYSTEM, SystemMessage),
    (Role.USER, HumanMessage),
    (Role.ASSISTANT, AIMessage),
])
def test_each_role_maps_to_its_message_class(role, expected):
    assert isinstance(to_lc(ChatMessage(role=role, content="x")), expected)


def test_a_tool_result_keeps_its_call_id():
    lc = to_lc(ChatMessage(role=Role.TOOL, content="{}", tool_call_id="call_7"))
    assert isinstance(lc, ToolMessage)
    assert lc.tool_call_id == "call_7"


def test_tool_calls_reach_the_ai_message():
    """Regression: these were built into a local variable and then never
    passed to AIMessage, so the model was told nothing had been called."""
    msg = ChatMessage(
        role=Role.ASSISTANT,
        content="",
        tool_calls=[ToolCall(id="c1", name="db_execute", args={"query": "SELECT 1"})],
    )
    lc = to_lc(msg)
    assert len(lc.tool_calls) == 1
    assert lc.tool_calls[0]["name"] == "db_execute"
    assert lc.tool_calls[0]["args"] == {"query": "SELECT 1"}
    assert lc.tool_calls[0]["id"] == "c1"


def test_tool_calls_are_read_as_attributes_not_dict_keys():
    """ToolCall is a dataclass. An earlier version called .get() on it, which
    raises rather than silently returning None - but only once a tool is
    actually called."""
    msg = ChatMessage(role=Role.ASSISTANT, content="",
                      tool_calls=[ToolCall(id="c1", name="t", args={})])
    to_lc(msg)  # must not raise


def test_an_assistant_message_without_tool_calls_is_still_valid():
    lc = to_lc(ChatMessage(role=Role.ASSISTANT, content="just an answer"))
    assert lc.content == "just an answer"
    assert lc.tool_calls == []


def test_none_content_becomes_empty_string():
    """A tool-calling assistant message often has no text, and AIMessage
    will not accept None there."""
    lc = to_lc(ChatMessage(role=Role.ASSISTANT, content=None))
    assert lc.content == ""


# --------------------------------------------------------------------------
# LangChain -> domain
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lc, role", [
    (SystemMessage(content="s"), Role.SYSTEM),
    (HumanMessage(content="h"), Role.USER),
    (AIMessage(content="a"), Role.ASSISTANT),
])
def test_each_message_class_maps_back_to_its_role(lc, role):
    assert to_domain(lc).role is role


def test_a_tool_message_maps_back_with_its_call_id():
    domain = to_domain(ToolMessage(content="{}", tool_call_id="call_9"))
    assert domain.role is Role.TOOL
    assert domain.tool_call_id == "call_9"


def test_tool_calls_survive_the_return_trip():
    lc = AIMessage(content="", tool_calls=[
        {"name": "db_execute", "args": {"query": "SELECT 1"}, "id": "c1"},
    ])
    domain = to_domain(lc)
    assert isinstance(domain.tool_calls[0], ToolCall)
    assert domain.tool_calls[0].name == "db_execute"
    assert domain.tool_calls[0].args == {"query": "SELECT 1"}


def test_no_tool_calls_becomes_none_not_an_empty_list():
    """RunAgentTurn and the graph both test truthiness here; an empty list
    would read the same, but None is what the entity documents."""
    assert to_domain(AIMessage(content="a")).tool_calls is None


def test_an_unknown_message_type_raises():
    class Strange:
        content = "?"

    with pytest.raises(ValueError):
        to_domain(Strange())


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("original", [
    ChatMessage(role=Role.SYSTEM, content="you are an agent"),
    ChatMessage(role=Role.USER, content="how many books?"),
    ChatMessage(role=Role.ASSISTANT, content="420"),
    ChatMessage(role=Role.TOOL, content='{"rows": []}', tool_call_id="c1"),
    ChatMessage(role=Role.ASSISTANT, content="",
                tool_calls=[ToolCall(id="c1", name="db_execute", args={"q": 1})]),
])
def test_a_message_survives_a_full_round_trip(original):
    back = to_domain(to_lc(original))

    assert back.role is original.role
    assert (back.content or "") == (original.content or "")
    assert back.tool_call_id == original.tool_call_id
    if original.tool_calls:
        assert [(c.id, c.name, c.args) for c in back.tool_calls] == \
               [(c.id, c.name, c.args) for c in original.tool_calls]
    else:
        assert not back.tool_calls


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------


pytestmark_async = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_run_returns_only_the_messages_this_turn_produced():
    """The input is not echoed back: RunAgentTurn concatenates, so returning
    the input would duplicate the whole window every turn."""
    llm = FakeLLM([ChatMessage(role=Role.ASSISTANT, content="the answer")])
    result = await adapter(llm=llm).run([
        ChatMessage(role=Role.SYSTEM, content="prompt"),
        ChatMessage(role=Role.USER, content="question"),
    ])

    assert isinstance(result, AgentTurnResult)
    assert [m.content for m in result.messages] == ["the answer"]


@pytest.mark.asyncio
async def test_run_returns_domain_messages_only():
    """Containment: nothing LangChain-shaped may leave this adapter."""
    result = await adapter().run([ChatMessage(role=Role.USER, content="hi")])
    assert all(isinstance(m, ChatMessage) for m in result.messages)


@pytest.mark.asyncio
async def test_a_tool_call_is_executed_and_its_result_returned():
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="db_execute", args={"query": "SELECT 1"})]),
        ChatMessage(role=Role.ASSISTANT, content="420 books"),
    ])
    tools = FakeTools({"db_execute": {"rows": [{"n": 420}], "has_more": False}})

    result = await adapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="how many?")])

    assert tools.calls == [("db_execute", {"query": "SELECT 1"})]
    roles = [m.role for m in result.messages]
    assert Role.TOOL in roles
    assert result.messages[-1].content == "420 books"


@pytest.mark.asyncio
async def test_several_tool_calls_in_one_step_run_concurrently():
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(id="c1", name="get_table_schema", args={"tables": ["books"]}),
            ToolCall(id="c2", name="get_filter", args={"columns": ["genre"], "table_name": "books"}),
        ]),
        ChatMessage(role=Role.ASSISTANT, content="done"),
    ])
    tools = FakeTools({"get_table_schema": {"books": "..."}, "get_filter": {"genre": "ENUM"}})

    await adapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="q")])

    assert {name for name, _ in tools.calls} == {"get_table_schema", "get_filter"}


@pytest.mark.asyncio
async def test_a_failing_tool_becomes_a_message_not_a_crash():
    """A tool failure is information for the model to react to. Raising here
    would end the turn and lose everything already gathered."""
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="db_execute", args={})]),
        ChatMessage(role=Role.ASSISTANT, content="I could not run that"),
    ])
    tools = FakeTools(error=ToolExecutionError("syntax error at or near FROM"))

    result = await adapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="q")])

    tool_msg = next(m for m in result.messages if m.role is Role.TOOL)
    assert "syntax error" in json.loads(tool_msg.content)["error"]


@pytest.mark.asyncio
async def test_an_unknown_tool_also_becomes_a_message():
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="no_such_tool", args={})]),
        ChatMessage(role=Role.ASSISTANT, content="sorry"),
    ])
    tools = FakeTools(error=UnknownToolError("Unknown tool: no_such_tool"))

    result = await adapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="q")])
    assert any(m.role is Role.TOOL for m in result.messages)


@pytest.mark.asyncio
async def test_an_llm_failure_propagates():
    """Unlike a tool failure, there is no way to continue - and the interactor
    relies on this to release the session lock."""
    llm = FakeLLM(error=LLMRequestError("upstream 503"))
    with pytest.raises(LLMRequestError):
        await adapter(llm=llm).run([ChatMessage(role=Role.USER, content="q")])


# --------------------------------------------------------------------------
# the pagination budget
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagination_state_is_recorded_per_tool():
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="db_execute", args={})]),
        ChatMessage(role=Role.ASSISTANT, content="done"),
    ])
    tools = FakeTools({"db_execute": {"rows": [], "has_more": True, "next_cursor": "cur1"}})

    result = await adapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="q")])

    state = result.pagination["db_execute"]
    assert state.has_more is True
    assert state.next_cursor == "cur1"
    assert state.pages_fetched == 1


@pytest.mark.asyncio
async def test_a_tool_without_pagination_records_no_state():
    """Only paginating tools should appear, or the budget is spent on tools
    that never page."""
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="get_table_schema", args={})]),
        ChatMessage(role=Role.ASSISTANT, content="done"),
    ])
    tools = FakeTools({"get_table_schema": {"books": "columns..."}})

    result = await adapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="q")])
    assert result.pagination == {}


@pytest.mark.asyncio
async def test_the_page_budget_stops_a_runaway_loop():
    """Without it, a model that keeps seeing has_more pages until the context
    is full. The refusal is returned as a tool result so the model can adapt
    rather than being cut off."""
    calls_before_limit = 2
    replies = []
    for _ in range(calls_before_limit + 3):
        replies.append(ChatMessage(role=Role.ASSISTANT, content="",
                                   tool_calls=[ToolCall(id="c", name="db_execute", args={})]))
    replies.append(ChatMessage(role=Role.ASSISTANT, content="stopping"))

    llm = FakeLLM(replies)
    tools = FakeTools({"db_execute": {"rows": [], "has_more": True, "next_cursor": "c"}})

    result = await adapter(llm=llm, tools=tools,
                           max_pages_per_tool=calls_before_limit).run(
        [ChatMessage(role=Role.USER, content="everything")]
    )

    assert len(tools.calls) <= calls_before_limit
    refusals = [
        m for m in result.messages
        if m.role is Role.TOOL and "Pagination limit" in str(m.content)
    ]
    assert refusals, "the model should be told why it stopped getting pages"
