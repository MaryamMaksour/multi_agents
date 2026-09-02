"""turn_id travels from the interactor to the sub-agent without the model.

An orchestrator turn fans out to several sub-agent turns. Their history rows
are tied back together by turn_id - and the model must not be the one that
carries it, because a value the model carries is a value the model can drop
or invent. So it goes interactor -> loop -> call_tool(turn_id=...) ->
request body, alongside the model's args rather than inside them.
"""

from __future__ import annotations

import httpx

from adapters.outbound.agent_loop.langgraph_agent_loop_adapter import LangGraphAgentLoopAdapter
from adapters.outbound.tools.http_delegate_tool_adapter import HttpDelegateToolAdapter
from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter
from domain.entities.chat_message import ChatMessage, Role, ToolCall

from tests.conftest import FakeAgentLoop, FakeCache, FakeDatabase, FakeEmbeddings, FakeLLM, FakeTools
from tests.unit.test_run_agent_turn import build


async def test_the_interactor_hands_the_turn_id_to_the_loop():
    loop = FakeAgentLoop()
    await build(agent_loop=loop).run("s1", "turn-42", "hello")
    assert loop.turn_ids == ["turn-42"]


async def test_the_loop_hands_the_turn_id_to_every_tool_call():
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(id="c1", name="catalog", args={"query": "books?"}),
            ToolCall(id="c2", name="circulation", args={"query": "loans?"}),
        ]),
        ChatMessage(role=Role.ASSISTANT, content="done"),
    ])
    tools = FakeTools()

    await LangGraphAgentLoopAdapter(llm=llm, tools=tools).run(
        [ChatMessage(role=Role.USER, content="q")], turn_id="turn-42",
    )

    assert tools.turn_ids == ["turn-42", "turn-42"]
    # ...and it never became part of what the model asked for.
    assert all("turn_id" not in args for _, args in tools.calls)


async def test_the_loop_without_a_turn_id_still_works():
    llm = FakeLLM([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="c1", name="x", args={})]),
        ChatMessage(role=Role.ASSISTANT, content="done"),
    ])
    tools = FakeTools()
    await LangGraphAgentLoopAdapter(llm=llm, tools=tools).run([ChatMessage(role=Role.USER, content="q")])
    assert tools.turn_ids == [None]


async def test_the_delegate_posts_the_turn_id_and_not_a_session():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"answer": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = HttpDelegateToolAdapter(
            client=client, tool_urls={"catalog": "http://catalog/run"},
            tool_descriptions={"catalog": "Books."},
        )
        await tool.call_tool(
            "catalog",
            {"query": "how many?", "cursor": None, "turn_id": "forged", "session_id": "forged"},
            turn_id="turn-42",
        )

    assert seen == [{
        "session_id": "",
        "user_input": "how many?",
        "context": {"cursor": None, "turn_id": "turn-42"},
    }]


async def test_sql_tools_accept_and_ignore_the_turn_id():
    """Same port, same signature - but SQL tools never leave the process, so
    there is nothing for the id to correlate against."""
    tool = SqlToolAdapter(
        db=FakeDatabase(), embeddings=FakeEmbeddings(), cache=FakeCache(),
        allowed_tables=["books"], schema={"books": {"columns": "id int4"}},
        filters={}, lsit_values={}, dist_op="<=>", vector_ttl_seconds=900,
    )
    result = await tool.call_tool("get_table_schema", {"tables": ["books"]}, turn_id="turn-42")
    assert "books" in result
