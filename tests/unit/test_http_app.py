"""The HTTP edge, against a fake runtime.

The routes should be boring, and these tests exist to keep them that way.
What they check is mostly translation - JSON in, arguments out, exceptions to
status codes - plus three things that are genuinely decisions:

    /run on the orchestrator is 404, and /ask on a sub-agent is too, with a
    message that says what to do instead. A process that accepted both would
    hide a misconfigured AGENT_KEY until an answer came back wrong.

    a busy session is 409, not 500. The caller did nothing wrong and retrying
    is correct, which is a different instruction from "this broke".

    /health reports the tables the agent actually resolved. A sub-agent that
    is up but reading the wrong tables is the failure the whole design exists
    to prevent, and this makes it visible from outside the process.

The runtime is injected through create_app, so none of this needs Postgres,
Redis or a model.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from adapters.inbound.http import schemas
from adapters.inbound.http.app import create_app, final_answer
from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.entities.provider_spec import AgentStatus, ProviderSpec
from domain.exceptions import CacheError, SessionBusyError
from libs.agent_core import config
from libs.agent_core.agent_startup import ReadyAgent
from libs.agent_core.composition import Runtime
from libs.agent_core.schema_bootstrap import AgentSchema


class FakeTurn:
    """RunAgentTurn's shape, recording what it was asked."""

    def __init__(self, answer="Twelve books match.", raises=None, pagination=None,
                 delegated=()):
        self.answer = answer
        self.raises = raises
        self.pagination = pagination or {}
        self.delegated = list(delegated)
        self.calls = []

    async def run(self, session_id, turn_id, user_input):
        self.calls.append({"session_id": session_id, "turn_id": turn_id,
                           "user_input": user_input})
        if self.raises:
            raise self.raises

        messages = [ChatMessage(role=Role.USER, content=user_input)]
        for i, (agent, question) in enumerate(self.delegated):
            messages.append(ChatMessage(
                role=Role.ASSISTANT, content=None,
                tool_calls=[ToolCall(id=f"c{i}", name=agent,
                                     args={"query": question})],
            ))
            messages.append(ChatMessage(role=Role.TOOL, content="{}", tool_call_id=f"c{i}"))
        messages.append(ChatMessage(role=Role.ASSISTANT, content=self.answer))

        return AgentTurnResult(messages=messages, pagination=self.pagination)


def sub_agent_runtime(turn=None, tables=("books", "authors")):
    spec = ProviderSpec(
        name="catalog", display_name="Catalogue",
        system_prompt="Answer about the catalogue.",
        description="Books and authors.", db_role="app_catalog",
        status=AgentStatus.ACTIVE,
    )
    schema = AgentSchema(tables=tuple(tables), schema={}, filters={}, classified={})
    return Runtime(kind="sub_agent", turn=turn or FakeTurn(),
                   agent=ReadyAgent(spec=spec, schema=schema))


def orchestrator_runtime(turn=None, routes=("catalog", "circulation"), registry=None):
    return Runtime(kind="orchestrator", turn=turn or FakeTurn(), routes_to=routes,
                   registry=registry)


def client_for(runtime):
    async def open_it():
        return runtime
    return TestClient(create_app(open_it))


# --------------------------------------------------------------------------
# health
# --------------------------------------------------------------------------


def test_health_reports_the_tables_the_agent_resolved():
    """Up is not the interesting question. Up and reading the right tables
    is, and that is answerable from outside without a query or a log."""
    with client_for(sub_agent_runtime()) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["kind"] == "sub_agent"
    assert body["agent"] == "catalog"
    assert body["tables"] == ["books", "authors"]


def test_health_on_the_orchestrator_reports_who_it_routes_to():
    with client_for(orchestrator_runtime()) as client:
        body = client.get("/health").json()

    assert body["kind"] == "orchestrator"
    assert body["agent"] is None
    assert body["routes_to"] == ["catalog", "circulation"]


# --------------------------------------------------------------------------
# /run - the sub-agent endpoint
# --------------------------------------------------------------------------


def test_run_answers_a_delegated_question():
    turn = FakeTurn(answer="Twelve.")
    with client_for(sub_agent_runtime(turn)) as client:
        body = client.post("/run", json={
            "session_id": "s-1",
            "user_input": "How many Arabic novels under 300 pages?",
            "context": {"cursor": None, "turn_id": "t-1"},
        }).json()

    assert body["answer"] == "Twelve."
    assert body["turn_id"] == "t-1"
    assert turn.calls[0]["user_input"].startswith("How many Arabic")


def test_run_accepts_exactly_what_the_delegate_adapter_posts():
    """The caller is HttpDelegateToolAdapter, not a person. Any divergence
    here is a 422 that only appears once two components run together."""
    from adapters.outbound.tools.http_delegate_tool_adapter import (
        HttpDelegateToolAdapter,
    )
    import inspect

    source = inspect.getsource(HttpDelegateToolAdapter.call_tool)
    for field in ('"session_id"', '"user_input"', '"context"', '"cursor"', '"turn_id"'):
        assert field in source

    with client_for(sub_agent_runtime()) as client:
        posted = {
            "session_id": "", "user_input": "q",
            "context": {"cursor": None, "turn_id": None},
        }
        assert client.post("/run", json=posted).status_code == 200


def test_run_invents_correlation_values_when_the_caller_omits_them():
    """A sub-agent can be called directly while debugging, and refusing for
    want of a uuid would make that harder for no benefit - the ids exist to
    tie rows together, and one generated here still does."""
    turn = FakeTurn()
    with client_for(sub_agent_runtime(turn)) as client:
        body = client.post("/run", json={"user_input": "q"}).json()

    assert body["session_id"].startswith("delegate:")
    assert body["turn_id"]
    assert turn.calls[0]["session_id"] == body["session_id"]


def test_run_returns_pagination_per_tool():
    """Per tool, not one cursor: a turn may have paged through more than one,
    and collapsing them makes "the next page" ambiguous."""
    turn = FakeTurn(pagination={
        "db_execute": PaginationState(has_more=True, next_cursor="c1", pages_fetched=2),
    })
    with client_for(sub_agent_runtime(turn)) as client:
        body = client.post("/run", json={"user_input": "q"}).json()

    assert body["pagination"]["db_execute"] == {
        "has_more": True, "next_cursor": "c1", "pages_fetched": 2,
    }


def test_run_rejects_an_empty_question():
    with client_for(sub_agent_runtime()) as client:
        assert client.post("/run", json={"user_input": ""}).status_code == 422


def test_run_on_the_orchestrator_says_what_to_do_instead():
    """A misconfigured AGENT_KEY is otherwise invisible until an answer comes
    back wrong."""
    with client_for(orchestrator_runtime()) as client:
        response = client.post("/run", json={"user_input": "q"})

    assert response.status_code == 404
    assert "/ask" in response.json()["detail"]


# --------------------------------------------------------------------------
# /ask - the orchestrator endpoint
# --------------------------------------------------------------------------


def test_ask_answers_a_persons_question():
    turn = FakeTurn(answer="Twelve books.")
    with client_for(orchestrator_runtime(turn)) as client:
        body = client.post("/ask", json={
            "question": "How many Arabic novels do we have?", "session_id": "s-9",
        }).json()

    assert body["answer"] == "Twelve books."
    assert body["session_id"] == "s-9"
    assert turn.calls[0]["session_id"] == "s-9"


def test_ask_requires_a_session_because_it_remembers():
    """The orchestrator is the component with a conversation window, so a
    question with nowhere to put it is a question that silently loses its
    own follow-ups."""
    with client_for(orchestrator_runtime()) as client:
        assert client.post("/ask", json={"question": "q"}).status_code == 422


def test_ask_on_a_sub_agent_names_the_agent_it_is_serving():
    with client_for(sub_agent_runtime()) as client:
        response = client.post("/ask", json={"question": "q", "session_id": "s"})

    assert response.status_code == 404
    assert "catalog" in response.json()["detail"]
    assert "/run" in response.json()["detail"]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_a_busy_session_is_a_conflict_not_a_server_error():
    """The caller did nothing wrong and retrying is correct. 500 tells them
    the opposite."""
    turn = FakeTurn(raises=SessionBusyError("Session is currently busy."))
    with client_for(orchestrator_runtime(turn)) as client:
        response = client.post("/ask", json={"question": "q", "session_id": "s"})

    assert response.status_code == 409


# --------------------------------------------------------------------------
# picking the answer out of a turn
# --------------------------------------------------------------------------


def test_the_answer_is_the_last_thing_the_model_said():
    result = AgentTurnResult(messages=[
        ChatMessage(role=Role.USER, content="q"),
        ChatMessage(role=Role.ASSISTANT, content="thinking"),
        ChatMessage(role=Role.TOOL, content='{"rows": []}'),
        ChatMessage(role=Role.ASSISTANT, content="Twelve."),
    ])
    assert final_answer(result) == "Twelve."


def test_a_turn_that_ends_on_a_tool_result_does_not_return_json_to_a_person():
    """A turn can end on a tool message when the loop hits its page limit.
    messages[-1] would hand the user a blob of rows and call it an answer."""
    result = AgentTurnResult(messages=[
        ChatMessage(role=Role.ASSISTANT, content="Here is what I found."),
        ChatMessage(role=Role.TOOL, content='{"rows": [1, 2, 3]}'),
    ])
    assert final_answer(result) == "Here is what I found."


def test_an_empty_turn_gives_an_empty_answer_rather_than_raising():
    assert final_answer(AgentTurnResult(messages=[])) == ""


# --------------------------------------------------------------------------
# /agents
# --------------------------------------------------------------------------


def test_agents_lists_what_the_registry_holds():
    from adapters.outbound.registry.file_agent_registry_adapter import (
        FileAgentRegistryAdapter,
    )

    registry = FileAgentRegistryAdapter("seeds/agents.example.json")
    with client_for(orchestrator_runtime(registry=registry)) as client:
        body = client.get("/agents").json()

    assert [a["key"] for a in body] == ["catalog", "circulation"]
    assert body[0]["display_name"] == "Catalogue"
    assert body[0]["status"] == "active"


def test_agents_is_404_where_there_is_no_registry():
    with client_for(sub_agent_runtime()) as client:
        assert client.get("/agents").status_code == 404


# --------------------------------------------------------------------------
# what the orchestrator actually asked
#
# Returned so a wrong answer can be attributed. The orchestrator rewrites the
# user's question into a self-contained one, and that rewrite can drop a
# constraint - asked for novels it may send "how many English books", and the
# agent answers that correctly. From outside, the two failures look the same
# and the wrong component gets changed.
# --------------------------------------------------------------------------


def test_ask_reports_the_question_that_was_delegated():
    turn = FakeTurn(delegated=[("catalog", "How many English novels are under 400 pages?")])
    with client_for(orchestrator_runtime(turn)) as client:
        body = client.post("/ask", json={
            "question": "كم رواية إنكليزية أقل من ٤٠٠ صفحة؟", "session_id": "s",
        }).json()

    assert body["delegated"] == [{
        "agent": "catalog",
        "question": "How many English novels are under 400 pages?",
    }]


def test_several_delegations_are_reported_in_order():
    turn = FakeTurn(delegated=[("catalog", "How many books?"),
                               ("circulation", "How many are on loan?")])
    with client_for(orchestrator_runtime(turn)) as client:
        body = client.post("/ask", json={"question": "q", "session_id": "s"}).json()

    assert [d["agent"] for d in body["delegated"]] == ["catalog", "circulation"]


def test_a_turn_with_no_delegation_reports_none():
    with client_for(orchestrator_runtime()) as client:
        body = client.post("/ask", json={"question": "q", "session_id": "s"}).json()

    assert body["delegated"] == []


def test_a_sub_agent_reports_no_delegation():
    """It delegates to nobody, and its own SQL tool calls are not questions
    asked of an agent - reporting them here would read as a fan-out that
    never happened."""
    turn = FakeTurn(delegated=[("db_execute", "SELECT 1")])
    with client_for(sub_agent_runtime(turn)) as client:
        body = client.post("/run", json={"user_input": "q"}).json()

    assert "delegated" not in body or body["delegated"] == []


# --------------------------------------------------------------------------
# what /health says beyond "ok"
# --------------------------------------------------------------------------


def test_health_names_the_model_it_is_running(monkeypatch):
    """Two processes of the same image differing only in a model name is a
    normal deployment, and telling them apart should not need a question."""
    monkeypatch.setattr(config, "QWEN_MODEL", "qwen-plus")

    with client_for(sub_agent_runtime()) as client:
        assert client.get("/health").json()["model"] == "qwen-plus"


# --------------------------------------------------------------------------
# request size limits
# --------------------------------------------------------------------------


def test_a_question_longer_than_the_limit_is_refused():
    """One question is an embedding call plus several model calls priced per
    token, so an unbounded body is an unbounded bill."""
    with client_for(orchestrator_runtime()) as client:
        response = client.post("/ask", json={
            "question": "x" * (schemas.MAX_QUESTION_CHARS + 1), "session_id": "s",
        })

    assert response.status_code == 422


def test_a_delegated_question_longer_than_the_limit_is_refused():
    with client_for(sub_agent_runtime()) as client:
        response = client.post("/run", json={
            "user_input": "x" * (schemas.MAX_QUESTION_CHARS + 1),
        })

    assert response.status_code == 422


def test_a_question_at_the_limit_is_accepted():
    with client_for(orchestrator_runtime()) as client:
        response = client.post("/ask", json={
            "question": "x" * schemas.MAX_QUESTION_CHARS, "session_id": "s",
        })

    assert response.status_code == 200


def test_an_empty_question_is_still_refused():
    with client_for(orchestrator_runtime()) as client:
        assert client.post("/ask", json={"question": "", "session_id": "s"}).status_code == 422


# --------------------------------------------------------------------------
# whether a 500 names the exception
# --------------------------------------------------------------------------


def test_a_domain_error_names_itself_when_errors_are_exposed(monkeypatch):
    monkeypatch.setattr(config, "EXPOSE_ERRORS", True)
    turn = FakeTurn(raises=CacheError("Extra data: line 1 column 9"))

    with client_for(orchestrator_runtime(turn)) as client:
        response = client.post("/ask", json={"question": "q", "session_id": "s"})

    assert response.status_code == 500
    assert response.json()["detail"] == "CacheError: Extra data: line 1 column 9"


def test_a_domain_error_is_generic_when_errors_are_not_exposed(monkeypatch):
    """A deployment reachable by people who are not operating it should not
    answer questions about its internals."""
    monkeypatch.setattr(config, "EXPOSE_ERRORS", False)
    turn = FakeTurn(raises=CacheError("Extra data: line 1 column 9"))

    with client_for(orchestrator_runtime(turn)) as client:
        response = client.post("/ask", json={"question": "q", "session_id": "s"})

    assert response.status_code == 500
    assert "CacheError" not in response.json()["detail"]
    assert "logs" in response.json()["detail"]


def test_the_exception_is_logged_either_way(monkeypatch, capsys):
    """Hiding the exception from the caller must not hide it from whoever is
    debugging. Read from stderr rather than caplog, because the app installs
    its own root handler on startup."""
    monkeypatch.setattr(config, "EXPOSE_ERRORS", False)
    turn = FakeTurn(raises=CacheError("boom"))

    with client_for(orchestrator_runtime(turn)) as client:
        client.post("/ask", json={"question": "q", "session_id": "s"})

    written = capsys.readouterr().err
    assert "domain error" in written
    assert "CacheError: boom" in written, "the traceback stays in the log"
