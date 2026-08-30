"""Domain entities.

Small, but two things here are worth pinning. Mutable default arguments on a
dataclass are a classic way to have every instance quietly share one list -
these use default_factory, and that is the kind of property that is easy to
break later while refactoring. And Role's values are what the LLM adapter
sends on the wire, so they are part of an external contract, not an internal
naming choice.
"""

from __future__ import annotations

from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.entities.chat_message import ChatMessage, Role, ToolCall


# --------------------------------------------------------------------------
# Role
# --------------------------------------------------------------------------


def test_role_values_match_the_wire_format():
    """These strings go straight into the OpenAI request body; renaming one
    would be a protocol change, not a rename."""
    assert {r.value for r in Role} == {"system", "user", "assistant", "tool"}


def test_roles_are_compared_by_identity_not_string():
    assert Role.USER is Role("user")
    assert Role.USER != "user"


# --------------------------------------------------------------------------
# ChatMessage
# --------------------------------------------------------------------------


def test_a_message_needs_only_a_role():
    m = ChatMessage(role=Role.ASSISTANT)
    assert m.content is None
    assert m.tool_calls is None
    assert m.tool_call_id is None


def test_tool_result_messages_carry_the_call_id():
    """Without it the model cannot pair a result with the call that asked
    for it, and the provider rejects the request."""
    m = ChatMessage(role=Role.TOOL, content="{}", tool_call_id="call_1", name="db_execute")
    assert m.tool_call_id == "call_1"
    assert m.name == "db_execute"


def test_instances_do_not_share_state():
    a = ChatMessage(role=Role.USER, content="a")
    b = ChatMessage(role=Role.USER, content="b")
    a.tool_calls = [ToolCall(id="1", name="t")]
    assert b.tool_calls is None


# --------------------------------------------------------------------------
# ToolCall
# --------------------------------------------------------------------------


def test_tool_call_args_default_to_an_empty_dict_per_instance():
    """A shared mutable default would leak one call's arguments into the
    next, which is invisible until two tools are called in one turn."""
    a = ToolCall(id="1", name="x")
    b = ToolCall(id="2", name="y")
    a.args["leaked"] = True
    assert b.args == {}


def test_tool_call_keeps_arguments_as_a_dict():
    """The LLM adapter is responsible for the JSON-string conversion at the
    boundary; inside the domain these stay structured."""
    call = ToolCall(id="1", name="db_execute", args={"query": "SELECT 1", "params": []})
    assert call.args["params"] == []


# --------------------------------------------------------------------------
# PaginationState
# --------------------------------------------------------------------------


def test_pagination_starts_at_zero_pages():
    state = PaginationState(has_more=True)
    assert state.pages_fetched == 0
    assert state.next_cursor is None


def test_pagination_carries_the_cursor_it_was_given():
    state = PaginationState(has_more=True, next_cursor="abc", pages_fetched=2)
    assert (state.has_more, state.next_cursor, state.pages_fetched) == (True, "abc", 2)


# --------------------------------------------------------------------------
# AgentTurnResult
# --------------------------------------------------------------------------


def test_an_empty_result_is_valid_and_unshared():
    a = AgentTurnResult()
    b = AgentTurnResult()
    a.messages.append(ChatMessage(role=Role.USER, content="x"))
    a.pagination["catalog"] = PaginationState(has_more=False)

    assert b.messages == []
    assert b.pagination == {}


def test_pagination_is_keyed_by_tool_so_limits_are_tracked_per_tool():
    """One tool exhausting its page budget must not stop another."""
    result = AgentTurnResult(pagination={
        "catalog": PaginationState(has_more=False, pages_fetched=5),
        "circulation": PaginationState(has_more=True, pages_fetched=1),
    })
    assert result.pagination["circulation"].has_more
    assert not result.pagination["catalog"].has_more
