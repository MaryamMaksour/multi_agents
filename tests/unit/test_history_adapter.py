"""What history records, and therefore what memory can ever offer back.

`get_memory` filtered on `valid` and `reason`, and nothing wrote them. Every
row had valid NULL, so `valid = true` matched nothing, `valid = false` matched
nothing, and the method returned an empty list for every question - after
paying for an embedding call to build a vector it then compared against
nothing. The docstring in RunAgentTurn describes the intended behaviour as
though it were implemented.

`judge` is that implementation, and these are its cases. A turn is worth
learning from when it produced a real answer and nothing failed on the way;
`reason` is the deduplication key that decides whether three examples show
three approaches or the same one three times.
"""

from __future__ import annotations

from adapters.outbound.history.postgres_history_adapter import judge
from domain.entities.chat_message import ChatMessage, Role, ToolCall

def trace(*messages):
    return list(messages)


def test_a_clean_turn_is_valid_and_tagged_by_its_approach():
    """`reason` is the deduplication key that get_memory takes DISTINCT ON,
    so making it the tool sequence means three examples of three different
    approaches rather than three of the same one."""
    valid, reason = judge(trace(
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="1", name="get_table_schema")]),
        ChatMessage(role=Role.TOOL, tool_call_id="1", content='{"books": {}}'),
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="2", name="db_execute")]),
        ChatMessage(role=Role.TOOL, tool_call_id="2", content='{"rows": [{"n": 12}]}'),
        ChatMessage(role=Role.ASSISTANT, content="There are 12."),
    ))
    assert valid is True
    assert reason == "get_table_schema>db_execute"


def test_a_turn_whose_tool_failed_is_not_a_worked_example():
    """The loop hands tool failures to the model as results rather than
    raising, so the trace is the only place they appear at all."""
    valid, reason = judge(trace(
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="1", name="db_execute")]),
        ChatMessage(role=Role.TOOL, tool_call_id="1",
                    content='{"error": "permission denied for table loans"}'),
        ChatMessage(role=Role.ASSISTANT, content="I cannot see loans."),
    ))
    assert valid is False
    assert "permission denied" in reason


def test_a_turn_that_never_answered_is_not_an_example_either():
    """The loop budget stopping a turn leaves a trace that goes nowhere.
    Showing the model a pattern that does not reach an answer is worse than
    showing it nothing."""
    valid, reason = judge(trace(
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="1", name="db_execute")]),
        ChatMessage(role=Role.TOOL, tool_call_id="1", content='{"rows": []}'),
    ))
    assert valid is False
    assert reason == "no answer"


def test_zero_rows_is_a_valid_turn():
    """An empty result is an answer. "There are none" is often the true one,
    and treating it as a failure would drop the examples that teach a model
    to say so."""
    valid, _ = judge(trace(
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="1", name="db_execute")]),
        ChatMessage(role=Role.TOOL, tool_call_id="1",
                    content='{"rows": [], "row_count": 0}'),
        ChatMessage(role=Role.ASSISTANT, content="There are none."),
    ))
    assert valid is True


def test_a_trace_stored_as_plain_dicts_is_judged_the_same():
    """History reads rows back as JSON, so judge has to accept both shapes -
    entities on the way in, dicts on the way out."""
    valid, reason = judge([
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "1", "name": "db_execute", "args": {}}]},
        {"role": "tool", "tool_call_id": "1", "content": '{"rows": [{"n": 1}]}'},
        {"role": "assistant", "content": "One."},
    ])
    assert (valid, reason) == (True, "db_execute")


def test_something_that_is_not_a_trace_is_not_valid():
    assert judge(None) == (False, "no trace")
    assert judge("a string") == (False, "no trace")


def test_a_turn_the_budget_stopped_is_not_a_worked_example():
    """The loop ends a budget-exhausted turn with an assistant message saying
    so - which has the same shape as a real answer, text and all.

    Without recognising it, these were stored `valid=true` and handed back by
    get_memory as worked examples: the model would be shown, as a model
    answer, a message reporting that no answer was found.
    """
    from domain.entities.agent_turn import GAVE_UP_PREFIX

    valid, reason = judge([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="1", name="db_execute")]),
        ChatMessage(role=Role.TOOL, tool_call_id="1", content='{"rows": []}'),
        ChatMessage(role=Role.ASSISTANT,
                    content=f"{GAVE_UP_PREFIX} within 12 steps, so I do not "
                            "have a reliable answer."),
    ])
    assert valid is False
    assert reason == "gave up"


def test_an_answer_that_merely_sounds_uncertain_is_still_valid():
    """The check is the loop's own sentence, not a mood. A model hedging in
    its own words has still answered."""
    valid, _ = judge([
        ChatMessage(role=Role.ASSISTANT, content="",
                    tool_calls=[ToolCall(id="1", name="db_execute")]),
        ChatMessage(role=Role.TOOL, tool_call_id="1", content='{"rows": [{"n": 12}]}'),
        ChatMessage(role=Role.ASSISTANT,
                    content="I could not find an exact match, but there are 12."),
    ])
    assert valid is True
