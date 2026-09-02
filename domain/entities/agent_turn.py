from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .chat_message import ChatMessage, ToolCall


# results of a single agent turn, 
# which may include multiple messages and tool calls

@dataclass
class PaginationState:
    has_more: bool
    next_cursor: Optional[str] = None
    pages_fetched: int = 0


@dataclass
class AgentTurnResult:
    messages: list[ChatMessage] = field(default_factory=list)
    pagination: dict[str, PaginationState] = field(default_factory=dict) # ["agent domain",PaginationState ]


@dataclass
class PipelineStep:
    content: Any
    tool_calls: Optional[ToolCall] = None


# How a turn says the loop gave up, and the one place that sentence is spelled.
#
# In the domain rather than in either adapter, because two adapters need it
# and neither may import the other: the agent loop writes it when the
# iteration budget stops a turn, and the history adapter reads it to tell a
# give-up apart from an answer. Putting it in the loop adapter made the
# history adapter import langgraph to look at a string.
#
# It matters because the message has the same shape as a real reply - an
# assistant message with text - so without something to recognise it, a turn
# that concluded nothing was stored valid and offered back by get_memory as a
# worked example.
GAVE_UP_PREFIX = "I could not finish this question"
