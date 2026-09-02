from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .chat_message import ChatMessage


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
