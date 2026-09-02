from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: Role
    content: Any = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    # What the model thought before it answered, when the provider returns it.
    #
    # Qwen's reasoning models put their chain of thought in a separate
    # `reasoning_content` field rather than in `content`, and it was being
    # dropped on the floor - which is a shame, because "why did it answer 129"
    # is precisely the question the trace exists to answer, and the reasoning
    # names the wrong turn directly where the tool arguments only imply it.
    #
    # Recorded, never resent. _to_provider_message does not read this field:
    # sending a previous turn's reasoning back is tokens paid for advice the
    # model already took, and some providers reject the field outright.
    reasoning: Optional[str] = None


# The wire shape of a message, defined once.
#
# Two adapters serialise these - Redis for the conversation window, Postgres
# for the history row - and they have to agree, because one writes what the
# other may later read back through get_memory. They did not: Redis had its
# own conversion and the history adapter had none at all, so every successful
# turn failed to record its own answer with
#
#     Object of type ChatMessage is not JSON serializable
#
# Dicts rather than JSON on purpose. Which encoding goes on the wire is the
# adapter's business; what a message *is* belongs here, next to the entity
# whose shape it describes.


def to_plain(message: "ChatMessage") -> dict:
    """A ChatMessage as plain data, ready for any encoder."""
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "args": call.args}
            for call in message.tool_calls
        ] if message.tool_calls else None,
        "tool_call_id": message.tool_call_id,
        "name": message.name,
        "reasoning": message.reasoning,
    }


def from_plain(data: dict) -> "ChatMessage":
    """The inverse. Tolerant of missing optional keys, because it reads rows
    written by an older version of this function as often as by this one."""
    return ChatMessage(
        role=Role(data["role"]),
        content=data.get("content"),
        tool_calls=[
            ToolCall(**call) for call in data["tool_calls"]
        ] if data.get("tool_calls") else None,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        # .get, like every field here: rows written before this field existed
        # are read back by this same function, and a KeyError on an old row
        # would take out get_memory for the whole retention window.
        reasoning=data.get("reasoning"),
    )
