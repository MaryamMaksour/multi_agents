from typing import Protocol

from domain.entities.chat_message import ChatMessage
from domain.entities.agent_turn import AgentTurnResult


class AgentLoopPort(Protocol):
    async def run(self, messages: list[ChatMessage], turn_id: str | None = None) -> AgentTurnResult:
        """Run the think -> call tool -> repeat loop until a final answer.

        `messages` is the full context (system + history + user turn).
        `turn_id` is handed to every tool call made during the loop, so a
        delegated question can be traced back to the turn that asked it.
        The returned AgentTurnResult.messages holds ONLY the messages
        generated during this loop (not the input) - assistant replies
        and tool results, in order.

        Raises:
            LLMRequestError: if the LLM call fails.
        """
        ...