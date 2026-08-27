from typing import Protocol

from domain.entities.chat_message import ChatMessage
from domain.entities.agent_turn import AgentTurnResult


class AgentLoopPort(Protocol):
    async def run(self, messages: list[ChatMessage]) -> AgentTurnResult:
        """Run the think -> call tool -> repeat loop until a final answer.

        `messages` is the full context (system + history + user turn).
        The returned AgentTurnResult.messages holds ONLY the messages
        generated during this loop (not the input) - assistant replies
        and tool results, in order.

        Raises:
            LLMRequestError: if the LLM call fails.
        """
        ...