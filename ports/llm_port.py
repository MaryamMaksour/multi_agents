from .domain.entities.chat_message import ChatMessage, ToolCall

from typing import Protocol, List

class LLMPort(Protocol):

    async def achat(self, message: List[ChatMessage]) -> None:
        """Send the full conversation history to the LLM and return its reply.

        Raises:
            LLMRequestError: if the LLM call fails.
        """
        ...

