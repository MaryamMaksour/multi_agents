from .domain.entities.chat_message import ChatMessage, ToolCall

from typing import Protocol, List

class LLMPort(Protocol):

    async def achat(self, message: List[ChatMessage]) -> None:
        """Send a message to the LLM."""
        pass

