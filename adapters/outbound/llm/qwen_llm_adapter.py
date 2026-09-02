import json
from typing import Any

from openai import AsyncOpenAI

from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.exceptions import LLMRequestError


class QwenLLMAdapter:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float, max_tokens: int,
                 tools: list[dict] | None = None, extra_body: dict[str, Any] | None = None):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = tools  # bound
        self._extra_body = extra_body

    @staticmethod
    def _to_provider_message(msg: ChatMessage) -> dict:
        if msg.role == Role.SYSTEM:
            return {"role": "system", "content": msg.content}

        if msg.role == Role.USER:
            return {"role": "user", "content": msg.content}

        if msg.role == Role.TOOL:
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }

        if msg.role == Role.ASSISTANT:
            provider_msg: dict = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                provider_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return provider_msg

        raise ValueError(f"Unsupported role: {msg.role}")

    @staticmethod
    def _from_provider_response(message: Any) -> ChatMessage:
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=json.loads(tc.function.arguments),
                )
                for tc in message.tool_calls
            ]
        return ChatMessage(role=Role.ASSISTANT, content=message.content, tool_calls=tool_calls)

    async def achat(self, messages: list[ChatMessage]) -> ChatMessage:
        """Send the full conversation history to the LLM and return its reply.

        Raises:
            LLMRequestError: if the LLM call fails.
        """
        try:
            provider_messages = [self._to_provider_message(m) for m in messages]
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=provider_messages,
                tools=self._tools,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                extra_body=self._extra_body,
            )
            return self._from_provider_response(response.choices[0].message)
        except Exception as e:
            raise LLMRequestError(f"Error {e} while calling Qwen") from e