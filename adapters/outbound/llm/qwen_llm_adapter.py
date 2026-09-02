import json
import logging
from typing import Any

from openai import AsyncOpenAI

from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.exceptions import LLMRequestError
from libs.agent_core.logging_setup import Timer, log_event

logger = logging.getLogger(__name__)


def _usage_fields(response: Any) -> dict:
    """Token counts, and the one number that says whether caching is working.

    Providers report a cached prefix under `prompt_tokens_details`, and this
    system resends a large fixed prefix on every call in the loop - the tool
    schemas alone are about a thousand tokens, and one question is roughly
    seven calls. Whether those are being paid for each time is a number, and
    now it is in the logs of every run rather than in a billing console.

    Defensive throughout: usage is optional in the OpenAI schema, a local
    vLLM may omit the details block, and a missing token count must not be
    the thing that turns a successful answer into an exception.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    fields = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    if cached is not None:
        fields["cached_tokens"] = cached
        prompt = fields.get("prompt_tokens") or 0
        if prompt:
            fields["cache_hit_pct"] = round(100 * cached / prompt, 1)
    return {k: v for k, v in fields.items() if v is not None}


class QwenLLMAdapter:
    def __init__(self, client: AsyncOpenAI, model: str, temperature: float, max_tokens: int,
                 tools: list[dict] | None = None):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = tools  # bound 

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
            tool_calls = []
            for tc in message.tool_calls:
                # A model can emit arguments that are not valid JSON, and it
                # does so under exactly the conditions that matter: long
                # argument lists, truncation at max_tokens. Left to raise,
                # json.loads takes down the turn with a ValueError several
                # frames from anything that names the tool. Turned into an
                # empty argument set, the loop calls the tool, the tool
                # rejects it, and the model gets a chance to correct itself -
                # which is the behaviour every other tool error already has.
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (ValueError, TypeError):
                    log_event(logger, "llm.bad_tool_arguments",
                              level=logging.WARNING, tool=tc.function.name,
                              raw=(tc.function.arguments or "")[:500])
                    args = {}
                if not isinstance(args, dict):
                    log_event(logger, "llm.bad_tool_arguments",
                              level=logging.WARNING, tool=tc.function.name,
                              reason="not an object", raw=str(args)[:500])
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

        # Reasoning models return their chain of thought beside the answer
        # rather than inside it. getattr because most models do not have it
        # and the field is simply absent.
        reasoning = getattr(message, "reasoning_content", None)

        return ChatMessage(role=Role.ASSISTANT, content=message.content,
                           tool_calls=tool_calls, reasoning=reasoning)

    async def achat(self, messages: list[ChatMessage]) -> ChatMessage:
        """Send the full conversation history to the LLM and return its reply.

        Raises:
            LLMRequestError: if the LLM call fails.
        """
        provider_messages = [self._to_provider_message(m) for m in messages]

        log_event(logger, "llm.request", level=logging.DEBUG,
                  model=self._model, messages=len(provider_messages),
                  tools=len(self._tools or []),
                  chars=sum(len(str(m.get("content") or "")) for m in provider_messages))

        try:
            with Timer() as timer:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=provider_messages,
                    tools=self._tools,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
        except Exception as e:
            # Two records on purpose. The event line is what a log search
            # finds and counts; the exc_info line carries the frame, and the
            # formatter strips the key out of whichever of the two the client
            # put it in.
            log_event(logger, "llm.error", level=logging.ERROR,
                      model=self._model, error=type(e).__name__)
            logger.error("the model call failed", exc_info=True)
            raise LLMRequestError(f"Error {e} while calling Qwen") from e

        choice = response.choices[0]
        reply = self._from_provider_response(choice.message)

        # finish_reason is the field that explains a truncated answer, and
        # "length" is the one worth noticing: the model was cut off at
        # max_tokens, so a half-written tool call or a missing conclusion is
        # a budget problem rather than a model that got it wrong.
        finish = getattr(choice, "finish_reason", None)
        log_event(
            logger, "llm.response",
            level=logging.WARNING if finish == "length" else logging.INFO,
            model=self._model, ms=timer.ms, finish=finish,
            tool_calls=[c.name for c in reply.tool_calls or []],
            content_chars=len(reply.content or ""),
            reasoning_chars=len(reply.reasoning or ""),
            **_usage_fields(response),
        )

        # The model's own account of what it was doing, at DEBUG so it is
        # there when a wrong answer needs explaining and out of the way when
        # it does not. This is the line that says whether "129" came from
        # misreading the question or from misreading the schema.
        if reply.reasoning:
            log_event(logger, "llm.reasoning", level=logging.DEBUG,
                      text=reply.reasoning)

        return reply