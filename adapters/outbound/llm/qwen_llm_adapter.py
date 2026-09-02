import json
import logging
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.exceptions import LLMRequestError
from libs.agent_core.logging_setup import Timer, log_event

logger = logging.getLogger(__name__)


class _Assembled(SimpleNamespace):
    """A streamed response wearing the shape of a non-streamed one.

    SimpleNamespace rather than a dataclass because everything downstream
    reads these with getattr and never constructs one: _from_provider_response
    and _usage_fields were written against the provider's objects, and the
    point of reassembling here is that they do not have to learn a second
    shape.
    """


def _assembled(content, reasoning, calls, finish, usage) -> Any:
    tool_calls = [
        SimpleNamespace(
            id=slot["id"] or f"call_{index}",
            type="function",
            function=SimpleNamespace(
                name=slot["name"] or "",
                # Joined at the end: each fragment is a few characters, so any
                # single one is invalid JSON on its own.
                arguments="".join(slot["arguments"]),
            ),
        )
        for index, slot in sorted(calls.items())
    ]

    message = _Assembled(
        role="assistant",
        content="".join(content) or None,
        # None rather than "" when the model returned no reasoning, so the
        # trace renderer shows the section only when there is something in it.
        reasoning_content="".join(reasoning) or None,
        tool_calls=tool_calls or None,
    )
    return _Assembled(
        choices=[_Assembled(message=message, finish_reason=finish)],
        usage=usage,
    )


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
                 tools: list[dict] | None = None, enable_thinking: bool = False,
                 extra_body: dict | None = None):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tools = tools  # bound
        # Thinking mode changes the request path, not just a parameter:
        # DashScope refuses enable_thinking on a non-streaming call, so this
        # decides between two code paths below.
        self._enable_thinking = enable_thinking
        # Whatever else this deployment's provider wants. Sent on both paths,
        # so a model that needs a field to answer at all needs it whether or
        # not thinking is on.
        self._extra_body = dict(extra_body or {})

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

    async def _thinking_call(self, provider_messages: list[dict]) -> Any:
        """Ask the model to think first, and reassemble the stream into one
        response the rest of this adapter can treat like any other.

        DashScope refuses `enable_thinking` on a non-streaming request, so
        thinking mode has to stream - which means the pieces arrive as deltas
        and something has to put them back together. That something is here
        rather than in the loop, so the loop never learns that two kinds of
        model call exist.

        Tool calls are the fiddly part. They arrive across many chunks: the
        first carries the id and the function name, and the arguments come a
        few characters at a time, so the JSON is only valid once the last
        chunk has landed. `index` is what says which call a fragment belongs
        to - not the id, which most chunks omit - and a model emitting two
        calls interleaves their fragments.
        """
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=provider_messages,
            tools=self._tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
            # Where the provider's own extras go. Unknown to the OpenAI
            # schema, passed through as-is - which is also why a provider
            # that does not know it may reject the call, and why this is off
            # by default.
            # The deployment's fields first, then the one this path is for -
            # so QWEN_ENABLE_THINKING remains the switch that decides thinking
            # mode, and QWEN_EXTRA_BODY cannot quietly turn it off while the
            # code is streaming for it.
            extra_body={**self._extra_body, "enable_thinking": True},
            stream_options={"include_usage": True},
        )

        content, reasoning, finish, usage = [], [], None, None
        calls: dict[int, dict] = {}

        async for chunk in stream:
            # The usage chunk carries no choices, so this has to come first or
            # chunk.choices[0] raises on the last chunk of every call.
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            finish = getattr(choice, "finish_reason", None) or finish
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            if getattr(delta, "content", None):
                content.append(delta.content)
            if getattr(delta, "reasoning_content", None):
                reasoning.append(delta.reasoning_content)

            for fragment in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(
                    getattr(fragment, "index", 0),
                    {"id": None, "name": None, "arguments": []},
                )
                if getattr(fragment, "id", None):
                    slot["id"] = fragment.id
                function = getattr(fragment, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        slot["name"] = function.name
                    if getattr(function, "arguments", None):
                        slot["arguments"].append(function.arguments)

        return _assembled(content, reasoning, calls, finish, usage)

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
                if self._enable_thinking:
                    response = await self._thinking_call(provider_messages)
                else:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=provider_messages,
                        tools=self._tools,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                        # Omitted entirely when empty: an empty extra_body is
                        # not the same as no extra_body to every client, and
                        # the default path should send exactly what it sent
                        # before this existed.
                        **({"extra_body": self._extra_body}
                           if self._extra_body else {}),
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