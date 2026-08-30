"""QwenLLMAdapter - the translation to and from the provider wire format.

The interesting asymmetry is tool-call arguments: the domain holds them as a
dict, the wire carries them as a JSON *string*, in both directions. Getting
that wrong is not a crash - it is a provider 400, or arguments that arrive as
the string "{'query': ...}" and fail to parse later, well away from here.

The client is faked. Nothing in this file makes a network call, so these run
in milliseconds and need no API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from adapters.outbound.llm.qwen_llm_adapter import QwenLLMAdapter
from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.exceptions import LLMRequestError

to_wire = QwenLLMAdapter._to_provider_message
from_wire = QwenLLMAdapter._from_provider_response


def provider_message(content=None, tool_calls=None):
    """Shape the OpenAI SDK returns: attribute access, arguments as a string."""
    calls = [
        SimpleNamespace(
            id=c["id"],
            function=SimpleNamespace(name=c["name"], arguments=json.dumps(c["args"])),
        )
        for c in (tool_calls or [])
    ]
    return SimpleNamespace(content=content, tool_calls=calls or None)


class FakeCompletions:
    def __init__(self, message=None, error=None):
        self.message = message if message is not None else provider_message("hello")
        self.error = error
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])


class FakeClient:
    def __init__(self, message=None, error=None):
        self.completions = FakeCompletions(message, error)
        self.chat = SimpleNamespace(completions=self.completions)


def adapter(client=None, **kw):
    params = dict(model="qwen3-14b", temperature=0.1, max_tokens=1000, tools=None)
    params.update(kw)
    return QwenLLMAdapter(client=client or FakeClient(), **params)


# --------------------------------------------------------------------------
# domain -> wire
# --------------------------------------------------------------------------


@pytest.mark.parametrize("role, expected", [
    (Role.SYSTEM, "system"),
    (Role.USER, "user"),
    (Role.ASSISTANT, "assistant"),
])
def test_roles_are_sent_as_their_wire_names(role, expected):
    assert to_wire(ChatMessage(role=role, content="x"))["role"] == expected


def test_a_tool_result_carries_its_call_id():
    """The provider rejects a tool message that cannot be paired with a call."""
    wire = to_wire(ChatMessage(role=Role.TOOL, content="{}", tool_call_id="call_1"))
    assert wire == {"role": "tool", "tool_call_id": "call_1", "content": "{}"}


def test_tool_call_arguments_are_sent_as_a_json_string():
    """Not a dict. The provider specifies a string here and rejects an object."""
    wire = to_wire(ChatMessage(
        role=Role.ASSISTANT, content="",
        tool_calls=[ToolCall(id="c1", name="db_execute", args={"query": "SELECT 1", "params": []})],
    ))
    arguments = wire["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"query": "SELECT 1", "params": []}


def test_a_tool_call_declares_its_type():
    wire = to_wire(ChatMessage(role=Role.ASSISTANT, content="",
                               tool_calls=[ToolCall(id="c1", name="t", args={})]))
    assert wire["tool_calls"][0]["type"] == "function"
    assert wire["tool_calls"][0]["id"] == "c1"


def test_an_assistant_message_without_tool_calls_omits_the_key():
    """Sending tool_calls: [] is not the same as omitting it; some providers
    treat the empty list as a malformed turn."""
    assert "tool_calls" not in to_wire(ChatMessage(role=Role.ASSISTANT, content="answer"))


def test_none_content_is_sent_as_empty_string():
    """A tool-calling turn usually has no text, and null content is refused."""
    assert to_wire(ChatMessage(role=Role.ASSISTANT, content=None))["content"] == ""


def test_arabic_content_is_preserved():
    text = "كم كتابًا لدينا؟"
    assert to_wire(ChatMessage(role=Role.USER, content=text))["content"] == text


def test_an_unknown_role_raises():
    class FakeRole:
        pass

    with pytest.raises(ValueError):
        to_wire(ChatMessage(role=FakeRole(), content="x"))


# --------------------------------------------------------------------------
# wire -> domain
# --------------------------------------------------------------------------


def test_a_plain_reply_becomes_an_assistant_message():
    msg = from_wire(provider_message("the answer"))
    assert msg.role is Role.ASSISTANT
    assert msg.content == "the answer"
    assert msg.tool_calls is None


def test_tool_call_arguments_are_parsed_back_into_a_dict():
    """The domain works with structured args; leaving them as a string would
    push json.loads into every tool adapter."""
    msg = from_wire(provider_message("", [
        {"id": "c1", "name": "db_execute", "args": {"query": "SELECT 1"}},
    ]))
    assert isinstance(msg.tool_calls[0], ToolCall)
    assert msg.tool_calls[0].args == {"query": "SELECT 1"}


def test_several_tool_calls_are_all_returned():
    msg = from_wire(provider_message("", [
        {"id": "c1", "name": "get_table_schema", "args": {"tables": ["books"]}},
        {"id": "c2", "name": "get_filter", "args": {"columns": ["genre"], "table_name": "books"}},
    ]))
    assert [c.name for c in msg.tool_calls] == ["get_table_schema", "get_filter"]


def test_no_tool_calls_becomes_none():
    assert from_wire(provider_message("answer")).tool_calls is None


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_a_tool_call_survives_a_round_trip():
    original = ChatMessage(
        role=Role.ASSISTANT, content="",
        tool_calls=[ToolCall(id="c1", name="db_execute",
                             args={"query": "SELECT 1", "params": [10, 0]})],
    )
    wire = to_wire(original)
    rebuilt = from_wire(provider_message(
        wire["content"],
        [{"id": tc["id"], "name": tc["function"]["name"],
          "args": json.loads(tc["function"]["arguments"])} for tc in wire["tool_calls"]],
    ))

    assert rebuilt.tool_calls[0].id == "c1"
    assert rebuilt.tool_calls[0].args == {"query": "SELECT 1", "params": [10, 0]}


# --------------------------------------------------------------------------
# achat
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_achat_sends_the_configured_model_and_sampling():
    client = FakeClient()
    await adapter(client, model="qwen3-32b", temperature=0.7, max_tokens=99).achat(
        [ChatMessage(role=Role.USER, content="hi")]
    )

    sent = client.completions.kwargs
    assert sent["model"] == "qwen3-32b"
    assert sent["temperature"] == 0.7
    assert sent["max_tokens"] == 99


@pytest.mark.asyncio
async def test_achat_sends_the_bound_tool_schemas():
    """The tools are fixed at construction, so every call advertises the same
    set - the model cannot be offered a tool the adapter cannot dispatch."""
    tools = [{"type": "function", "function": {"name": "db_execute",
                                               "description": "...", "parameters": {}}}]
    client = FakeClient()
    await adapter(client, tools=tools).achat([ChatMessage(role=Role.USER, content="hi")])

    assert client.completions.kwargs["tools"] == tools


@pytest.mark.asyncio
async def test_achat_sends_the_whole_conversation_in_order():
    client = FakeClient()
    await adapter(client).achat([
        ChatMessage(role=Role.SYSTEM, content="prompt"),
        ChatMessage(role=Role.USER, content="one"),
        ChatMessage(role=Role.ASSISTANT, content="two"),
        ChatMessage(role=Role.USER, content="three"),
    ])

    assert [m["role"] for m in client.completions.kwargs["messages"]] == \
        ["system", "user", "assistant", "user"]


@pytest.mark.asyncio
async def test_achat_returns_a_domain_message():
    client = FakeClient(provider_message("the answer"))
    reply = await adapter(client).achat([ChatMessage(role=Role.USER, content="q")])

    assert isinstance(reply, ChatMessage)
    assert reply.content == "the answer"


@pytest.mark.asyncio
async def test_achat_returns_one_message_not_a_list():
    """The loop adapter appends the result directly; a list here would nest."""
    reply = await adapter().achat([ChatMessage(role=Role.USER, content="q")])
    assert not isinstance(reply, list)


@pytest.mark.asyncio
async def test_a_provider_failure_becomes_a_domain_error():
    """Callers catch LLMRequestError. A raw openai exception would escape
    every handler written against the domain's error type."""
    client = FakeClient(error=RuntimeError("upstream 503"))
    with pytest.raises(LLMRequestError, match="503"):
        await adapter(client).achat([ChatMessage(role=Role.USER, content="q")])


@pytest.mark.asyncio
async def test_a_malformed_tool_argument_string_is_also_a_domain_error():
    """Models do emit invalid JSON. It must not surface as a bare
    JSONDecodeError from inside the adapter."""
    bad = SimpleNamespace(
        content="",
        tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
            name="db_execute", arguments="{not json"))],
    )
    with pytest.raises(LLMRequestError):
        await adapter(FakeClient(bad)).achat([ChatMessage(role=Role.USER, content="q")])
