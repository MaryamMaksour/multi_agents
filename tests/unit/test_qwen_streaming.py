"""Reassembling a streamed reply, which is the part no live test covers here.

Thinking mode has to stream: DashScope refuses `enable_thinking` on a
non-streaming request. So the pieces arrive as deltas and something has to put
them back together, and that something cannot be checked against the real
endpoint from this environment - the network policy blocks it outright, with
or without a key.

Which makes these tests the whole verification, so they are written against
what the wire actually does rather than what would be convenient:

    a tool call's id and name arrive in the first fragment, and its arguments
    a few characters at a time across many more, so the JSON is invalid until
    the last one lands;

    `index` is what identifies a call, not the id, which most fragments omit,
    and two concurrent calls interleave their fragments;

    the usage chunk carries no `choices` at all, so anything reading
    chunk.choices[0] unconditionally raises on the last chunk of every call.

The reassembled object deliberately wears the shape of a non-streamed
response, so _from_provider_response and _usage_fields keep working unchanged.
That is the property most worth pinning: the loop must never learn that two
kinds of model call exist.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters.outbound.llm.qwen_llm_adapter import QwenLLMAdapter
from domain.entities.chat_message import ChatMessage, Role


def delta(content=None, reasoning=None, tool_calls=None, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, reasoning_content=reasoning,
                              tool_calls=tool_calls),
        finish_reason=finish,
    )], usage=None)


def call_fragment(index, id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index, id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def usage_chunk(prompt=100, completion=20, cached=80):
    """No `choices` at all - the shape that broke the naive reader."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    ))


class FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk
        return gen()


class FakeStreamingClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.kwargs = None
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self._chunks)


def adapter(client, **kw):
    return QwenLLMAdapter(client=client, model="qwen-plus", temperature=0.1,
                          max_tokens=1000, enable_thinking=True, **kw)


async def reply(chunks, **kw):
    client = FakeStreamingClient(chunks)
    result = await adapter(client, **kw).achat([ChatMessage(role=Role.USER, content="q")])
    return result, client


# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_content_deltas_are_joined_in_order():
    result, _ = await reply([
        delta(content="There "), delta(content="are "), delta(content="12."),
        delta(finish="stop"),
    ])
    assert result.content == "There are 12."


@pytest.mark.asyncio
async def test_reasoning_deltas_are_joined_separately_from_the_answer():
    """The two must not be concatenated: the reasoning is recorded and never
    resent, and merging it into content would send it back on the next call
    and show it to the user as the answer."""
    result, _ = await reply([
        delta(reasoning="The question says novels, "),
        delta(reasoning="so genre must be filtered."),
        delta(content="There are 12."),
        delta(finish="stop"),
    ])
    assert result.reasoning == "The question says novels, so genre must be filtered."
    assert result.content == "There are 12."


@pytest.mark.asyncio
async def test_tool_arguments_split_across_fragments_parse_once_joined():
    """Any single fragment is invalid JSON. Parsing before the last one lands
    is the mistake this reassembly exists to avoid."""
    result, _ = await reply([
        delta(tool_calls=[call_fragment(0, id="c1", name="db_execute", arguments='{"que')]),
        delta(tool_calls=[call_fragment(0, arguments='ry": "SELECT 1", "par')]),
        delta(tool_calls=[call_fragment(0, arguments='ams": [10, 0]}')]),
        delta(finish="tool_calls"),
    ])

    call = result.tool_calls[0]
    assert call.name == "db_execute"
    assert call.args == {"query": "SELECT 1", "params": [10, 0]}


@pytest.mark.asyncio
async def test_two_interleaved_tool_calls_stay_separate():
    """`index` identifies the call, not the id - most fragments omit the id.
    Keying on anything else merges two calls into one unparseable string."""
    result, _ = await reply([
        delta(tool_calls=[call_fragment(0, id="c1", name="get_filter", arguments='{"a"')]),
        delta(tool_calls=[call_fragment(1, id="c2", name="get_table_schema", arguments='{"b"')]),
        delta(tool_calls=[call_fragment(0, arguments=': 1}')]),
        delta(tool_calls=[call_fragment(1, arguments=': 2}')]),
        delta(finish="tool_calls"),
    ])

    assert [c.name for c in result.tool_calls] == ["get_filter", "get_table_schema"]
    assert result.tool_calls[0].args == {"a": 1}
    assert result.tool_calls[1].args == {"b": 2}


@pytest.mark.asyncio
async def test_the_usage_chunk_has_no_choices_and_must_not_raise():
    """It arrives last on every call. Reading chunk.choices[0] before checking
    would turn every successful streamed answer into an IndexError."""
    result, _ = await reply([
        delta(content="There are 12."), delta(finish="stop"), usage_chunk(),
    ])
    assert result.content == "There are 12."


@pytest.mark.asyncio
async def test_a_stream_that_yields_nothing_is_not_a_crash():
    """An empty answer is a problem for the loop to handle - it appends a
    closing message - not an exception from inside the adapter."""
    result, _ = await reply([])
    assert result.content is None
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_no_reasoning_leaves_the_field_unset_rather_than_empty():
    """None, not "": the trace renderer shows the section only when there is
    something in it."""
    result, _ = await reply([delta(content="hi"), delta(finish="stop")])
    assert result.reasoning is None


@pytest.mark.asyncio
async def test_thinking_mode_asks_for_it_and_asks_for_usage():
    result, client = await reply([delta(content="hi"), delta(finish="stop")])
    assert client.kwargs["stream"] is True
    assert client.kwargs["extra_body"] == {"enable_thinking": True}
    # Without include_usage there is no usage chunk, and the token counts and
    # cache-hit percentage disappear from the logs in thinking mode only.
    assert client.kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_thinking_mode_is_off_unless_asked_for():
    """The default path must be the plain one: streaming changes the request
    shape, costs reasoning tokens, and not every model accepts the flag."""
    # A plain (non-streamed) response, because that is what the non-thinking
    # path expects back - handing it a stream would fail for the wrong reason.
    class PlainClient(FakeStreamingClient):
        async def _create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="hi", tool_calls=None),
                finish_reason="stop")], usage=None)

    client = PlainClient([])
    plain = QwenLLMAdapter(client=client, model="qwen-plus", temperature=0.1,
                           max_tokens=1000)
    await plain.achat([ChatMessage(role=Role.USER, content="q")])
    assert "stream" not in client.kwargs
    assert "extra_body" not in client.kwargs


@pytest.mark.asyncio
async def test_a_malformed_argument_string_still_degrades_rather_than_raises():
    """Truncation at max_tokens leaves a half-written argument string, and
    streaming is where that is most likely - the stream simply stops."""
    result, _ = await reply([
        delta(tool_calls=[call_fragment(0, id="c1", name="db_execute", arguments='{"query": "SEL')]),
        delta(finish="length"),
    ])
    assert result.tool_calls[0].args == {}


@pytest.mark.asyncio
async def test_a_fragment_with_no_id_gets_a_stable_one():
    """Some providers omit the id entirely. The loop pairs tool results back
    to their calls by id, so an empty one silently breaks the pairing."""
    result, _ = await reply([
        delta(tool_calls=[call_fragment(0, name="db_execute", arguments="{}")]),
        delta(finish="tool_calls"),
    ])
    assert result.tool_calls[0].id
