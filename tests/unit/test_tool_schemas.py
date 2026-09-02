"""get_tool_schemas on both tool adapters.

A schema that does not match its handler fails at runtime as an
UnknownToolError or a TypeError, mid-conversation, rather than at startup.
These checks compare the declared schemas against the real dispatch table and
the real method signatures, so a rename on either side breaks the build
instead of a user's turn.

Both adapters are constructed with None for their collaborators: these
methods describe, they do not call, so there is nothing for a database or an
HTTP client to do here. If that ever stops being true, these tests fail
loudly - which is the right outcome.
"""

from __future__ import annotations

import inspect
import json
import sys
import types

import pytest

# httpx is only needed for the type hint on the delegate adapter's client.
sys.modules.setdefault("httpx", types.SimpleNamespace(AsyncClient=object))

from adapters.outbound.tools.http_delegate_tool_adapter import HttpDelegateToolAdapter  # noqa: E402
from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter  # noqa: E402


def sql_adapter() -> SqlToolAdapter:
    return SqlToolAdapter(
        db=None, embeddings=None, cache=None,
        allowed_tables=["books", "authors"],
        schema={}, filters={},
        dist_op="<=>", vector_ttl_seconds=900,
    )


def delegate_adapter(**kw) -> HttpDelegateToolAdapter:
    kwargs = dict(
        client=None,
        tool_urls={"catalog": "http://catalog/chat", "circulation": "http://circ/chat"},
        tool_descriptions={"catalog": "Books and authors.", "circulation": "Loans and members."},
    )
    kwargs.update(kw)
    return HttpDelegateToolAdapter(**kwargs)


def functions(schemas: list[dict]) -> dict[str, dict]:
    return {s["function"]["name"]: s["function"] for s in schemas}


# --------------------------------------------------------------------------
# shape, shared by both adapters
# --------------------------------------------------------------------------


@pytest.fixture(params=["sql", "delegate"])
def schemas(request):
    return sql_adapter().get_tool_schemas() if request.param == "sql" else delegate_adapter().get_tool_schemas()


def test_every_schema_is_a_function_tool(schemas):
    for s in schemas:
        assert s["type"] == "function"
        assert set(s["function"]) >= {"name", "description", "parameters"}


def test_every_schema_is_json_serialisable(schemas):
    """It is handed to the OpenAI client as-is; anything else fails at the
    request, which is the worst place to find out."""
    json.dumps(schemas)


def test_parameters_are_valid_json_schema_objects(schemas):
    for s in schemas:
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert isinstance(params["properties"], dict)
        for prop in params["properties"].values():
            assert "description" in prop, "an undescribed parameter is one the model will guess at"
        for required in params.get("required", []):
            assert required in params["properties"]


def test_tool_names_are_unique(schemas):
    names = [s["function"]["name"] for s in schemas]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# SqlToolAdapter - schemas must match the dispatcher and the methods
# --------------------------------------------------------------------------


def test_every_declared_sql_tool_is_dispatchable():
    """The direction that fails at runtime: a name the model is told about
    and the dispatcher does not know raises UnknownToolError mid-answer."""
    adapter = sql_adapter()
    declared = {s["function"]["name"] for s in adapter.get_tool_schemas()}
    assert declared <= set(adapter._handlers)


def test_every_dispatchable_sql_tool_is_declared():
    """The other direction: a handler the model is never told about is dead
    code, or a tool nobody decided to hide."""
    adapter = sql_adapter()
    declared = {s["function"]["name"] for s in adapter.get_tool_schemas()}
    assert set(adapter._handlers) == declared


def test_sql_declared_properties_are_accepted_by_their_handlers():
    """call_tool does handler(**args), so an undeclared-but-passed key or a
    declared-but-unaccepted one is a TypeError at call time."""
    adapter = sql_adapter()
    for name, fn in functions(adapter.get_tool_schemas()).items():
        accepted = set(inspect.signature(adapter._handlers[name]).parameters)
        declared = set(fn["parameters"]["properties"])
        assert declared <= accepted, f"{name}: {declared - accepted} not accepted by the handler"


def test_sql_required_matches_the_parameters_without_defaults():
    adapter = sql_adapter()
    for name, fn in functions(adapter.get_tool_schemas()).items():
        params = inspect.signature(adapter._handlers[name]).parameters
        no_default = {p for p, v in params.items() if v.default is inspect.Parameter.empty}
        assert set(fn["parameters"].get("required", [])) == no_default, (
            f"{name}: required set disagrees with the handler's signature"
        )


def test_the_misspelled_tool_name_is_preserved():
    """get_list_values is a typo, but call_tool dispatches on that string, so
    correcting it here alone would break every call. It is renamed in both
    places or in neither."""
    adapter = sql_adapter()
    assert "get_list_values" in {s["function"]["name"] for s in adapter.get_tool_schemas()}


def test_sql_descriptions_are_substantial():
    """The model picks tools by reading these, and for this adapter they are
    written in code, so their quality is the adapter's responsibility.

    Not asserted for the delegate adapter: there the text comes from the
    registry, and passing it through unchanged is the correct behaviour -
    checking its length would only be testing this file's own fixture."""
    for s in sql_adapter().get_tool_schemas():
        assert len(s["function"]["description"]) > 60


def test_db_execute_states_the_rules_it_enforces():
    """These are returned as {"error": ...} at runtime. Saying them up front
    turns a wasted round trip into a query that is right first time."""
    fn = functions(sql_adapter().get_tool_schemas())["db_execute"]
    text = fn["description"].upper()
    for rule in ("SELECT", "LIMIT", "OFFSET", "100"):
        assert rule in text, f"db_execute should mention {rule}"


# --------------------------------------------------------------------------
# HttpDelegateToolAdapter
# --------------------------------------------------------------------------


def test_one_tool_per_registered_sub_agent():
    """Registering an agent should be enough to make it callable - no second
    list to keep in step."""
    adapter = delegate_adapter()
    assert {s["function"]["name"] for s in adapter.get_tool_schemas()} == set(adapter._tool_urls)


def test_the_registered_description_is_what_the_model_reads():
    fn = functions(delegate_adapter().get_tool_schemas())["catalog"]
    assert fn["description"] == "Books and authors."


def test_correlation_ids_are_not_exposed_to_the_model():
    """call_tool reads session_id and turn_id from args, but the agent loop
    injects those - they are not choices for the model. Declaring them would
    put them in its context, which is the leak the old codebase had to strip
    back out afterwards."""
    for s in delegate_adapter().get_tool_schemas():
        props = s["function"]["parameters"]["properties"]
        assert "session_id" not in props
        assert "turn_id" not in props


def test_query_is_required_and_cursor_is_not():
    fn = functions(delegate_adapter().get_tool_schemas())["catalog"]
    assert fn["parameters"]["required"] == ["query"]
    assert "cursor" in fn["parameters"]["properties"]


def test_a_missing_description_raises_rather_than_defaulting():
    """A placeholder description does not fail loudly - it quietly makes an
    agent unroutable, which is far harder to notice."""
    adapter = delegate_adapter(tool_descriptions={"catalog": "Books."})
    with pytest.raises(ValueError, match="circulation"):
        adapter.get_tool_schemas()


def test_no_tools_registered_yields_no_schemas():
    adapter = delegate_adapter(tool_urls={}, tool_descriptions={})
    assert adapter.get_tool_schemas() == []
