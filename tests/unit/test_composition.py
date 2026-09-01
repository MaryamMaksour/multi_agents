"""The composition root - every wiring decision, none of the I/O.

This is the only module that knows both ports and adapters, so it is where a
wrong decision is invisible: nothing type-checks "the orchestrator got the
orchestrator prompt", and a sub-agent handed a conversation window would work
perfectly in every test that asks it one question.

So the assertions here are mostly about *which* of two plausible things was
chosen, and each one has a failure that is quiet rather than loud:

    a sub-agent with history          two callers' contexts mix, and the same
                                      question answers differently depending
                                      on what was asked before it
    tables from the registry          the model is told about tables the role
                                      may not read
    an orchestrator routing to itself a loop with a convincing first step

`open_runtime` is not tested here. It opens pools and clients and contains no
decisions, which is exactly the split it was written for.
"""

from __future__ import annotations

import os

import pytest

from domain.entities.provider_spec import AgentStatus, AgentType, ProviderSpec
from domain.entities.table_schema import ColumnSchema, TableSchema
from libs.agent_core.agent_startup import ReadyAgent
from libs.agent_core.composition import (
    Runtime,
    assemble_orchestrator,
    assemble_sub_agent,
    delegate_targets,
)
from libs.agent_core.prompts import ORCHESTRATOR_PROMPT, describe_agent
from libs.agent_core.schema_bootstrap import load_agent_schema

BOOKS = TableSchema(name="books", columns=(
    ColumnSchema("id", "integer", nullable=False),
    ColumnSchema("title", "text"),
))


def spec(**overrides) -> ProviderSpec:
    kwargs = dict(
        name="catalog",
        display_name="Catalogue",
        system_prompt="You answer questions about the catalogue.",
        description="Books, authors and publishers.",
        db_role="app_catalog",
        status=AgentStatus.ACTIVE,
        tables=["books"],
    )
    kwargs.update(overrides)
    return ProviderSpec(**kwargs)


class SchemaPortStub:
    async def list_tables(self):
        return ("books",)

    async def describe(self, tables):
        return {t: BOOKS for t in tables if t == "books"}

    async def distinct_count(self, table, column):
        return 5

    async def distinct_values(self, table, column, limit):
        return ("one", "two", "three")[:limit]


async def ready(**overrides) -> ReadyAgent:
    return ReadyAgent(
        spec=spec(**overrides),
        schema=await load_agent_schema(SchemaPortStub()),
    )


class RecordingLLMFactory:
    """Captures the tool schemas the LLM was bound to.

    The tools an agent has are decided here and nowhere else, so this is the
    only place that binding can be checked.
    """

    def __init__(self):
        self.tool_schemas = None

    def __call__(self, tool_schemas):
        self.tool_schemas = tool_schemas
        return object()


def parts(**over):
    base = dict(
        db=object(), embeddings=object(), cache=object(), history=object(),
        llm_factory=RecordingLLMFactory(),
    )
    base.update(over)
    return base


# --------------------------------------------------------------------------
# a sub-agent
# --------------------------------------------------------------------------


async def test_a_sub_agent_gets_the_prompt_its_deployment_wrote():
    """The untrusted one, verbatim. Safe because the GRANT is the boundary."""
    turn = assemble_sub_agent(await ready(), **parts())
    assert "You answer questions about the catalogue." in turn.system_prompt


async def test_a_sub_agent_is_also_told_how_to_use_its_tools():
    """Driving these tools is system behaviour, not something each person
    registering an agent should have to write out and keep in step."""
    turn = assemble_sub_agent(await ready(), **parts())
    assert "Call the schema tool first" in turn.system_prompt


async def test_the_shared_half_comes_first():
    """Every sub-agent gets the same tool schemas and the same method, so
    those bytes are identical across agents - a provider caching by prefix
    can reuse them between agents, not only between calls to one."""
    turn = assemble_sub_agent(await ready(), **parts())
    method_at = turn.system_prompt.index("How to work")
    deployment_at = turn.system_prompt.index("You answer questions")
    assert method_at < deployment_at


async def test_two_agents_share_the_same_leading_bytes():
    a = assemble_sub_agent(await ready(), **parts())
    b = assemble_sub_agent(
        await ready(name="circulation", db_role="app_circulation",
                    description="Loans.", system_prompt="You answer about loans."),
        **parts(),
    )
    shared = os.path.commonprefix([a.system_prompt, b.system_prompt])
    assert "aggregate in SQL" in shared


async def test_the_method_does_not_name_a_real_table():
    """The examples use placeholders on purpose. A worked example naming a
    table that does not exist in this deployment is an invitation to the
    exact failure it is meant to prevent - one model invented two tables and
    then said the question could not be answered."""
    from libs.agent_core.prompts import SUB_AGENT_METHOD

    for name in ("books", "authors", "publishers", "loans", "members"):
        assert name not in SUB_AGENT_METHOD.lower()


async def test_a_sub_agent_keeps_no_conversation_history():
    """It is called with a self-contained question and must answer from that
    alone. A window would let two callers' contexts mix, and would make the
    same question answer differently depending on what came before it."""
    turn = assemble_sub_agent(await ready(), **parts())
    assert turn.use_conversation_history is False


async def test_a_sub_agents_tables_come_from_introspection():
    """Not from spec.tables. That list was compared against the GRANTs at
    startup and its job is finished - what the model is told it may read has
    to be what the database will let it read."""
    agent = await ready(tables=["books", "authors", "publishers"])
    turn = assemble_sub_agent(agent, **parts())

    assert turn.agent_loop._tools._allowed_tables == {"books"}


async def test_a_sub_agent_is_given_the_sql_tools():
    factory = RecordingLLMFactory()
    assemble_sub_agent(await ready(), **parts(llm_factory=factory))

    names = {s["function"]["name"] for s in factory.tool_schemas}
    assert {"db_execute", "get_table_schema", "get_filter"} <= names


async def test_a_sub_agent_is_not_given_a_delegate_tool():
    """Sub-agents do not call each other. The orchestrator is the only
    component that fans out, and an agent that could delegate would make the
    call graph a graph rather than a tree."""
    factory = RecordingLLMFactory()
    assemble_sub_agent(await ready(), **parts(llm_factory=factory))

    assert "circulation" not in {s["function"]["name"] for s in factory.tool_schemas}


async def test_the_classified_filters_reach_the_tool():
    turn = assemble_sub_agent(await ready(), **parts())
    answer = await turn.agent_loop._tools.call_tool(
        "get_filter", {"columns": ["title"], "table_name": "books"}
    )
    assert isinstance(answer["title"], str) and answer["title"]


async def test_the_distance_operator_reaches_the_tool():
    turn = assemble_sub_agent(await ready(), **parts(), dist_op="<->")
    assert turn.agent_loop._tools._dist_op == "<->"


# --------------------------------------------------------------------------
# which agents become tools
# --------------------------------------------------------------------------


def test_a_url_is_derived_from_the_agents_own_name():
    """Which is also its service name in a compose file or a Kubernetes
    Service, so the default is right for the ordinary deployment."""
    urls, _ = delegate_targets([spec()], "http://agent-{key}:8000/run")
    assert urls == {"catalog": "http://agent-catalog:8000/run"}


def test_an_explicit_endpoint_wins_over_the_template():
    urls, _ = delegate_targets(
        [spec(endpoint="https://catalog.internal/run")], "http://agent-{key}:8000/run"
    )
    assert urls["catalog"] == "https://catalog.internal/run"


def test_the_orchestrator_is_not_a_tool_of_itself():
    """A registry may describe one. Handing it to itself is a loop whose
    first step looks entirely reasonable."""
    urls, _ = delegate_targets([
        spec(),
        spec(name="main", db_role="app_main", description="Routes.",
             type=AgentType.ORCHESTRATOR),
    ])
    assert set(urls) == {"catalog"}


def test_the_routing_text_names_the_agent():
    """The model is choosing between named things, and naming them is most
    of that choice."""
    _, descriptions = delegate_targets([spec()])
    assert descriptions["catalog"].startswith("Catalogue.")
    assert "Books, authors and publishers." in descriptions["catalog"]


def test_the_display_name_falls_back_to_the_key():
    assert describe_agent(spec(display_name="")).startswith("catalog.")


# --------------------------------------------------------------------------
# the orchestrator
# --------------------------------------------------------------------------


def orchestrator(specs=None, **over):
    base = dict(
        http_client=object(), cache=object(), history=object(),
        llm_factory=RecordingLLMFactory(),
    )
    base.update(over)
    return assemble_orchestrator(specs or [spec()], **base)


def test_the_orchestrator_gets_the_system_prompt_not_a_registry_one():
    """Its prompt is part of how the system works, like the tool
    descriptions, so it is versioned with the code that depends on it."""
    assert orchestrator().system_prompt == ORCHESTRATOR_PROMPT


def test_the_orchestrator_keeps_a_conversation_window():
    """The opposite of a sub-agent, for the same reason. This is the end that
    talks to a person, so it is the one that has to know what "it" refers to -
    and resolving those references before delegating is its job precisely
    because it is the only component that can."""
    assert orchestrator().use_conversation_history is True


def test_one_tool_per_registered_sub_agent():
    factory = RecordingLLMFactory()
    orchestrator(
        [spec(), spec(name="circulation", db_role="app_circulation",
                      description="Loans and members.")],
        llm_factory=factory,
    )
    assert {s["function"]["name"] for s in factory.tool_schemas} == {
        "catalog", "circulation",
    }


def test_the_delegate_tool_does_not_expose_correlation_values_to_the_model():
    """session_id and turn_id are injected by the loop, not chosen. Declaring
    them would put them in the model's context, which is the problem the old
    codebase had to strip back out afterwards."""
    factory = RecordingLLMFactory()
    orchestrator(llm_factory=factory)

    properties = factory.tool_schemas[0]["function"]["parameters"]["properties"]
    assert set(properties) == {"query", "cursor"}


def test_an_orchestrator_with_nothing_to_route_to_is_refused():
    """It would accept questions and be unable to answer any of them, which
    reads as a broken model rather than an empty registry."""
    with pytest.raises(ValueError, match="No sub-agents"):
        orchestrator([spec(name="main", db_role="app_main", description="Routes.",
                           type=AgentType.ORCHESTRATOR)])


def test_the_orchestrator_has_no_sql_tools():
    """It has no database of its own. Everything factual it says has to come
    from an agent's reply, and a tool that could query directly would make
    that untrue without anybody noticing."""
    factory = RecordingLLMFactory()
    orchestrator(llm_factory=factory)
    assert "db_execute" not in {s["function"]["name"] for s in factory.tool_schemas}


# --------------------------------------------------------------------------
# shutdown
# --------------------------------------------------------------------------


async def test_resources_close_in_reverse_order():
    """Opened pool-then-client, closed client-then-pool. The order matters
    where one depends on the other."""
    closed = []

    async def closer(name):
        closed.append(name)

    runtime = Runtime(kind="sub_agent", turn=None, closers=[
        lambda: closer("pool"), lambda: closer("redis"), lambda: closer("http"),
    ])
    await runtime.aclose()
    assert closed == ["http", "redis", "pool"]


async def test_one_failing_close_does_not_strand_the_others():
    """Shutdown is the one path where continuing past an error is right:
    there is nothing left to protect, and a pool that fails to close must not
    leave the Redis connection open behind it."""
    closed = []

    async def ok(name):
        closed.append(name)

    async def boom():
        raise OSError("connection already gone")

    runtime = Runtime(kind="sub_agent", turn=None, closers=[
        lambda: ok("pool"), boom, lambda: ok("http"),
    ])

    with pytest.raises(RuntimeError, match="shutting down"):
        await runtime.aclose()
    assert closed == ["http", "pool"]
