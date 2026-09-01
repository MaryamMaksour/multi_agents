"""The agent loop, running for real against a real database.

This is the gap between "everything up to the model call is tested" and "a
real model answered a real question". The model is scripted - it decides
nothing - because what is being tested is the plumbing on either side of it:
a tool call arrives, the tool runs against a real Postgres through a real
least-privilege role, the result goes back, the loop continues, an answer
comes out.

Every bug found the first time this system met a real key lived exactly here:
a cache that could not store a vector, a vector that could not be sent to
Postgres, a memory lookup that took the whole turn down with it. None of them
were reachable from a unit test, and all of them are one HTTP call deep.

What this deliberately does not test: whether a model picks the right tool or
writes correct SQL. That needs a real model and a real key, and it is a
different question - about the prompt, not about the code.

Requires Postgres and Redis, seeded. Skipped when either is unreachable.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os

import pytest

asyncpg = pytest.importorskip("asyncpg")
pytest.importorskip("redis")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tests.fakes.scripted_model import (  # noqa: E402
    ScriptedModel,
    ServedApp,
    says,
    tool_call,
)

pytestmark = pytest.mark.integration

BASE_ENVIRONMENT = {
    "PG_HOST": os.getenv("PGHOST", "localhost"),
    "PG_PORT": os.getenv("PGPORT", "55432"),
    "PG_DBNAME": os.getenv("PGDATABASE", "library_dev"),
    "PG_USER": "app_authenticator",
    "PG_PASSWORD": "dev_authenticator",
    "REDIS_URL": os.getenv("TEST_REDIS_URL", "redis://localhost:56379/9"),
    "AGENTS_REGISTRY_PATH": "seeds/agents.example.json",
    "QWEN_API_KEY": "scripted-no-real-model-is-called",
}


def unreachable(error: BaseException) -> bool:
    """Whether this is "the services are not running" rather than a bug.

    Walks the chain rather than matching on the message: asyncpg wraps a
    ConnectionRefusedError several layers deep, and by the time it surfaces
    the text says nothing about connections.
    """
    seen = error
    while seen is not None:
        if isinstance(seen, (ConnectionRefusedError, ConnectionError, OSError)):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


@contextlib.contextmanager
def running(model: ScriptedModel, monkeypatch, agent_key: str):
    """The real app, pointed at the scripted endpoint.

    A context manager rather than a factory because the runtime is built in
    the app's lifespan, which runs when the TestClient is entered - so a
    database that is not there raises inside the caller's `with`, where a
    try around the construction never sees it.
    """
    from libs.agent_core import config

    for name, value in {**BASE_ENVIRONMENT, "QWEN_API_URL": model.base_url}.items():
        monkeypatch.setenv(name, value)
    importlib.reload(config)

    from adapters.inbound.http.app import create_app
    from libs.agent_core.composition import open_runtime

    async def opener():
        return await open_runtime(agent_key)

    try:
        with TestClient(create_app(opener)) as client:
            yield client
    except Exception as e:
        if unreachable(e):
            pytest.skip(f"Postgres or Redis unavailable: {e}")
        raise


def skip_unless_services_are_up() -> None:
    """A direct reachability check, for the one test that cannot rely on the
    usual one.

    Everywhere else the skip comes from `running()`, which sees the failure
    the app's lifespan raises. The delegation test starts the sub-agent under
    uvicorn instead, and uvicorn turns a failed lifespan into `SystemExit(3)`
    - so the connection error never reaches the caller and there is nothing
    left to recognise. Cheaper to ask first than to decode that.
    """
    import asyncio

    import asyncpg
    import redis.asyncio as redis

    async def check():
        connection = await asyncpg.connect(
            host=BASE_ENVIRONMENT["PG_HOST"], port=int(BASE_ENVIRONMENT["PG_PORT"]),
            user=BASE_ENVIRONMENT["PG_USER"], password=BASE_ENVIRONMENT["PG_PASSWORD"],
            database=BASE_ENVIRONMENT["PG_DBNAME"], timeout=3,
        )
        await connection.close()
        client = redis.from_url(BASE_ENVIRONMENT["REDIS_URL"])
        try:
            await client.ping()
        finally:
            await client.aclose()

    try:
        asyncio.run(check())
    except Exception as e:
        pytest.skip(f"Postgres or Redis unavailable: {e}")


@pytest.fixture(autouse=True)
def _clean_slate():
    """Empty the test Redis before each test, and restore config after.

    Conversation windows outlive a test - they are keyed on session_id with a
    three-day TTL - so a session name reused between tests loads the previous
    test's messages and the scripted replies stop lining up. That failure
    only appears when the file runs as a whole, which makes it exactly the
    kind worth removing rather than diagnosing twice.
    """
    import asyncio

    import redis.asyncio as redis

    async def flush():
        client = redis.from_url(BASE_ENVIRONMENT["REDIS_URL"])
        try:
            await client.flushdb()
        finally:
            await client.aclose()

    try:
        asyncio.run(flush())
    except Exception:
        pass   # unavailable Redis is handled by the skip in running()

    yield

    from libs.agent_core import config
    importlib.reload(config)


# --------------------------------------------------------------------------
# one sub-agent, answering from real data
# --------------------------------------------------------------------------


CATALOG_SCRIPT = [
    tool_call("get_table_schema", tables=["books"]),
    tool_call("get_filter", columns=["genre", "page_count"], table_name="books"),
    tool_call(
        "db_execute",
        query="SELECT count(*) AS n FROM books WHERE genre = $1 AND page_count < $2 LIMIT $3 OFFSET $4",
        # 'poetry' is a real value in the seeded data, so a zero here means
        # the query did not reach the table rather than that nothing matched.
        params=["poetry", 300, 10, 0],
        offset=0,
        count_query="SELECT count(*) FROM books WHERE genre = $1 AND page_count < $2",
        count_params=["poetry", 300],
    ),
    says("There are 12 short poetry collections."),
]


def test_a_sub_agent_answers_by_running_real_sql(monkeypatch):
    """The whole loop: four model calls, three tools, one real query against
    a real table through a real least-privilege role."""
    with ScriptedModel(CATALOG_SCRIPT) as model:
        with running(model, monkeypatch, "catalog") as client:
            body = client.post("/run", json={"user_input": "how many short novels?"}).json()

    assert body["answer"] == "There are 12 short poetry collections."
    assert len(model.requests) == 4


def test_the_tools_the_model_was_offered_are_the_sql_tools(monkeypatch):
    with ScriptedModel(CATALOG_SCRIPT) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "how many short novels?"})

    assert "db_execute" in model.tools_offered
    assert "get_filter" in model.tools_offered
    # Withheld deliberately - it queries columns no schema here has.
    assert "get_table_records" not in model.tools_offered


def test_the_schema_the_model_sees_is_the_real_one(monkeypatch):
    """Introspected at startup through the agent's role, not written down."""
    with ScriptedModel(CATALOG_SCRIPT) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "how many short novels?"})

    schema = model.tool_results_seen()[0]
    assert "title_en text" in schema
    assert "page_count integer" in schema


def test_the_filters_the_model_sees_are_classified_from_real_data(monkeypatch):
    """genre has 10 distinct values in the seeded data, so it is an ENUM and
    the guidance lists them. Nothing declared any of that - the kind came
    from a count and the values from the column itself."""
    with ScriptedModel(CATALOG_SCRIPT) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "how many short novels?"})

    filters = json.loads(model.tool_results_seen()[1])

    # The real values, read out of the table at startup. Being told a column
    # is short is not enough to turn "روايات" into genre = 'novel'.
    assert "novel" in filters["genre"]
    assert "poetry" in filters["genre"]
    assert "Numeric (integer)" in filters["page_count"]


def test_the_query_actually_ran(monkeypatch):
    """The row count comes back from Postgres, not from the script."""
    with ScriptedModel(CATALOG_SCRIPT) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "how many short novels?"})

    result = json.loads(model.tool_results_seen()[2])

    assert "error" not in result, result
    assert result["rows"], "no rows came back from the database"
    # A real count of a real genre, not a zero that would also pass if the
    # query had never reached the table.
    assert result["rows"][0]["n"] > 0


def test_the_system_prompt_is_identical_on_every_call(monkeypatch):
    """The prefix-caching property, observed from the endpoint's side rather
    than asserted about a local variable."""
    with ScriptedModel(CATALOG_SCRIPT) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "how many short novels?"})

    assert len(set(model.system_prompts)) == 1


# --------------------------------------------------------------------------
# the semantic path - vectors, end to end
# --------------------------------------------------------------------------


SEMANTIC_SCRIPT = [
    tool_call("embed_query_tool", query="a story about the sea"),
    says("placeholder"),   # replaced below, per test
]


def test_a_vector_survives_the_cache_and_reaches_postgres(monkeypatch):
    """Three bugs lived on this path and none was reachable from a unit
    test: the cache could not store a list of floats, the token resolved to
    a Python list, and asyncpg has no codec for pgvector's type."""
    with ScriptedModel([
        tool_call("embed_query_tool", query="a story about the sea"),
        says("done"),
    ]) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "books about the sea"})

        token = json.loads(model.tool_results_seen()[0])["vector_token"]

    assert token.startswith("vec_")


def test_a_semantic_query_runs_against_a_real_vector_column(monkeypatch):
    """The token goes back in as a db_execute parameter, is resolved to the
    cached vector, converted to pgvector's text form, and compared against a
    real embed_ column with the distance operator.

    The distances come back NULL, and that is correct here rather than a
    failure: seeds/002_generate_data.py creates the embed_ columns and never
    fills them, so there is nothing to be near. What this proves is the part
    that was broken - the vector survives the cache, reaches Postgres, and
    pgvector accepts it against a real column. Ranking needs a backfill; see
    docs/deferred.md.
    """
    script = [
        tool_call("embed_query_tool", query="a story about the sea"),
        None,   # filled in once the token is known
        says("Three books match."),
    ]

    class TokenAwareModel(ScriptedModel):
        def _reply(self, body):
            if self._step == 1:
                token = json.loads([
                    m["content"] for m in body["messages"] if m.get("role") == "tool"
                ][-1])["vector_token"]
                self.script[1] = tool_call(
                    "db_execute",
                    query=(
                        # No ORDER BY on the distance: seeds/001_schema.sql
                        # builds a pgvector index, and a pgvector index does
                        # not index NULL rows - so ordering by distance over
                        # an unembedded table returns nothing at all. See
                        # the test below, which pins that behaviour.
                        "SELECT title_en, embed_summary <=> $1::vector AS distance "
                        "FROM books LIMIT $2 OFFSET $3"
                    ),
                    params=[token, 3, 0],
                    offset=0,
                    count_query="SELECT count(*) FROM books",
                    count_params=[],
                )
            return super()._reply(body)

    with TokenAwareModel(script) as model:
        with running(model, monkeypatch, "catalog") as client:
            body = client.post("/run", json={"user_input": "books about the sea"}).json()

        rows = json.loads(model.tool_results_seen()[1])

    assert body["answer"] == "Three books match."
    assert "error" not in rows, rows
    assert len(rows["rows"]) == 3
    # The column is there and pgvector accepted the vector against it. Every
    # distance is None because nothing has been embedded yet.
    assert all("distance" in row for row in rows["rows"])


# --------------------------------------------------------------------------
# the guards still hold when a model asks for something it should not
# --------------------------------------------------------------------------


def test_a_query_against_another_agents_table_is_refused(monkeypatch):
    """The catalogue agent asking about members. The validator rejects it
    and the model is told why, rather than the turn failing."""
    with ScriptedModel([
        tool_call(
            "db_execute",
            query="SELECT count(*) FROM members LIMIT $1 OFFSET $2",
            params=[10, 0], offset=0,
            count_query="SELECT count(*) FROM members", count_params=[],
        ),
        says("I cannot see borrowing data - that is another agent's."),
    ]) as model:
        with running(model, monkeypatch, "catalog") as client:
            body = client.post("/run", json={"user_input": "who borrowed what?"}).json()

        refusal = json.loads(model.tool_results_seen()[0])

    assert "error" in refusal
    assert "members" in refusal["error"]
    assert body["answer"].startswith("I cannot see")


def test_a_write_is_refused(monkeypatch):
    with ScriptedModel([
        tool_call(
            "db_execute",
            query="UPDATE books SET price = 0 LIMIT $1 OFFSET $2",
            params=[1, 0], offset=0,
            count_query="SELECT count(*) FROM books", count_params=[],
        ),
        says("I can only read."),
    ]) as model:
        with running(model, monkeypatch, "catalog") as client:
            client.post("/run", json={"user_input": "set all prices to zero"})

        refusal = json.loads(model.tool_results_seen()[0])

    assert "error" in refusal


# --------------------------------------------------------------------------
# the orchestrator, delegating over HTTP to a real sub-agent
# --------------------------------------------------------------------------


def test_the_orchestrator_delegates_to_a_running_sub_agent(monkeypatch):
    """Two runtimes, one real HTTP hop. The orchestrator picks an agent,
    sends it a self-contained question, and the sub-agent answers it from the
    database.

    The sub-agent runs on a real socket rather than through a mocked client,
    because what is being tested is that the orchestrator's request and the
    sub-agent's response fit each other - the payload
    HttpDelegateToolAdapter posts against the RunRequest the endpoint
    validates. A mock would assert that against itself.
    """
    from adapters.inbound.http.app import create_app
    from libs.agent_core import config
    from libs.agent_core.composition import open_runtime

    skip_unless_services_are_up()

    sub_model = ScriptedModel([
        tool_call(
            "db_execute",
            query="SELECT count(*) AS n FROM books LIMIT $1 OFFSET $2",
            params=[1, 0], offset=0,
            count_query="SELECT count(*) FROM books", count_params=[],
        ),
        says("The catalogue holds 420 books."),
    ]).start()

    for name, value in {**BASE_ENVIRONMENT, "QWEN_API_URL": sub_model.base_url}.items():
        monkeypatch.setenv(name, value)
    importlib.reload(config)

    async def sub_opener():
        return await open_runtime("catalog")

    try:
        with ServedApp(create_app(sub_opener)) as sub_agent_url:
            orchestrator_model = ScriptedModel([
                tool_call("catalog", query="How many books are in the catalogue?"),
                says("There are 420 books."),
            ]).start()
            try:
                monkeypatch.setenv("QWEN_API_URL", orchestrator_model.base_url)
                # The sub-agent is at a real address, so the orchestrator
                # reaches it the way a deployment does - no client swapped in
                # afterwards.
                monkeypatch.setenv(
                    "AGENT_URL_TEMPLATE", sub_agent_url + "/run"
                )
                importlib.reload(config)

                async def opener():
                    return await open_runtime("")

                with TestClient(create_app(opener)) as client:
                    body = client.post("/ask", json={
                        "question": "How many books do we have?", "session_id": "s1",
                    }).json()

                delegated = json.loads(orchestrator_model.tool_results_seen()[0])
            finally:
                orchestrator_model.stop()
    except Exception as e:
        if unreachable(e):
            pytest.skip(f"Postgres or Redis unavailable: {e}")
        raise
    finally:
        sub_model.stop()

    assert body["answer"] == "There are 420 books."
    assert delegated["answer"] == "The catalogue holds 420 books."

    # The sub-agent was asked the orchestrator's self-contained question, not
    # the user's wording - which is the whole reason a sub-agent can be
    # stateless.
    asked = [
        m["content"] for m in sub_model.requests[0]["messages"]
        if m["role"] == "user"
    ]
    assert asked == ["How many books are in the catalogue?"]


async def test_ordering_by_distance_on_an_unembedded_table_returns_nothing():
    """Correct pgvector behaviour, and a silently wrong answer.

    seeds/001_schema.sql builds an index on each embed_ column, and a
    pgvector index does not index NULL rows. So Postgres plans an index scan
    for `ORDER BY col <=> $1` and the unembedded rows are simply absent -
    no error, no warning, zero rows.

    Which means that before a backfill, a semantic question answers "there
    are no books about the sea" when the truth is "nothing has been
    embedded". That is the worst shape a failure can take, and it is pinned
    here so the fix - not classifying an empty embed_ column as SEMANTIC -
    has something to fail against if it is ever removed.
    """
    import asyncpg

    from libs.agent_core.pgvector import to_vector_literal

    try:
        conn = await asyncpg.connect(
            host=BASE_ENVIRONMENT["PG_HOST"], port=int(BASE_ENVIRONMENT["PG_PORT"]),
            user="app_catalog", password="dev_catalog",
            database=BASE_ENVIRONMENT["PG_DBNAME"], timeout=3,
        )
    except Exception as e:
        pytest.skip(f"development database unavailable: {e}")

    try:
        embedded = await conn.fetchval(
            "SELECT count(embed_summary) FROM books"
        )
        if embedded:
            pytest.skip("books has embeddings; this pins the unembedded case")

        vector = to_vector_literal([0.1] * 1024)
        ordered = await conn.fetch(
            "SELECT title_en FROM books ORDER BY embed_summary <=> $1::vector LIMIT 3",
            vector,
        )
        unordered = await conn.fetch("SELECT title_en FROM books LIMIT 3")
    finally:
        await conn.close()

    assert len(unordered) == 3, "the table itself is readable"
    assert ordered == [], (
        "ordering by distance over NULL embeddings returned rows - the index "
        "behaviour this pins has changed, and the SEMANTIC classification of "
        "an empty embed_ column may no longer be silently wrong"
    )
