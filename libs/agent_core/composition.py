"""The composition root: where ports meet adapters, and the only place that
knows both.

Split in two on purpose.

`assemble_*` take already-built ports and return an interactor. They open
nothing, so they are testable with fakes and hold every wiring decision worth
arguing about - which prompt, which tools, which agent becomes a tool.

`open_runtime` acquires the real resources: pools, a Redis connection, an
HTTP client, an OpenAI client. It is the part that cannot be unit-tested and
therefore the part that should contain no decisions.

One image, two modes, selected by AGENT_KEY:

    AGENT_KEY=catalog   a sub-agent runtime. Connects as that agent's role,
                        verifies its scope against the GRANTs, and serves one
                        endpoint that answers self-contained questions.

    AGENT_KEY=""        the orchestrator. No database of its own beyond
                        history; it holds one delegate tool per registered
                        sub-agent and routes by their descriptions.

Connecting as an agent's role
-----------------------------
Every agent pool logs in as the *authenticator* and runs `SET ROLE <db_role>`
on each new connection. One credential for the whole service rather than one
per agent, which past a handful of agents is the difference between a config
file and a secret-distribution problem.

The authenticator is NOINHERIT (see seeds/003_roles.sql), so it holds nothing
until it becomes somebody. That is what makes this safe rather than merely
convenient: a connection that has not run SET ROLE can read nothing, and one
that ran RESET ROLE goes back to reading nothing. The floor is empty, not the
union of every agent's privileges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from adapters.outbound.agent_loop.langgraph_agent_loop_adapter import (
    DEFAULT_MAX_ITERATIONS,
    LangGraphAgentLoopAdapter,
)
from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter
from adapters.outbound.history.postgres_history_adapter import PostgresHistoryAdapter
from adapters.outbound.registry.file_agent_registry_adapter import (
    FileAgentRegistryAdapter,
)
from adapters.outbound.schema.postgres_introspection_adapter import (
    PostgresIntrospectionAdapter,
)
from adapters.outbound.tools.http_delegate_tool_adapter import HttpDelegateToolAdapter
from adapters.outbound.tools.sql_tool_adapter import SqlToolAdapter
from domain.entities.provider_spec import AgentType, ProviderSpec
from domain.interactors.run_agent_turn import RunAgentTurn
from libs.agent_core import config
from libs.agent_core.agent_startup import ReadyAgent, start_agent
from libs.agent_core.prompts import (
    ORCHESTRATOR_PROMPT,
    describe_agent,
    sub_agent_prompt,
)
from libs.agent_core.sql_validation import validate_identifier


# --------------------------------------------------------------------------
# assembly - decisions, no I/O
# --------------------------------------------------------------------------


def assemble_sub_agent(
    ready: ReadyAgent,
    *,
    db,
    embeddings,
    cache,
    history,
    llm_factory: Callable[[list[dict]], Any],
    dist_op: str = "<=>",
    vector_ttl_seconds: int = 900,
    max_pages_per_tool: int = 5,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_session_messages: int = 40,
    context_messages_sent: int = 20,
    session_ttl_seconds: int = 60 * 60 * 24 * 3,
) -> RunAgentTurn:
    """One sub-agent's turn interactor, wired to its own tables.

    `allowed_tables`, `schema` and `filters` all come from `ready`, which came
    from introspecting through the agent's role - so what the model is told it
    may read is what the database will actually let it read. Nothing here
    consults the registry's table list; that was already compared against the
    GRANTs at startup and its job is done.

    `use_conversation_history=False` is the significant argument. A sub-agent
    is called by the orchestrator with a self-contained question and must
    answer it from that question alone. Giving it a conversation window would
    let two callers' contexts mix, and would make the same question return
    different answers depending on what was asked before it.
    """
    tools = SqlToolAdapter(
        db=db,
        embeddings=embeddings,
        cache=cache,
        allowed_tables=list(ready.allowed_tables),
        schema=ready.schema.schema,
        filters=ready.schema.filters,
        lsit_values={},
        dist_op=dist_op,
        vector_ttl_seconds=vector_ttl_seconds,
    )
    loop = LangGraphAgentLoopAdapter(
        llm=llm_factory(tools.get_tool_schemas()),
        tools=tools,
        max_pages_per_tool=max_pages_per_tool,
        max_iterations=max_iterations,
    )
    return RunAgentTurn(
        agent_loop=loop,
        history=history,
        cache=cache,
        system_prompt=sub_agent_prompt(ready.spec),
        use_conversation_history=False,
        session_ttl_seconds=session_ttl_seconds,
        max_session_messages=max_session_messages,
        context_messages_sent=context_messages_sent,
    )


def delegate_targets(
    specs, url_template: str = config.AGENT_URL_TEMPLATE,
) -> tuple[dict[str, str], dict[str, str]]:
    """The orchestrator's tools: one per sub-agent, as URLs and descriptions.

    Only SUB_AGENT-typed entries. A registry may describe an orchestrator too,
    and giving the orchestrator a tool that calls itself is a loop with a
    plausible-looking first step.

    A missing description is impossible to reach from here - ProviderSpec
    refuses to construct without one - which is deliberate: the orchestrator
    routes on these, and a placeholder does not fail loudly, it quietly makes
    an agent unreachable.
    """
    urls, descriptions = {}, {}
    for spec in specs:
        if spec.type is not AgentType.SUB_AGENT:
            continue
        urls[spec.name] = spec.endpoint or url_template.format(key=spec.name)
        descriptions[spec.name] = describe_agent(spec)
    return urls, descriptions


def assemble_orchestrator(
    specs,
    *,
    http_client,
    cache,
    history,
    llm_factory: Callable[[list[dict]], Any],
    url_template: str = config.AGENT_URL_TEMPLATE,
    timeout: int = 60,
    max_pages_per_tool: int = 5,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_session_messages: int = 40,
    context_messages_sent: int = 20,
    session_ttl_seconds: int = 60 * 60 * 24 * 3,
) -> RunAgentTurn:
    """The routing interactor. One delegate tool per registered sub-agent.

    `use_conversation_history=True`, the opposite of a sub-agent, and for the
    same reason: this is the end of the system that talks to a person, so it
    is the one that has to remember what "it" refers to. Resolving those
    references before delegating is the orchestrator's job precisely because
    it is the only component that can.

    The tool list is built from the registry, so registering an agent is
    enough to make it routable. There is no second list here to update and
    forget.
    """
    urls, descriptions = delegate_targets(specs, url_template)
    if not urls:
        raise ValueError(
            "No sub-agents registered, so the orchestrator has nothing to route "
            "to. It would accept questions and be unable to answer any of them."
        )

    tools = HttpDelegateToolAdapter(
        client=http_client,
        tool_urls=urls,
        tool_descriptions=descriptions,
        timeout=timeout,
    )
    loop = LangGraphAgentLoopAdapter(
        llm=llm_factory(tools.get_tool_schemas()),
        tools=tools,
        max_pages_per_tool=max_pages_per_tool,
        max_iterations=max_iterations,
    )
    return RunAgentTurn(
        agent_loop=loop,
        history=history,
        cache=cache,
        system_prompt=ORCHESTRATOR_PROMPT,
        use_conversation_history=True,
        session_ttl_seconds=session_ttl_seconds,
        max_session_messages=max_session_messages,
        context_messages_sent=context_messages_sent,
    )


# --------------------------------------------------------------------------
# the running process
# --------------------------------------------------------------------------


@dataclass
class Runtime:
    """What one process is, and how to shut it down.

    `closers` rather than a list of typed handles: the only thing the caller
    does with them is call them in reverse, and naming each resource here
    would mean this class changing every time a dependency does.
    """

    kind: str
    turn: RunAgentTurn
    agent: ReadyAgent | None = None
    routes_to: tuple[str, ...] = ()
    closers: list[Callable] = field(default_factory=list)

    def allowed_tables_or_empty(self) -> tuple[str, ...]:
        """This process's tables, or nothing for an orchestrator.

        The orchestrator has no tables of its own, and asking it for them is
        a reasonable question with a real answer rather than an error.
        """
        return self.agent.allowed_tables if self.agent else ()

    async def aclose(self) -> None:
        """Close every resource, in reverse, and never stop early.

        A pool that fails to close must not leave the Redis connection open
        behind it - shutdown is the one path where continuing past an error
        is right, because there is nothing left to protect.
        """
        errors = []
        for close in reversed(self.closers):
            try:
                await close()
            except Exception as e:  # noqa: BLE001 - reported, not swallowed
                errors.append(e)
        if errors:
            raise RuntimeError(f"Errors while shutting down: {errors}")


ORCHESTRATOR_HISTORY_TABLE = "history_orchestrator"


async def verify_history_table(db, table: str) -> None:
    """Check the service can actually write history, before serving anything.

    The service does not create this table. Creating one is DDL, and a
    process that answers questions must not hold DDL rights - the same
    argument that keeps CREATE ROLE in the provisioner. So startup asserts
    the table is there rather than making it, and says what to run when it is
    not.

    Checked at startup rather than discovered on the first question, because
    a history write happens partway through answering: the model has already
    been called, the SQL has already run, and the failure surfaces as a lost
    answer rather than as a missing table.
    """
    name = validate_identifier(table)
    try:
        await db.fetch(f"SELECT 1 FROM {name} LIMIT 1")
    except Exception as e:
        raise RuntimeError(
            f"Cannot read the history table {name!r}: {e}\n"
            "This process does not create it - creating a table is DDL, which "
            "belongs to the provisioner. Run seeds/004_history.sql, or have the "
            "provisioner create it, then start again."
        ) from e


async def _agent_pool(asyncpg, spec: ProviderSpec, **kw):
    """A pool that has already become the agent, on every connection.

    `init` runs once per new connection, and the pool is dedicated to one
    agent, so there is no path by which a connection serves a different role
    than the one it was opened for. The identifier is validated again here
    even though ProviderSpec already did: this is the line where it reaches
    SQL, and a check at the point of interpolation is the one that cannot be
    bypassed by a caller that built its spec another way.
    """
    role = validate_identifier(spec.db_role)

    async def become_the_agent(connection):
        await connection.execute(f"SET ROLE {role}")

    return await asyncpg.create_pool(init=become_the_agent, **kw)


def _pool_kwargs() -> dict:
    return dict(
        host=config.PG_HOST, port=config.PG_PORT, database=config.PG_DBNAME,
        user=config.PG_USER, password=config.PG_PASSWORD,
        min_size=config.DB_POOL_MIN, max_size=config.DB_POOL_MAX,
        command_timeout=config.DB_COMMAND_TIMEOUT,
        ssl="require" if config.PG_SSL else None,
    )


async def open_runtime(agent_key: str | None = None) -> Runtime:
    """Build whatever this process is, from configuration and the registry.

    Imports its dependencies inside the function rather than at module level.
    That is not style: importing this module has to stay free of asyncpg,
    redis, httpx and openai so that assemble_* can be tested without any of
    them installed, and so a test suite never accidentally reaches the
    network by importing a name.
    """
    import asyncpg
    import httpx
    import redis.asyncio as redis
    from openai import AsyncOpenAI

    from adapters.outbound.cache.redis_cache_adapter import RedisCacheAdapter
    from adapters.outbound.embedding.qwen_embedding_adapter import (
        QwenEmbeddingAdapter,
    )
    from adapters.outbound.llm.qwen_llm_adapter import QwenLLMAdapter

    config.validate()
    key = config.AGENT_KEY if agent_key is None else agent_key
    registry = FileAgentRegistryAdapter(config.AGENTS_REGISTRY_PATH)

    closers: list[Callable] = []
    llm_client = AsyncOpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_API_URL)
    closers.append(llm_client.close)

    # One client when both point at the same endpoint, two when they do not.
    # Building a second unconditionally would open a second connection pool
    # to the same host for no reason; sharing one unconditionally would send
    # the chat key to the embedding server.
    if (config.EMBED_API_URL, config.EMBED_API_KEY) == (
        config.QWEN_API_URL, config.QWEN_API_KEY
    ):
        embed_client = llm_client
    else:
        embed_client = AsyncOpenAI(
            api_key=config.EMBED_API_KEY, base_url=config.EMBED_API_URL
        )
        closers.append(embed_client.close)

    redis_client = redis.from_url(config.REDIS_URL, decode_responses=False)
    closers.append(redis_client.aclose)

    # Asked at startup, because from_url does not connect - it builds a
    # client and waits. Without this a process comes up healthy against a
    # Redis that is not there and fails on the first question, after the
    # model has been called and paid for, with a message about a lock.
    #
    # The same rule as the history table and the GRANT check: a
    # misconfigured process should be one that refuses to start, not one
    # that answers wrongly.
    try:
        await redis_client.ping()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Redis at {config.REDIS_URL}: {e}\n"
            "It holds the per-session lock and the conversation window, so "
            "nothing can be answered without it."
        ) from e

    cache = RedisCacheAdapter(redis_client)

    embeddings = QwenEmbeddingAdapter(embed_client, config.QWEN_EMBED_MODEL)

    def llm_factory(tool_schemas):
        return QwenLLMAdapter(
            client=llm_client, model=config.QWEN_MODEL,
            temperature=config.QWEN_TEMPERATURE, max_tokens=config.QWEN_MAX_TOKENS,
            tools=tool_schemas,
            enable_thinking=config.QWEN_ENABLE_THINKING,
        )

    # History is written by the service, not by an agent, so it goes through
    # the authenticator's own pool - no SET ROLE. An agent role holds SELECT
    # on its own tables and nothing else, and giving it INSERT anywhere would
    # widen exactly the privilege this design spends its effort narrowing.
    # seeds/004_history.sql grants SELECT and INSERT on the history tables to
    # the authenticator alone - and no DELETE or UPDATE, because a process
    # that can rewrite history has an audit trail that means less than it
    # appears to.
    service_pool = await asyncpg.create_pool(**_pool_kwargs())
    closers.append(service_pool.close)

    try:
        if key:
            spec = await registry.get(key)
            agent_pool = await _agent_pool(asyncpg, spec, **_pool_kwargs())
            closers.append(agent_pool.close)

            agent_db = PostgresDatabaseAdapter(agent_pool)
            ready = await start_agent(
                spec,
                PostgresIntrospectionAdapter(agent_db),
                dist_op=config.DIST_OP,
            )
            service_db = PostgresDatabaseAdapter(service_pool)
            await verify_history_table(service_db, spec.history_table)
            history = PostgresHistoryAdapter(
                db=service_db, embeddings=embeddings,
                table_name=spec.history_table, embedding_dim=config.EMBEDDING_DIM,
            )

            return Runtime(
                kind="sub_agent",
                turn=assemble_sub_agent(
                    ready, db=agent_db, embeddings=embeddings, cache=cache,
                    history=history, llm_factory=llm_factory,
                    dist_op=config.DIST_OP,
                    vector_ttl_seconds=config.VECTOR_TTL_SECONDS,
                    max_pages_per_tool=config.MAX_PAGES_PER_TOOL,
                    max_iterations=config.MAX_LOOP_ITERATIONS,
                    max_session_messages=config.MAX_SESSION_MESSAGES,
                    context_messages_sent=config.CONTEXT_MESSAGES_SENT,
                    session_ttl_seconds=config.SESSION_TTL_SECONDS,
                ),
                agent=ready,
                closers=closers,
            )

        http_client = httpx.AsyncClient()
        closers.append(http_client.aclose)

        specs = await registry.list_active()
        service_db = PostgresDatabaseAdapter(service_pool)
        await verify_history_table(service_db, ORCHESTRATOR_HISTORY_TABLE)
        history = PostgresHistoryAdapter(
            db=service_db, embeddings=embeddings,
            table_name=ORCHESTRATOR_HISTORY_TABLE, embedding_dim=config.EMBEDDING_DIM,
        )

        turn = assemble_orchestrator(
            specs, http_client=http_client, cache=cache, history=history,
            llm_factory=llm_factory, url_template=config.AGENT_URL_TEMPLATE,
            timeout=config.TOOLS_HTTP_TIMEOUT_SECS,
            max_pages_per_tool=config.MAX_PAGES_PER_TOOL,
            max_iterations=config.MAX_LOOP_ITERATIONS,
            max_session_messages=config.MAX_SESSION_MESSAGES,
            context_messages_sent=config.CONTEXT_MESSAGES_SENT,
            session_ttl_seconds=config.SESSION_TTL_SECONDS,
        )
        urls, _ = delegate_targets(specs, config.AGENT_URL_TEMPLATE)
        return Runtime(kind="orchestrator", turn=turn,
                       routes_to=tuple(urls), closers=closers)

    except Exception:
        # Half-open resources are worse than none: a failed startup that
        # leaves a pool behind will not be cleaned up by anything, because
        # nothing has a handle to it.
        await Runtime(kind="failed", turn=None, closers=closers).aclose()
        raise
