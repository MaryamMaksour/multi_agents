"""The HTTP edge. One image, two shapes, decided by AGENT_KEY.

Thin on purpose. Every routing decision, every prompt and every wiring choice
lives in libs/agent_core/composition.py; this file converts JSON to arguments
and exceptions to status codes. If a change to how the system behaves needs
editing this file, it is in the wrong place.

The lifespan is where the process's resources are owned. Pools open once at
startup and close once at shutdown, and startup verification - introspecting
through the agent's role and comparing it against the registry - happens here
too, which is what makes a misconfigured agent a container that refuses to
start rather than one that answers wrongly.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from adapters.inbound.http.schemas import (
    AgentSummary,
    AskRequest,
    HealthResponse,
    RunRequest,
    TurnResponse,
)
from domain.entities.chat_message import Role
from domain.exceptions import (
    DomainError,
    GrantMismatchError,
    RegistryError,
    SessionBusyError,
    UnknownAgentError,
)
from libs.agent_core import config
from libs.agent_core.composition import open_runtime


def final_answer(result) -> str:
    """The last thing the model said, as text.

    A turn's messages include tool calls and tool results; the answer is the
    last assistant message with content. Walking backwards rather than taking
    messages[-1] because a turn can end on a tool result when the loop hits
    its page limit, and returning JSON to a person is not an answer.
    """
    for message in reversed(result.messages):
        if message.role is Role.ASSISTANT and (message.content or "").strip():
            return message.content
    return ""


def pagination_payload(result) -> dict:
    return {
        tool: {
            "has_more": state.has_more,
            "next_cursor": state.next_cursor,
            "pages_fetched": state.pages_fetched,
        }
        for tool, state in (result.pagination or {}).items()
    }


def create_app(open_runtime_fn=open_runtime) -> FastAPI:
    """Build the app. `open_runtime_fn` is the only seam.

    Injected rather than imported at the call site so the routes can be
    exercised against a fake runtime - a test that needed Postgres, Redis and
    an LLM before it could check that /run returns 404 on the orchestrator
    would not get written.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await open_runtime_fn()
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.aclose()

    app = FastAPI(
        title="Multi-agent runtime",
        summary="One process per agent, plus an orchestrator that routes between them.",
        lifespan=lifespan,
    )

    @app.exception_handler(DomainError)
    async def _domain_error(request: Request, exc: DomainError):
        """Answer with what actually failed, not "Internal Server Error".

        FastAPI's default 500 body says nothing, which sends whoever is
        debugging to `docker compose logs` for every failure - and on a
        machine where the containers are not theirs, sometimes nowhere at
        all. The traceback stays in the logs; the type and message come back
        here, because in practice that pair is the whole diagnosis:
        "CacheError: Extra data: line 1 column 9" named the bug exactly.

        Scoped to DomainError, so this is only ever our own exceptions with
        our own wording. An unexpected one still returns a bare 500 rather
        than whatever a library happened to put in its message.
        """
        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        )

    @app.exception_handler(SessionBusyError)
    async def _busy(request: Request, exc: SessionBusyError):
        # 409, not 500: the caller did nothing wrong and retrying is the
        # correct response, which is a different instruction from "this broke".
        raise HTTPException(status_code=409, detail=str(exc))

    @app.exception_handler(UnknownAgentError)
    async def _unknown(request: Request, exc: UnknownAgentError):
        raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        runtime = request.app.state.runtime
        return HealthResponse(
            status="ok",
            kind=runtime.kind,
            agent=runtime.agent.spec.name if runtime.agent else None,
            tables=list(runtime.agent.allowed_tables) if runtime.agent else [],
            routes_to=list(runtime.routes_to),
        )

    @app.post("/run", response_model=TurnResponse)
    async def run(body: RunRequest, request: Request) -> TurnResponse:
        """Answer one self-contained question. The sub-agent endpoint.

        Called by the orchestrator, not by a person. A sub-agent holds no
        conversation history, so `session_id` here is for correlation and
        locking rather than for memory - two questions with the same id are
        serialised, and neither can see the other.
        """
        runtime = request.app.state.runtime
        if runtime.kind != "sub_agent":
            raise HTTPException(
                status_code=404,
                detail="This process is the orchestrator; it has no /run. Post "
                       "to /ask, or set AGENT_KEY to serve one agent.",
            )

        session_id = body.session_id or f"delegate:{uuid.uuid4()}"
        turn_id = body.context.turn_id or str(uuid.uuid4())

        result = await runtime.turn.run(
            session_id=session_id, turn_id=turn_id, user_input=body.user_input,
        )
        return TurnResponse(
            answer=final_answer(result), session_id=session_id, turn_id=turn_id,
            pagination=pagination_payload(result),
        )

    @app.post("/ask", response_model=TurnResponse)
    async def ask(body: AskRequest, request: Request) -> TurnResponse:
        """Answer a person's question by delegating. The orchestrator endpoint."""
        runtime = request.app.state.runtime
        if runtime.kind != "orchestrator":
            raise HTTPException(
                status_code=404,
                detail=f"This process serves the {runtime.agent.spec.name!r} agent "
                       "and cannot route. Post to /run, or unset AGENT_KEY to run "
                       "the orchestrator.",
            )

        turn_id = str(uuid.uuid4())
        result = await runtime.turn.run(
            session_id=body.session_id, turn_id=turn_id, user_input=body.question,
        )
        return TurnResponse(
            answer=final_answer(result), session_id=body.session_id, turn_id=turn_id,
            pagination=pagination_payload(result),
        )

    @app.get("/agents", response_model=list[AgentSummary])
    async def agents(request: Request) -> list[AgentSummary]:
        """Who this orchestrator can route to.

        Reads the registry rather than the tool list, so an agent that was
        registered but is not routable is visible as such instead of simply
        absent - "why is nothing answering about loans" is a much shorter
        question when the answer is a status.
        """
        from adapters.outbound.registry.file_agent_registry_adapter import (
            FileAgentRegistryAdapter,
        )

        registry = FileAgentRegistryAdapter(config.AGENTS_REGISTRY_PATH)
        return [
            AgentSummary(
                key=spec.name, display_name=spec.display_name,
                description=spec.description, status=spec.status.value,
            )
            for spec in await registry.list_active()
        ]

    return app
