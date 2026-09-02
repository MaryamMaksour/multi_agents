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

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from adapters.inbound.http.schemas import (
    AgentSummary,
    AskRequest,
    DelegatedQuestion,
    HealthResponse,
    RunRequest,
    TurnResponse,
)
from domain.entities.chat_message import Role
from domain.exceptions import (
    DomainError,
    SessionBusyError,
    UnknownAgentError,
)
from libs.agent_core import config
from libs.agent_core.composition import open_runtime
from libs.agent_core.logging_setup import (
    Timer,
    bind,
    configure_logging,
    context,
    current_context,
    log_event,
    new_request_id,
    unbind,
)

logger = logging.getLogger(__name__)


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


def delegated_questions(result) -> list[DelegatedQuestion]:
    """What the orchestrator actually asked each agent.

    Read from the turn's own messages rather than recorded separately: the
    tool calls are already there, and a second record of the same fact is a
    second thing that can be wrong.
    """
    asked = []
    for message in result.messages:
        for call in getattr(message, "tool_calls", None) or []:
            question = (call.args or {}).get("query")
            if isinstance(question, str) and question.strip():
                asked.append(DelegatedQuestion(agent=call.name, question=question))
    return asked


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

    # Before anything else in the process, and before open_runtime can raise:
    # a startup that fails while logging is still unconfigured fails silently,
    # which is the failure this whole file exists to make visible.
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_event(logger, "startup.begin", agent_key=config.AGENT_KEY or "(orchestrator)")
        with Timer() as timer:
            try:
                runtime = await open_runtime_fn()
            except Exception:
                # exc_info, not str(e): a startup failure is usually a
                # connection error several frames down, and the frame that
                # raised is the diagnosis. The formatter redacts the DSN.
                logger.critical("startup.failed", exc_info=True)
                raise

        # Bound for the life of the process, not per request: every line this
        # container writes says which agent wrote it, which is what makes one
        # `docker compose logs` readable across five of them.
        bind(agent=runtime.agent.spec.name if runtime.agent else "orchestrator")

        log_event(
            logger, "startup.ready", ms=timer.ms, kind=runtime.kind,
            tables=list(runtime.allowed_tables_or_empty()),
            routes_to=list(runtime.routes_to),
            model=config.QWEN_MODEL, embed_model=config.QWEN_EMBED_MODEL,
        )
        app.state.runtime = runtime
        try:
            yield
        finally:
            log_event(logger, "shutdown.begin")
            try:
                await runtime.aclose()
            except Exception:
                logger.error("shutdown.errors", exc_info=True)
                raise
            log_event(logger, "shutdown.done")

    app = FastAPI(
        title="Multi-agent runtime",
        summary="One process per agent, plus an orchestrator that routes between them.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _log_request(request: Request, call_next):
        """One line in, one line out, with an id that ties them together.

        The id also crosses process boundaries: the orchestrator sends its
        request id to a sub-agent as X-Request-ID, so a question and the two
        delegated calls it produced share one id across three containers. A
        caller that supplies one keeps it, which is what makes the id usable
        from outside as well.

        /health is logged at DEBUG. A container health check runs it every few
        seconds, and at INFO it buries every line that matters.
        """
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming[:64] if incoming else new_request_id()
        tokens = bind(request_id=request_id)

        level = logging.DEBUG if request.url.path == "/health" else logging.INFO
        log_event(logger, "http.request", level=level,
                  method=request.method, path=request.url.path)

        with Timer() as timer:
            try:
                response = await call_next(request)
            except Exception:
                # FastAPI's handlers convert DomainError into a response, so
                # anything arriving here is unexpected - and the only place it
                # is recorded, because the client gets a bare 500.
                log_event(logger, "http.unhandled", level=logging.ERROR,
                          method=request.method, path=request.url.path, ms=timer.ms)
                logger.error("unhandled exception serving the request", exc_info=True)
                unbind(tokens)
                raise

        log_event(logger, "http.response",
                  level=logging.ERROR if response.status_code >= 500 else level,
                  method=request.method, path=request.url.path,
                  status=response.status_code, ms=timer.ms)

        # Echoed so a caller can quote it when reporting a problem, and so the
        # orchestrator can record which id a sub-agent used.
        response.headers["x-request-id"] = request_id
        unbind(tokens)
        return response

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
        # The body carries the type and message; the traceback stays here.
        # Logged at ERROR because a DomainError reaching the edge means a turn
        # was lost, whatever the caller does with the 500.
        logger.error("domain error answering the request", exc_info=exc)

        # ...unless the deployment says otherwise. A DatabaseError carries the
        # SQL that failed, which describes the schema; that is fine when the
        # callers own the database and not fine when they do not. The log has
        # the full message either way, so switching this off costs nothing but
        # a longer walk to the diagnosis.
        if not config.HTTP_ERROR_DETAIL:
            raise HTTPException(
                status_code=500,
                detail="The request could not be completed. The reason is in "
                       "the service log, under this request id.",
                headers={"x-request-id": current_context().get("request_id", "")},
            )

        raise HTTPException(
            status_code=500, detail=f"{type(exc).__name__}: {exc}"
        )

    @app.exception_handler(SessionBusyError)
    async def _busy(request: Request, exc: SessionBusyError):
        # 409, not 500: the caller did nothing wrong and retrying is the
        # correct response, which is a different instruction from "this broke".
        # WARNING, not ERROR, for the same reason - but logged, because a
        # session that is busy every time is a lock that is not being released.
        log_event(logger, "session.busy", level=logging.WARNING, detail=str(exc))
        raise HTTPException(status_code=409, detail=str(exc))

    @app.exception_handler(UnknownAgentError)
    async def _unknown(request: Request, exc: UnknownAgentError):
        log_event(logger, "agent.unknown", level=logging.WARNING, detail=str(exc))
        raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        runtime = request.app.state.runtime
        return HealthResponse(
            status="ok",
            kind=runtime.kind,
            agent=runtime.agent.spec.name if runtime.agent else None,
            model=config.QWEN_MODEL,
            thinking=config.QWEN_ENABLE_THINKING,
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

        # The ids label every line the turn produces, at any depth - the SQL
        # the tool ran, the model call, the history write - so a wrong answer
        # can be read back from the logs as one sequence rather than searched
        # for among interleaved turns.
        with context(session_id=session_id, turn_id=turn_id):
            log_event(logger, "run.received", question=body.user_input)
            with Timer() as timer:
                result = await runtime.turn.run(
                    session_id=session_id, turn_id=turn_id,
                    user_input=body.user_input,
                )
            answer = final_answer(result)
            log_event(logger, "run.answered", ms=timer.ms, answer=answer,
                      messages=len(result.messages))

        return TurnResponse(
            answer=answer, session_id=session_id, turn_id=turn_id,
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

        with context(session_id=body.session_id, turn_id=turn_id):
            log_event(logger, "ask.received", question=body.question)
            with Timer() as timer:
                result = await runtime.turn.run(
                    session_id=body.session_id, turn_id=turn_id,
                    user_input=body.question,
                )
            answer = final_answer(result)
            delegated = delegated_questions(result)

            # The delegated questions are the orchestrator's actual output.
            # An answer of 129 to a question about novels is either a bad
            # delegation or a bad sub-agent, and this line is what separates
            # them without opening the database.
            log_event(logger, "ask.answered", ms=timer.ms, answer=answer,
                      delegated=[{"agent": d.agent, "question": d.question}
                                 for d in delegated])

        return TurnResponse(
            answer=answer, session_id=body.session_id, turn_id=turn_id,
            pagination=pagination_payload(result),
            delegated=delegated,
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
