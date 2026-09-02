# domain/interactors/run_agent_turn.py
from __future__ import annotations

import logging
import time

from domain.ports.agent_loop_port import AgentLoopPort
from domain.ports.history_port import HistoryPort
from domain.ports.cache_port import CachePort

from domain.entities.agent_turn import AgentTurnResult
from domain.entities.chat_message import ChatMessage, Role

from domain.exceptions import HistoryError, SessionBusyError

logger = logging.getLogger(__name__)

# Structured logging from the domain, without importing the logging library
# that renders it.
#
# The interactor is the layer that knows what a turn is, so it is the layer
# whose log lines are worth the most - but it may not import libs/. So it
# logs through stdlib `logging` with `extra=`, which is an interface the
# standard library defines, and the formatter in libs/agent_core renders
# those fields in whichever format the deployment asked for. The turn and
# session ids are not passed either: they are bound at the HTTP edge and
# travel by contextvars, so they arrive on these records without the domain
# knowing they exist.
def _event(event: str, level: int = logging.INFO, **fields) -> None:
    if logger.isEnabledFor(level):
        logger.log(level, event, extra=fields)


class RunAgentTurn:
    def __init__(self,
                 agent_loop: AgentLoopPort,
                 history: HistoryPort,
                 cache: CachePort,
                 system_prompt: str,
                 use_conversation_history: bool = True,
                 session_ttl_seconds: int = 60 * 60 * 24 * 3,  # 3 days
                 max_session_messages: int = 40,
                 context_messages_sent: int = 20
                 ):

        self.agent_loop = agent_loop
        self.history = history
        self.cache = cache
        self.system_prompt = system_prompt
        self.use_conversation_history = use_conversation_history
        self.session_ttl_seconds = session_ttl_seconds
        self.max_session_messages = max_session_messages
        self.context_messages_sent = context_messages_sent

    async def run(self, session_id: str, turn_id: str, user_input: str) -> AgentTurnResult:
        started_at = time.time()

        # Serialize turns for the same session: two overlapping requests for the
        # same session_id must not run concurrently and corrupt each other's history.
        lock_started = time.monotonic()
        acquired = await self.cache.acquire_lock(key=session_id, timeout=120)
        lock_ms = (time.monotonic() - lock_started) * 1000
        if not acquired:
            # Not an error here - the edge turns it into a 409 - but recorded,
            # because a session that is always busy means a lock that is not
            # being released, and that is invisible from the 409 alone.
            _event("turn.session_busy", logging.WARNING, waited_ms=lock_ms)
            raise SessionBusyError("Session is currently busy. Please try again later.")

        _event("turn.start", chars=len(user_input), lock_ms=lock_ms,
               history=self.use_conversation_history)

        try:
            # Sub-agents are stateless (every call is a fresh, self-contained
            # request), so they never load a conversation window. The orchestrator
            # does, for multi-turn continuity with the end user.
            cache_data = []
            if self.use_conversation_history:
                cache_data = await self.cache.get(key=session_id)
                _event("turn.window_loaded", logging.DEBUG,
                       messages=len(cache_data))

            # Semantic search over past turns: enrich the prompt with worked
            # examples (question + reasoning trace) similar to this new
            # question.
            #
            # An optimisation, and treated as one. It costs an embedding call
            # and a vector search, and when either fails the right answer is
            # a turn with no examples - not a lost answer. The same rule the
            # ENUM probe follows at startup: degrade quality, never raise.
            #
            # This matters more than it looks. The examples are the *first*
            # thing a turn does, so without this an embedding model the
            # account cannot reach takes down every question, including the
            # ones that need no memory at all.
            try:
                memory_started = time.monotonic()
                memory_examples = await self.history.get_memory(query=user_input)
                _event("turn.memory", logging.DEBUG,
                       examples=len(memory_examples),
                       ms=(time.monotonic() - memory_started) * 1000)
            except HistoryError:
                logger.warning(
                    "No worked examples this turn - the memory lookup failed. "
                    "The answer is unaffected; the prompt is thinner.",
                    exc_info=True,
                )
                memory_examples = []

            # Persist the incoming user message for audit + future memory
            # lookups. Also non-fatal, and for a narrower reason: losing an
            # audit row is bad, losing the user's answer because an audit row
            # could not be written is worse. It is logged loudly rather than
            # swallowed, so a history that has stopped recording is visible
            # instead of merely absent.
            try:
                await self.history.log_user_message(session_id=session_id, turn_id=turn_id, message=user_input)
            except HistoryError:
                logger.error(
                    "History not recorded for turn %s - this turn will be "
                    "answered but will not appear in the audit trail.",
                    turn_id, exc_info=True,
                )

            # `messages` is what's actually sent to the LLM. `new_messages`
            # mirrors it but excludes the system prompt and the examples -
            # both are rebuilt fresh every turn, so only the rest is persisted
            # back to the conversation window.
            #
            # The order matters for cost, not only for the model. Providers
            # cache by matching the *prefix* of a request, and the examples
            # change on every turn - so concatenating them into the system
            # prompt, as this did, moves the first difference to byte zero and
            # nothing after it can ever be cached. The tool schemas alone are
            # ~1,070 tokens resent on every call in the loop, which is most of
            # the bill. System prompt first and unchanged, then the window
            # (append-only within a session), then the volatile part last.
            #
            # It reads better too: worked examples belong next to the question
            # they are examples for, not several thousand tokens above it.
            context_window = cache_data[-self.context_messages_sent:]
            messages = [ChatMessage(role=Role.SYSTEM, content=self.system_prompt)] + context_window

            if memory_examples:
                messages.append(ChatMessage(
                    role=Role.SYSTEM,
                    content="Worked examples from earlier turns: " + str(memory_examples),
                ))

            messages.append(ChatMessage(role=Role.USER, content=user_input))
            new_messages = [ChatMessage(role=Role.USER, content=user_input)]

            _event("turn.loop_begin", logging.DEBUG,
                   sent=len(messages), examples=len(memory_examples))

            loop_started = time.monotonic()
            try:
                loop_result = await self.agent_loop.run(messages)
            except Exception:
                # The one place a failed turn is fully described: which
                # question, how far in, and the frame that raised. The edge
                # sees only the exception type.
                _event("turn.failed", logging.ERROR,
                       ms=(time.monotonic() - loop_started) * 1000,
                       chars=len(user_input))
                logger.error("the agent loop raised", exc_info=True)
                raise
            loop_ms = (time.monotonic() - loop_started) * 1000

            messages += loop_result.messages
            new_messages += loop_result.messages
            pagination = loop_result.pagination

            # Persist the updated conversation window (system prompt excluded -
            # it's rebuilt fresh every turn), trimmed to the retention limit.
            if self.use_conversation_history:
                updated_window = (cache_data + new_messages)[-self.max_session_messages:]
                await self.cache.set(key=session_id, value=updated_window, ttl=self.session_ttl_seconds)
                _event("turn.window_saved", logging.DEBUG,
                       messages=len(updated_window),
                       trimmed=len(cache_data) + len(new_messages) - len(updated_window))


            # Log the full trace (tool calls + results + final answer) - this is
            # what get_memory() later returns as a worked example, and what marks
            # this turn valid/invalid based on whether an error shows up in it.
            elapsed = time.time() - started_at

            # The turn's summary line: how long, how many round trips, how
            # many tools, and what came back. Everything needed to see that a
            # turn went wrong without reading the trace, and enough to know
            # whether reading the trace is worth it.
            tool_calls = [
                call.name
                for message in loop_result.messages
                for call in (getattr(message, "tool_calls", None) or [])
            ]
            _event("turn.done", ms=loop_ms, total_ms=elapsed * 1000,
                   messages=len(loop_result.messages),
                   tool_calls=tool_calls, pagination=bool(pagination))

            try:
                await self.history.log_assistant_final(session_id=session_id, turn_id=turn_id, final_answer=new_messages, elapsed=elapsed)
            except HistoryError:
                logger.error(
                    "Final answer not recorded for turn %s.", turn_id, exc_info=True,
                )

            return AgentTurnResult(messages=messages, pagination=pagination)

        finally:
            # Always release, whether this turn succeeded or raised.
            await self.cache.release_lock(key=session_id)