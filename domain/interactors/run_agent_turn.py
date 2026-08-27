# domain/interactors/run_agent_turn.py
from __future__ import annotations

import time

from domain.ports.agent_loop_port import AgentLoopPort
from domain.ports.history_port import HistoryPort
from domain.ports.cache_port import CachePort

from domain.entities.agent_turn import AgentTurnResult
from domain.entities.chat_message import ChatMessage, Role

from domain.exceptions import SessionBusyError


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
        acquired = await self.cache.acquire_lock(key=session_id, timeout=120)
        if not acquired:
            raise SessionBusyError("Session is currently busy. Please try again later.")

        try:
            # Sub-agents are stateless (every call is a fresh, self-contained
            # request), so they never load a conversation window. The orchestrator
            # does, for multi-turn continuity with the end user.
            cache_data = []
            if self.use_conversation_history:
                cache_data = await self.cache.get(key=session_id)

            # Semantic search over past turns: enrich the prompt with worked
            # examples (question + reasoning trace) similar to this new question.
            memory_examples = await self.history.get_memory(query=user_input)

            # Local variable only - never mutate self.system_prompt, since this
            # instance is shared across every session and every request.
            system_prompt = self.system_prompt + "\n\n" + "History: " + str(memory_examples)

            # Persist the incoming user message for audit + future memory lookups.
            await self.history.log_user_message(session_id=session_id, turn_id=turn_id, message=user_input)

            # `messages` is what's actually sent to the LLM (system + prior window +
            # this turn). `new_messages` mirrors it but excludes the system prompt -
            # only this is what gets persisted back to the conversation window.
            context_window = cache_data[-self.context_messages_sent:]
            messages = [ChatMessage(role=Role.SYSTEM, content=system_prompt)] + context_window + [ChatMessage(role=Role.USER, content=user_input)]
            new_messages = [ChatMessage(role=Role.USER, content=user_input)]

            loop_result = await self.agent_loop.run(messages)
            messages += loop_result.messages
            new_messages += loop_result.messages
            pagination = loop_result.pagination

            # Persist the updated conversation window (system prompt excluded -
            # it's rebuilt fresh every turn), trimmed to the retention limit.
            if self.use_conversation_history:
                updated_window = (cache_data + new_messages)[-self.max_session_messages:]
                await self.cache.set(key=session_id, value=updated_window, ttl=self.session_ttl_seconds)


            # Log the full trace (tool calls + results + final answer) - this is
            # what get_memory() later returns as a worked example, and what marks
            # this turn valid/invalid based on whether an error shows up in it.
            elapsed = time.time() - started_at
            await self.history.log_assistant_final(session_id=session_id, turn_id=turn_id, final_answer=new_messages, elapsed=elapsed)

            return AgentTurnResult(messages=messages, pagination=pagination)

        finally:
            # Always release, whether this turn succeeded or raised.
            await self.cache.release_lock(key=session_id)