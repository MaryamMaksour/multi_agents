from __future__ import annotations

import time

from domain.ports.llm_port import LLMPort
from domain.ports.tool_port import ToolPort
from domain.ports.history_port import HistoryPort
from domain.ports.cache_port import CachePort

from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.entities.chat_message import ChatMessage, Role

from domain.exceptions import SessionBusyError, UnknownToolError, ToolExecutionError


class RunAgentTurn:
    def __init__(self,
                 llm: LLMPort,
                 tools: ToolPort,
                 history: HistoryPort,
                 cache: CachePort,
                 system_prompt: str,
                 use_conversation_history: bool = True,
                 session_ttl_seconds: int = 60 * 60 * 24 * 3,  # 3 days
                 max_session_messages: int = 40
                 ):

        self.llm = llm
        self.tools = tools
        self.history = history
        self.cache = cache
        self.system_prompt = system_prompt
        self.use_conversation_history = use_conversation_history
        self.session_ttl_seconds = session_ttl_seconds
        self.max_session_messages = max_session_messages

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

            # Tracks has_more/next_cursor per tool name, so the caller can resume
            # any tool's pagination independently.
            pagination: dict[str, PaginationState] = {}

            # `messages` is what's actually sent to the LLM (system + prior window +
            # this turn). `new_messages` mirrors it but excludes the system prompt -
            # only this is what gets persisted back to the conversation window.
            messages = [ChatMessage(role=Role.SYSTEM, content=system_prompt)] + cache_data + [ChatMessage(role=Role.USER, content=user_input)]
            new_messages = [ChatMessage(role=Role.USER, content=user_input)]

            llm_response = await self.llm.achat(messages=messages)
            messages += [ChatMessage(role=Role.ASSISTANT, content=llm_response.content, tool_calls=llm_response.tool_calls, tool_call_id=llm_response.tool_call_id)]
            new_messages += [ChatMessage(role=Role.ASSISTANT, content=llm_response.content, tool_calls=llm_response.tool_calls, tool_call_id=llm_response.tool_call_id)]

            # Think -> call tool(s) -> feed results back -> think again, until the
            # LLM stops requesting tools and gives a final answer.
            while llm_response.tool_calls:
                for tool_call in llm_response.tool_calls:
                    try:
                        tool_response = await self.tools.call_tool(tool_name=tool_call.name, args=tool_call.args)
                    except (UnknownToolError, ToolExecutionError) as e:
                        # A tool failure is not fatal: it's fed back to the LLM as
                        # a normal result, so it can retry or adjust.
                        tool_response = {"Error": str(e)}

                    if isinstance(tool_response, dict) and "has_more" in tool_response:
                        pagination[tool_call.name] = PaginationState(
                            has_more=tool_response["has_more"],
                            next_cursor=tool_response.get("next_cursor"),
                        )

                    messages += [ChatMessage(role=Role.TOOL, content=str(tool_response), tool_call_id=tool_call.id)]
                    new_messages += [ChatMessage(role=Role.TOOL, content=str(tool_response), tool_call_id=tool_call.id)]

                # Give the LLM the tool results and let it decide: more tools, or done.
                llm_response = await self.llm.achat(messages=messages)
                messages += [ChatMessage(role=Role.ASSISTANT, content=llm_response.content, tool_calls=llm_response.tool_calls, tool_call_id=llm_response.tool_call_id)]
                new_messages += [ChatMessage(role=Role.ASSISTANT, content=llm_response.content, tool_calls=llm_response.tool_calls, tool_call_id=llm_response.tool_call_id)]

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