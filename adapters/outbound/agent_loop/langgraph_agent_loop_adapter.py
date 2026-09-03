from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated
from operator import add as add_messages

from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.exceptions import UnknownToolError, ToolExecutionError
from domain.ports.llm_port import LLMPort
from domain.ports.tool_port import ToolPort

import json
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class _GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pagination: dict[str, PaginationState]
    turn_id: str | None
    steps: int


class LangGraphAgentLoopAdapter:
    def __init__(self, llm: LLMPort, tools: ToolPort, max_pages_per_tool: int = 5,
                 max_steps: int = 12):
        self._llm = llm
        self._tools = tools
        self._max_pages_per_tool = max_pages_per_tool
        self._max_steps = max_steps
        self._graph = self._build_graph()

    # ----  ChatMessage <-> LangChain BaseMessage ----
    @staticmethod
    def _to_chat_message(msg: BaseMessage) -> ChatMessage:
        if isinstance(msg, HumanMessage):
            return ChatMessage(role=Role.USER, content=msg.content)

        if isinstance(msg, SystemMessage):
            return ChatMessage(role=Role.SYSTEM, content=msg.content)

        if isinstance(msg, ToolMessage):
            return ChatMessage(role=Role.TOOL, content=msg.content, tool_call_id=msg.tool_call_id)

        if isinstance(msg, AIMessage):
            tool_calls = [
                ToolCall(name=call.get("name"), args=call.get("args", {}), id=call.get("id"))
                for call in msg.tool_calls
            ] or None
            return ChatMessage(role=Role.ASSISTANT, content=msg.content, tool_calls=tool_calls)

        raise ValueError(f"Unsupported LangChain message type: {type(msg)}")
        

    @staticmethod
    def _to_lc_message(msg: ChatMessage) -> BaseMessage: 
           if msg.role == Role.SYSTEM:
                  return SystemMessage(content = msg.content)
           
           if msg.role == Role.USER:
                  return HumanMessage(content = msg.content)
           
           if msg.role == Role.TOOL:
                  return ToolMessage(content = msg.content, tool_call_id = msg.tool_call_id)
           
           if msg.role == Role.ASSISTANT:
                tool_calls = [
                    {"name": call.name, "args": call.args, "id": call.id}
                    for call in (msg.tool_calls or [])
                ]
                return AIMessage(content=msg.content or "", tool_calls=tool_calls)

           raise ValueError(f"Unsupported  message type: {type(msg)}")
           

    # ----  graph nodes ----
    async def _call_llm(self, state: _GraphState) -> dict:

        chat_messages = [ self._to_chat_message(message) for message in state["messages"] ]

        started = time.monotonic()
        result = await self._llm.achat(chat_messages)
        logger.info(
            "model answered",
            extra={
                "turn_id": state.get("turn_id"),
                "step": state.get("steps", 0) + 1,
                "messages_sent": len(chat_messages),
                "tool_calls": [call.name for call in (result.tool_calls or [])],
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        )

        ai_messages = self._to_lc_message(result)

        return {"messages": [ai_messages], "steps": state.get("steps", 0) + 1}


    async def _invoke_one_tool(self, tool_call_msg: dict, pagination: dict,
                               turn_id: str | None) -> tuple:
        tool_name = tool_call_msg.get("name")
        page_info = pagination.get(tool_name)

        args = tool_call_msg.get("args", {})
        started = time.monotonic()

        if page_info and page_info.pages_fetched >= self._max_pages_per_tool:
            result = {"error": f"Pagination limit reached: max_pages_per_tool={self._max_pages_per_tool}"}
        else:
            try:
                result = await self._tools.call_tool(
                    tool_name=tool_name, args=args, turn_id=turn_id,
                )
            except (UnknownToolError, ToolExecutionError) as e:
                result = {"error": str(e)}

        # The arguments in full, because "how many Arabic novels under 300
        # pages" arriving as a query with no genre filter is the single line
        # that separates a bad delegation from a bad sub-agent. The formatter
        # in logging_setup keeps configured secrets out of it.
        logger.info(
            "tool called",
            extra={
                "turn_id": turn_id,
                "tool": tool_name,
                # not "args": logging.LogRecord already has one, and `extra`
                # refuses to overwrite it.
                "tool_args": args,
                "error": result.get("error") if isinstance(result, dict) else None,
                "row_count": result.get("row_count") if isinstance(result, dict) else None,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            },
        )

        pagination_update = None
        if isinstance(result, dict) and "has_more" in result:
            pages_fetched = page_info.pages_fetched + 1 if page_info else 1
            pagination_update = (tool_name, PaginationState(
                has_more=result["has_more"],
                next_cursor=result.get("next_cursor"),
                pages_fetched=pages_fetched,
            ))

        tool_message = ToolMessage(
            tool_call_id=tool_call_msg.get("id"),
            name=tool_name,
            content=json.dumps(result, ensure_ascii=False),
        )
        return tool_message, pagination_update


    async def _take_action(self, state: _GraphState) -> dict:
        tool_calls_msgs = state["messages"][-1].tool_calls
        pagination = dict(state.get("pagination", {}))
        turn_id = state.get("turn_id")
        results = await asyncio.gather(
            *(self._invoke_one_tool(call, pagination, turn_id) for call in tool_calls_msgs)
        )

        tool_messages = [tm for tm, _ in results]
        for _, update in results:
            if update:
                key, value = update
                pagination[key] = value

        return {"messages": tool_messages, "pagination": pagination}

    def _should_continue(self, state: _GraphState) -> bool:
        """Whether to run the tools the model just asked for.

        The step budget stops here rather than at the tool call, so a turn
        that hits it ends on the model's own last message instead of on a
        tool result. Without a budget the loop runs until LangGraph's
        recursion limit raises - after paying for every call in between.
        """
        last = state["messages"][-1]
        if state.get("steps", 0) >= self._max_steps:
            logger.warning(
                "step budget reached; answering with what the turn has",
                extra={"turn_id": state.get("turn_id"), "max_steps": self._max_steps},
            )
            return False
        return bool(getattr(last, "tool_calls", None))

    def _build_graph(self):
        graph = StateGraph(_GraphState)
        graph.add_node("llm", self._call_llm)
        graph.add_node("tools", self._take_action)
        graph.add_conditional_edges("llm", self._should_continue, {True: "tools", False: END})
        graph.add_edge("tools", "llm")
        graph.set_entry_point("llm")
        return graph.compile()

    # ---- intrance point ----
    async def run(self, messages: list[ChatMessage], turn_id: str | None = None) -> AgentTurnResult:
        """Raises: LLMRequestError: if the LLM call fails."""
        messages_lc = [self._to_lc_message(m) for m in messages]
        initial_state = {
            "messages": messages_lc,
            "pagination": {},
            "turn_id": turn_id,
            "steps": 0,
        }
        # Two graph nodes run per model call, so a budget above LangGraph's
        # own recursion limit would raise there before _should_continue ever
        # stopped the turn - and the deployment would get a 500 instead of the
        # partial answer the budget exists to produce.
        state = await self._graph.ainvoke(
            initial_state, config={"recursion_limit": 2 * self._max_steps + 2},
        )
        result_messages = [self._to_chat_message(m) for m in state["messages"][len(messages):]]

        return AgentTurnResult(
            messages=self._settle_unanswered_tool_calls(result_messages),
            pagination=state["pagination"],
        )

    @staticmethod
    def _settle_unanswered_tool_calls(messages: list[ChatMessage]) -> list[ChatMessage]:
        """Never end a turn on a tool call nobody answered.

        Stopping at the budget ends on the model's own last message, which is
        a request for tools that were then not run. The orchestrator persists
        the turn, so the next question would replay a tool call with no result
        after it - a conversation an OpenAI-compatible provider rejects.
        """
        if not messages:
            return messages

        last = messages[-1]
        if last.role is not Role.ASSISTANT or not last.tool_calls:
            return messages

        return messages[:-1] + [ChatMessage(
            role=Role.ASSISTANT,
            content=(
                # Whatever the model said alongside the tool call is a
                # preamble ("let me check the loans table"), not an answer, so
                # it is kept but never left standing on its own.
                (f"{last.content}\n\n" if last.content else "")
                + "I stopped before finishing this question: the turn reached "
                "its step budget. Please ask again, more narrowly."
            ),
        )]




    