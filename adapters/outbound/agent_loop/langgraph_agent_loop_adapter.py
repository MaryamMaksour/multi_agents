from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated
from operator import add as add_messages

from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.entities.agent_turn import AgentTurnResult, PaginationState
from domain.exceptions import UnknownToolError, ToolExecutionError, LLMRequestError
from domain.ports.llm_port import LLMPort
from domain.ports.tool_port import ToolPort

import json
import asyncio


class _GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pagination: dict[str, PaginationState]


class LangGraphAgentLoopAdapter:
    def __init__(self, llm: LLMPort, tools: ToolPort, max_pages_per_tool: int = 5):
        self._llm = llm
        self._tools = tools
        self._max_pages_per_tool = max_pages_per_tool
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
 
        result = await self._llm.achat(chat_messages)

        ai_messages = self._to_lc_message(result)
  
        return {"messages": [ai_messages]}

        

    async def _invoke_one_tool(self, tool_call_msg: dict, pagination: dict) -> tuple:
        tool_name = tool_call_msg.get("name")
        page_info = pagination.get(tool_name)

        if page_info and page_info.pages_fetched >= self._max_pages_per_tool:
            result = {"error": f"Pagination limit reached: max_pages_per_tool={self._max_pages_per_tool}"}
        else:
            try:
                result = await self._tools.call_tool(tool_name=tool_name, args=tool_call_msg.get("args", {}))
            except (UnknownToolError, ToolExecutionError) as e:
                result = {"error": str(e)}

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
        results = await asyncio.gather(*(self._invoke_one_tool(call, pagination) for call in tool_calls_msgs))

        tool_messages = [tm for tm, _ in results]
        for _, update in results:
            if update:
                key, value = update
                pagination[key] = value

        return {"messages": tool_messages, "pagination": pagination}

    def _should_continue(self, state: _GraphState) -> bool:
        last = state["messages"][-1]
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
    async def run(self, messages: list[ChatMessage]) -> AgentTurnResult:
        """Raises: LLMRequestError: if the LLM call fails."""
        messages_lc = [self._to_lc_message(m) for m in messages]
        initial_state = {
            "messages": messages_lc,
            "pagination": {},
        }
        state = await self._graph.ainvoke(initial_state)
        result_messages = [self._to_chat_message(m) for m in state["messages"][len(messages):]]

        return AgentTurnResult(messages=result_messages, pagination=state["pagination"])




    