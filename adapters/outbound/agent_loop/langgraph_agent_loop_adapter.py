from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated
from operator import add as add_messages

from domain.entities.chat_message import ChatMessage, Role, ToolCall
from domain.entities.agent_turn import (
    GAVE_UP_PREFIX,
    AgentTurnResult,
    PaginationState,
)
from domain.exceptions import UnknownToolError, ToolExecutionError
from domain.ports.llm_port import LLMPort
from domain.ports.tool_port import ToolPort
from libs.agent_core.logging_setup import Timer, log_event

import json
import asyncio
import logging

logger = logging.getLogger(__name__)

# How many model calls one question may take before the loop stops on its own.
#
# There was no cap. `_should_continue` returned True for as long as the model
# kept emitting tool calls, so a model that never stops asking - and they do,
# on a schema it cannot make sense of, or after a tool error it keeps
# retrying the same way - ran until LangGraph's own recursion_limit raised
# GraphRecursionError, which is a library exception with nothing in it about
# agents, tools or this question. Every one of those calls was paid for.
#
# Twelve because a real question is about seven: schema, filters, values,
# execute, answer, with room for a correction or two. A question that needs
# more than twelve is not converging, and stopping is the better outcome.
DEFAULT_MAX_ITERATIONS = 12


def _drop_unanswered_tool_calls(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Remove tool calls that never got a result, and messages left empty.

    Written as a sweep over the whole list rather than a special case for the
    last message: the budget is the way this happens today, but "the model
    asked and nothing answered" is the shape of the bug, and a future path
    that produces it should not need to remember this.

    A message that had only the unanswered call and no text is dropped
    entirely - an assistant message with neither content nor tool calls is
    not something to send back to a provider.
    """
    answered = {m.tool_call_id for m in messages if m.role is Role.TOOL}

    kept: list[ChatMessage] = []
    for message in messages:
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            kept.append(message)
            continue

        live = [call for call in message.tool_calls if call.id in answered]
        if len(live) == len(message.tool_calls):
            kept.append(message)
            continue

        if not live and not (message.content or "").strip():
            continue  # nothing left in it

        kept.append(ChatMessage(
            role=message.role, content=message.content,
            tool_calls=live or None, tool_call_id=message.tool_call_id,
            name=message.name, reasoning=message.reasoning,
        ))
    return kept


def _result_shape(result) -> dict:
    """What came back, in a few numbers rather than in full.

    A tool result can be a page of rows, and logging it whole turns one line
    into a screen and puts the data itself in the log - which is a retention
    question as much as a readability one. The row count is what answers "did
    the filter match anything", and zero rows followed by a confident number
    is the failure this system actually has.
    """
    if not isinstance(result, dict):
        return {"result_type": type(result).__name__}

    shape = {}
    rows = result.get("rows")
    if isinstance(rows, list):
        shape["rows"] = len(rows)
    for key in ("row_count", "has_more"):
        if key in result:
            shape[key] = result[key]
    if not shape:
        shape["keys"] = sorted(result)[:8]
    return shape


class _GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pagination: dict[str, PaginationState]
    # Counted in the state rather than on the adapter: one adapter serves
    # every concurrent turn in the process, so an instance attribute would
    # have two questions decrementing each other's budget.
    iterations: int


class LangGraphAgentLoopAdapter:
    def __init__(self, llm: LLMPort, tools: ToolPort, max_pages_per_tool: int = 5,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS):
        self._llm = llm
        self._tools = tools
        self._max_pages_per_tool = max_pages_per_tool
        self._max_iterations = max_iterations
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
            # additional_kwargs is LangChain's own carrier for fields it does
            # not model. Without this the reasoning captured by the LLM
            # adapter is dropped the moment it crosses into the graph, so
            # QWEN_ENABLE_THINKING, the history `reasoning` column and
            # show_history.py's THINKS block all stayed empty - the feature
            # was wired at both ends and severed in the middle.
            return ChatMessage(
                role=Role.ASSISTANT, content=msg.content, tool_calls=tool_calls,
                reasoning=(msg.additional_kwargs or {}).get("reasoning"),
            )

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
                return AIMessage(
                    content=msg.content or "", tool_calls=tool_calls,
                    # Carried, not sent: _to_provider_message never reads it,
                    # so this survives the graph without reaching the model.
                    additional_kwargs=({"reasoning": msg.reasoning}
                                       if msg.reasoning else {}),
                )

           raise ValueError(f"Unsupported  message type: {type(msg)}")
           

    # ----  graph nodes ----
    async def _call_llm(self, state: _GraphState) -> dict:
        iteration = state.get("iterations", 0) + 1
        chat_messages = [self._to_chat_message(message) for message in state["messages"]]

        log_event(logger, "loop.llm", level=logging.DEBUG,
                  iteration=iteration, budget=self._max_iterations,
                  context_messages=len(chat_messages))

        result = await self._llm.achat(chat_messages)
        return {"messages": [self._to_lc_message(result)], "iterations": iteration}

        

    async def _invoke_one_tool(self, tool_call_msg: dict, pagination: dict) -> tuple:
        tool_name = tool_call_msg.get("name")
        args = tool_call_msg.get("args", {})
        page_info = pagination.get(tool_name)

        # The arguments are the most useful line in the whole system: they are
        # where "how many Arabic novels" becomes a query with no genre filter.
        # Logged before the call, so they survive a tool that hangs.
        log_event(logger, "tool.call", tool=tool_name, arguments=args)

        if page_info and page_info.pages_fetched >= self._max_pages_per_tool:
            result = {"error": f"Pagination limit reached: max_pages_per_tool={self._max_pages_per_tool}"}
            log_event(logger, "tool.page_limit", level=logging.WARNING,
                      tool=tool_name, limit=self._max_pages_per_tool)
        else:
            # Every log line here sits *outside* the `with`, and that is not
            # style. Timer.ms is written in __exit__, so a line inside the
            # block reads it before it has been set - every tool.result,
            # tool.error and tool.crashed reported ms=0.0, which is worse than
            # no timing at all because it looks like a measurement.
            event, fields = "tool.result", {}
            level = logging.INFO
            crashed = False

            with Timer() as timer:
                try:
                    result = await self._tools.call_tool(tool_name=tool_name, args=args)
                except (UnknownToolError, ToolExecutionError) as e:
                    # Returned to the model as a result rather than raised, so
                    # it can correct itself - which means without this line it
                    # is invisible. A tool failing every turn looks exactly
                    # like a model choosing not to use it.
                    result = {"error": str(e)}
                    event, level, fields = "tool.error", logging.WARNING, {"error": str(e)}
                except Exception as e:
                    # An adapter that raised something outside the domain's
                    # error types. Same treatment - the turn continues - but
                    # at ERROR with the frame, because it is a bug here rather
                    # than a mistake by the model.
                    result = {"error": f"{type(e).__name__}: {e}"}
                    event, level = "tool.crashed", logging.ERROR
                    fields, crashed = {"error": type(e).__name__}, True
                else:
                    fields = _result_shape(result)

            log_event(logger, event, level=level, tool=tool_name, ms=timer.ms, **fields)
            if crashed:
                logger.error("tool %s raised an unexpected error", tool_name,
                             exc_info=True)

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
        if not getattr(last, "tool_calls", None):
            return False

        if state.get("iterations", 0) >= self._max_iterations:
            # Stop here rather than let LangGraph's recursion_limit raise. The
            # turn keeps everything the model has produced so far, run() adds
            # an honest closing message, and the caller gets an answer that
            # says what happened instead of a 500 naming a graph.
            log_event(logger, "loop.budget_exhausted", level=logging.WARNING,
                      iterations=state.get("iterations", 0),
                      budget=self._max_iterations,
                      pending=[c.get("name") for c in last.tool_calls])
            return False
        return True

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
            "iterations": 0,
        }

        # recursion_limit is LangGraph's own backstop, and it has to sit above
        # ours or it fires first and raises where we would have stopped
        # cleanly. Two graph steps per iteration - llm, then tools - plus room
        # for the final llm call that ends the loop.
        with Timer() as timer:
            state = await self._graph.ainvoke(
                initial_state,
                config={"recursion_limit": self._max_iterations * 2 + 4},
            )

        result_messages = [self._to_chat_message(m)
                           for m in state["messages"][len(messages):]]

        # A tool call with no reply must not leave this method.
        #
        # When the budget stops the loop, it stops *after* the model has asked
        # for another tool and *before* that tool runs - so the last assistant
        # message carries a tool_call nothing ever answered. RunAgentTurn
        # persists these messages into the orchestrator's conversation window,
        # and every provider requires each tool_call to be followed by its
        # result. The next question in that session would be rejected with a
        # 400 before reaching the model, and the one after it, for the three
        # days the window lives. One stopped turn poisoned the session.
        result_messages = _drop_unanswered_tool_calls(result_messages)

        # A turn must end with something a person can read. It normally does -
        # the loop ends when the model answers in prose - but not when the
        # budget stopped it mid-tool-call, and an empty answer is the one
        # outcome that tells the caller nothing at all. Saying what happened
        # is worth more than an empty string, and far more than a 500.
        if not self._has_final_answer(result_messages):
            log_event(logger, "loop.no_answer", level=logging.WARNING,
                      iterations=state.get("iterations", 0),
                      messages=len(result_messages))
            result_messages.append(ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    f"{GAVE_UP_PREFIX} within {self._max_iterations} steps, so "
                    "I do not have a reliable answer. Asking for one thing at "
                    "a time usually gets further."
                ),
            ))

        log_event(logger, "loop.done", ms=timer.ms,
                  iterations=state.get("iterations", 0),
                  messages=len(result_messages))

        return AgentTurnResult(messages=result_messages, pagination=state["pagination"])

    @staticmethod
    def _has_final_answer(messages: list[ChatMessage]) -> bool:
        """Did the loop end on prose, rather than on a tool result?

        The same walk the HTTP edge does to build its response body, which is
        the point: this asks the question the caller is about to ask, before
        the caller gets an empty string.
        """
        for message in reversed(messages):
            if message.role is Role.ASSISTANT and (message.content or "").strip():
                return True
        return False




    