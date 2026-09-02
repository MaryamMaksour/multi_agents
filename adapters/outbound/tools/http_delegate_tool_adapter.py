import logging
from typing import Any

import httpx

from domain.exceptions import UnknownToolError, ToolExecutionError
from libs.agent_core.logging_setup import Timer, current_context, log_event

logger = logging.getLogger(__name__)


class HttpDelegateToolAdapter:
    def __init__(self, client: httpx.AsyncClient, tool_urls: dict[str, str],
                 tool_descriptions: dict[str, str], timeout: int = 60):
        self._client = client
        self._tool_urls = tool_urls
        self._tool_descriptions = tool_descriptions
        self._timeout = timeout

    def get_tool_schemas(self) -> list[dict]:
        """Describe the delegate tools in the format the LLM adapter passes on.

        One tool per sub-agent, built from _tool_urls, so registering an agent
        is enough to make it callable - there is no second list to update.

        The descriptions are the orchestrator's whole basis for routing, which
        is why they arrive through the constructor rather than being written
        here: they belong to the deployment, alongside the URLs. A missing one
        is raised rather than defaulted, since a placeholder description does
        not fail loudly - it quietly makes an agent unroutable.

        Note what is NOT declared: session_id and turn_id. call_tool reads
        them from args, but they are correlation values the agent loop injects,
        not choices the model should make. Leaving them out of the schema is
        what keeps them out of the model's context, which is the problem the
        old codebase had to strip back out afterwards.
        """
        missing = sorted(set(self._tool_urls) - set(self._tool_descriptions))
        if missing:
            raise ValueError(
                f"No description for delegate tool(s): {', '.join(missing)}. "
                "The orchestrator routes on these, so it cannot be left empty."
            )

        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self._tool_descriptions[name],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": (
                                    "A complete, self-contained question for this agent. "
                                    "It has no memory of this conversation and cannot see "
                                    "the user's other messages, so resolve every reference "
                                    "before sending: name the entity instead of 'it', and "
                                    "state the period instead of 'last year'."
                                ),
                            },
                            "cursor": {
                                "type": "string",
                                "description": (
                                    "Leave unset for a first request. To get the next page, "
                                    "call this same tool again with the same query and the "
                                    "next_cursor value returned previously, exactly as given. "
                                    "Never invent or edit a cursor."
                                ),
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
            for name in self._tool_urls
        ]

    async def call_tool(self, tool_name: str, args: dict) -> Any:
        """Invoke the named tool with the given arguments and return its result.

        Raises:
            UnknownToolError: if tool_name is not a recognized tool.
            ToolExecutionError: if the tool itself fails while executing.
        """
        url = self._tool_urls.get(tool_name)
        if url is None:
            raise UnknownToolError(f"Unknown tool: {tool_name}")

        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolExecutionError(f"{tool_name}: missing or invalid query")

        payload = {
            "session_id": args.get("session_id", ""),
            "user_input": query,
            "context": {
                "cursor": args.get("cursor"),
                "turn_id": args.get("turn_id"),
            },
        }

        # The question as delegated, which is the orchestrator's real output.
        # An answer of 129 to a question about novels is either this line
        # having dropped the word, or the sub-agent having ignored it - and
        # nothing else in the system distinguishes those two.
        log_event(logger, "delegate.ask", agent=tool_name, question=query)

        try:
            with Timer() as timer:
                response = await self._client.post(
                    url, json=payload, timeout=self._timeout,
                    # The id crosses the process boundary here, so one
                    # question and the sub-agent calls it produced share an id
                    # across three containers.
                    headers={"x-request-id": current_context().get("request_id", "")},
                )
                response.raise_for_status()
                body = response.json()
        except Exception as e:
            log_event(logger, "delegate.failed", level=logging.ERROR,
                      agent=tool_name, url=url, error=type(e).__name__)
            logger.error("delegated call to %s failed", tool_name, exc_info=True)
            raise ToolExecutionError(f"Error {e} while calling {tool_name} at {url}") from e

        log_event(logger, "delegate.answer", agent=tool_name, ms=timer.ms,
                  answer=body.get("answer") if isinstance(body, dict) else None,
                  turn_id=body.get("turn_id") if isinstance(body, dict) else None)
        return body