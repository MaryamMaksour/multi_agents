from typing import Any

import httpx

from domain.exceptions import UnknownToolError, ToolExecutionError


class HttpDelegateToolAdapter:
    def __init__(self, client: httpx.AsyncClient, tool_urls: dict[str, str], timeout: int = 60):
        self._client = client
        self._tool_urls = tool_urls
        self._timeout = timeout

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

        try:
            response = await self._client.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ToolExecutionError(f"Error {e} while calling {tool_name} at {url}") from e