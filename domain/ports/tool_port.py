# for both http and sql tools, we can use the same interface for the tool port

from typing import Any, Protocol

class ToolPort(Protocol):
    async def call_tool(self, tool_name: str, args: dict) -> Any:
        """Invoke the named tool with the given arguments and return its result.

        Raises:
            UnknownToolError: if tool_name is not a recognized tool.
            ToolExecutionError: if the tool itself fails while executing.
        """
        ...