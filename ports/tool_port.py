# for both http and sql tools, we can use the same interface for the tool port

from typing import Any, Protocol

class ToolPort(Protocol):
    async def call_tool(self, tool_name: str, args: dict) -> Any:
        """Call a tool with the given name and input."""
        ...