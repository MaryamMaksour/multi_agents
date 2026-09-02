# for both http and sql tools, we can use the same interface for the tool port

from typing import Any, Protocol

class ToolPort(Protocol):
    async def call_tool(self, tool_name: str, args: dict, turn_id: str | None = None) -> Any:
        """Invoke the named tool with the given arguments and return its result.

        `args` is what the model chose. `turn_id` is the turn the call belongs
        to - a correlation value the loop supplies, kept out of `args` so it
        never appears in a tool schema or the model's context. Tools that do
        not propagate it ignore it.

        Raises:
            UnknownToolError: if tool_name is not a recognized tool.
            ToolExecutionError: if the tool itself fails while executing.
        """
        ...