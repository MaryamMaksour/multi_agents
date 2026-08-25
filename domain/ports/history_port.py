from typing import Any, Protocol


class HistoryPort(Protocol):

    async def log_user_message(self, session_id: str, turn_id: str,message: str) -> None:
        """Log a user message to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        ...

    async def log_assistant_final(self, session_id: str, turn_id: str, final_answer: Any, elapsed: float) -> None:
        """Log the final assistant message (including its reasoning trace) to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        ...

    async def log_tool_call(self, session_id: str, turn_id: str, tool_name: str, input_data: Any, output_data: Any) -> None:
        """Log a tool call to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        ...

    async def log_sql_query(self, session_id: str, turn_id: str, query: str, params: Any) -> None:
        """Log an SQL query to the history.

        Raises:
            HistoryError: if the history call fails.
        """
        ...

    async def get_memory(self, query: str) -> list[dict]:
        """Retrieve the memory from the history.

        Raises:
            HistoryError: if the history call fails.
        """
        ...

    async def ensure_schema(self) -> None:
        """Ensure the history schema is set up correctly.

        Raises:
            HistoryError: if the history call fails.
        """
        ...

