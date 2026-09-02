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

    # log_tool_call and log_sql_query were declared here and implemented by
    # the Postgres adapter, and nothing ever called either one.
    #
    # The port is where their absence matters most: a port is the contract
    # every adapter must satisfy, so two methods nobody calls are two methods
    # every future adapter has to write. And they were redundant -
    # log_assistant_final receives the turn's whole message list, tool calls
    # and results and SQL together, and stores it as the single row that
    # get_memory reads back.

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

