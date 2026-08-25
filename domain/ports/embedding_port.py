from typing import Protocol

class EmbeddingPort(Protocol):
    async def embed(self, text: str) -> list[float]:
        """Convert text into an embedding vector.

        Raises:
            EmbeddingError: if the embedding call fails.
        """
        ...