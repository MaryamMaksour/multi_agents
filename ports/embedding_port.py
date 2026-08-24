from typing import Protocol

class EmbeddingPort(Protocol):
    async def embed(self, text: str) -> list[float]:
        ...