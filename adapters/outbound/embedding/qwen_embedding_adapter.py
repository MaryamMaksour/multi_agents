from openai import AsyncOpenAI

from domain.exceptions import EmbeddingError


class QwenEmbeddingAdapter:
    def __init__(self, client: AsyncOpenAI, model: str):
        self._client = client
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Convert text into an embedding vector.

        Raises:
            EmbeddingError: if the embedding call fails.
        """
        try:
            response = await self._client.embeddings.create(model=self._model, input=text)
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"Error {e} while embedding text") from e