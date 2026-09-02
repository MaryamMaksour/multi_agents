"""Embeddings, behind EmbeddingPort.

Two callers, and they fail differently. `get_memory` embeds the question to
find worked examples, and a failure there costs a thinner prompt - the
interactor degrades rather than raising. `embed_query_tool` embeds a phrase
the model wants to search semantically, and a failure there is the answer.

Both are logged, because the difference between "the embedding endpoint is
unreachable" and "semantic search returned nothing" is invisible from the
answer, and the first has been the cause twice.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from domain.exceptions import EmbeddingError
from libs.agent_core.logging_setup import Timer, log_event

logger = logging.getLogger(__name__)


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
            with Timer() as timer:
                response = await self._client.embeddings.create(
                    model=self._model, input=text)
            vector = response.data[0].embedding
        except Exception as e:
            log_event(logger, "embed.error", level=logging.ERROR,
                      model=self._model, error=type(e).__name__)
            logger.error("the embedding call failed", exc_info=True)
            raise EmbeddingError(f"Error {e} while embedding text") from e

        # The width is here for one reason: it must equal EMBEDDING_DIM, which
        # is fixed in the column as vector(N). A model that returns 1536 where
        # the column takes 1024 fails at the INSERT, several layers away, with
        # a message about a vector rather than about a model. Logged on every
        # call so the mismatch is visible from the first one.
        log_event(logger, "embed.ok", level=logging.DEBUG,
                  model=self._model, dim=len(vector), chars=len(text), ms=timer.ms)
        return vector
