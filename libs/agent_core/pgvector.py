"""Passing a vector to Postgres.

asyncpg has no codec for pgvector's `vector` type - it is an extension type,
not a built-in - so handing it a Python list raises

    invalid input for query argument $1: [-0.053, 0.014, ...] (expected str, got list)

which reads like a type mistake in the caller rather than a missing codec.
pgvector's text input format is a bracketed, comma-separated list, and every
query here already casts with `$n::vector`, so a string is all that is needed.

One function rather than a `str(...)` at each call site, because there are
four of them across two adapters and they have to agree: a stray space or a
value rendered in scientific notation is a runtime error at the far end of an
embedding call, not a failure anyone sees while writing the line.

The alternative is registering a type codec on every connection, which is
less to remember at each call site and more to get wrong once - the codec has
to be installed on the pool that runs the query, and this codebase has pools
created in three places.
"""

from __future__ import annotations

from typing import Sequence


def to_vector_literal(values: Sequence[float]) -> str:
    """Render an embedding in pgvector's text input format.

    `repr` on a float round-trips exactly, which matters: a vector written
    with fewer digits than it was computed with is a different vector, and
    the distances it produces are quietly slightly wrong rather than absent.
    """
    if not values:
        # pgvector rejects an empty vector too, with "vector must have at
        # least 1 dimension" - true, and it does not say where the missing
        # vector was meant to come from. Usually an expired token: the model
        # embedded a phrase, the cache entry aged out, and the lookup handed
        # back nothing.
        raise ValueError(
            "No vector to send. The embedding call returned nothing, or a "
            "vector token was resolved after its cache entry expired."
        )

    return "[" + ",".join(repr(float(v)) for v in values) + "]"
