"""Fill the vector columns the seeds created and left empty.

seeds/002_generate_data.py writes 420 books, 28 authors and every vector
column beside them - and fills none of the vectors. That is not a cosmetic
gap. A pgvector index does not index NULL, so

    SELECT title_en FROM books ORDER BY embed_summary <=> $1 LIMIT 10

returns **zero rows and no error** against a table with four hundred rows in
it, and the model reports that there are none. Every semantic search on this
database has been doing that.

Startup now detects it: a vector column holding nothing stops its partner
being advertised as searchable, the column classifies as TEXT instead, and
the log says so. That makes the answers honest. This script makes the feature
work.

    python3 scripts/backfill_embeddings.py --dry-run     what would be filled
    python3 scripts/backfill_embeddings.py               fill everything
    python3 scripts/backfill_embeddings.py --table books --column summary
    python3 scripts/backfill_embeddings.py --limit 50    a cheap first pass

Which columns exist is discovered, not listed: every `embed_<name>` column
whose `<name>` is a text column in the same table. So a schema that grows a
new embedded column is covered without editing this file - the same rule
TableSchema.embedding_partner uses, which is what keeps the two in step.

Costs one embedding call per row per column, batched. Rows already filled are
skipped, so re-running after a failure resumes rather than repeats.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg  # noqa: E402

from libs.agent_core import config  # noqa: E402
from libs.agent_core.logging_setup import configure_logging  # noqa: E402
from libs.agent_core.pgvector import to_vector_literal  # noqa: E402

EMBED_PREFIX = "embed_"

# How many rows to embed per request. The provider caps a batch, and a larger
# one is not faster past the point where the request itself dominates - but a
# smaller one turns 420 rows into 420 round trips.
BATCH = 32


async def embeddable_columns(conn) -> list[tuple[str, str, str]]:
    """(table, source column, vector column) for everything fillable.

    Derived from the catalogue by the same rule the classifier uses: a vector
    column named embed_<name> belongs to the column <name> in its own table.
    A vector column with no such partner is skipped rather than guessed at -
    user_message_embed on the history tables is one, and it is written by the
    service on every turn, not backfilled here.
    """
    rows = await conn.fetch("""
        SELECT c.table_name, c.column_name, c.udt_name
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
        ORDER BY c.table_name, c.ordinal_position
    """)

    by_table: dict[str, dict[str, str]] = {}
    for row in rows:
        by_table.setdefault(row["table_name"], {})[row["column_name"]] = row["udt_name"]

    pairs = []
    for table, columns in by_table.items():
        for name, udt in columns.items():
            if udt != "vector" or not name.startswith(EMBED_PREFIX):
                continue
            source = name[len(EMBED_PREFIX):]
            if columns.get(source) in ("text", "varchar", "bpchar"):
                pairs.append((table, source, name))
    return pairs


async def pending_count(conn, table: str, source: str, vector: str) -> int:
    """Rows with something to embed and no embedding yet.

    `source IS NOT NULL` as well: a row with no summary has nothing to embed,
    and counting it as pending would make the script look like it never
    finishes.
    """
    return await conn.fetchval(
        f'SELECT count(*) FROM "{table}" '
        f'WHERE "{vector}" IS NULL AND "{source}" IS NOT NULL'
    )


async def fill(conn, embeddings, table: str, source: str, vector: str,
               limit: int | None, dry_run: bool) -> int:
    pending = await pending_count(conn, table, source, vector)
    target = pending if limit is None else min(limit, pending)

    print(f"{table}.{source} -> {vector}: {pending} rows to fill"
          + (f", doing {target}" if target != pending else ""))
    if dry_run or not target:
        return 0

    done = 0
    while done < target:
        rows = await conn.fetch(
            f'SELECT id, "{source}" AS text FROM "{table}" '
            f'WHERE "{vector}" IS NULL AND "{source}" IS NOT NULL '
            f'ORDER BY id LIMIT $1',
            min(BATCH, target - done),
        )
        if not rows:
            break

        # One call per row rather than one call per batch: EmbeddingPort takes
        # a single string, and going around it here would mean this script and
        # the service embedding text two different ways - which is exactly how
        # a backfill ends up in a different vector space from the queries that
        # search it.
        vectors = [await embeddings.embed(row["text"]) for row in rows]

        await conn.executemany(
            f'UPDATE "{table}" SET "{vector}" = $1::vector WHERE id = $2',
            [(to_vector_literal(v), row["id"]) for v, row in zip(vectors, rows)],
        )
        done += len(rows)
        print(f"  {done}/{target}", end="\r", flush=True)

    print(f"  {done}/{target} done      ")
    return done


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", help="only this table")
    parser.add_argument("--column", help="only this source column")
    parser.add_argument("--limit", type=int, help="at most this many rows per column")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be filled, call nothing")
    args = parser.parse_args()

    configure_logging()

    # Writes, so it connects as the owner rather than as an agent role - an
    # agent holds SELECT and could not do this, which is the point of the
    # split. Defaults to the development superuser; PG* override it.
    user = os.getenv("BACKFILL_PG_USER") or os.getenv("PG_USER") or "dev"
    password = os.getenv("BACKFILL_PG_PASSWORD") or os.getenv("PG_PASSWORD") or "dev"
    host = os.getenv("PG_HOST") or "localhost"
    port = int(os.getenv("PG_PORT") or "55432")
    database = os.getenv("PG_DBNAME") or "library_dev"

    try:
        conn = await asyncpg.connect(host=host, port=port, database=database,
                                     user=user, password=password)
    except Exception as e:
        print(f"Cannot reach Postgres at {user}@{host}:{port}/{database}: {e}",
              file=sys.stderr)
        return 1

    embeddings = None
    if not args.dry_run:
        if not config.EMBED_API_KEY:
            print("No embedding key. Set EMBED_API_KEY (or QWEN_API_KEY, which "
                  "it falls back to), or pass --dry-run to see what is missing.",
                  file=sys.stderr)
            await conn.close()
            return 1

        from openai import AsyncOpenAI

        from adapters.outbound.embedding.qwen_embedding_adapter import (
            QwenEmbeddingAdapter,
        )

        embeddings = QwenEmbeddingAdapter(
            AsyncOpenAI(api_key=config.EMBED_API_KEY, base_url=config.EMBED_API_URL),
            config.QWEN_EMBED_MODEL,
        )

        # One call before touching any rows, to fail on a wrong model or a
        # wrong width now rather than after four hundred paid-for calls. A
        # vector of the wrong length is rejected by the column, and the error
        # names the vector rather than the model that produced it.
        probe = await embeddings.embed("dimension check")
        if len(probe) != config.EMBEDDING_DIM:
            print(f"{config.QWEN_EMBED_MODEL} returns {len(probe)} dimensions but "
                  f"the columns are vector({config.EMBEDDING_DIM}). Set "
                  f"EMBEDDING_DIM to match, or use a model that returns "
                  f"{config.EMBEDDING_DIM}.", file=sys.stderr)
            await conn.close()
            return 1

    try:
        pairs = await embeddable_columns(conn)
        if args.table:
            pairs = [p for p in pairs if p[0] == args.table]
        if args.column:
            pairs = [p for p in pairs if p[1] == args.column]

        if not pairs:
            print("No embeddable columns matched.")
            return 0

        total = 0
        for table, source, vector in pairs:
            total += await fill(conn, embeddings, table, source, vector,
                                args.limit, args.dry_run)

        if args.dry_run:
            print("\nDry run. Nothing was called and nothing was written.")
        else:
            print(f"\nFilled {total} embeddings. Restart the agents so startup "
                  "reclassifies these columns as semantically searchable.")
    finally:
        await conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
