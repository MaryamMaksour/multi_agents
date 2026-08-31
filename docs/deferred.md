# Deferred

Decisions and fixes that were raised, understood, and deliberately put off.
Kept here rather than in a conversation so they survive being forgotten.

Each entry says what the thing is, why it was parked, and what has to be true
before picking it up.

---

## Embedding cost, and who pays

**Parked because** the embedding model in use is currently free, so no
decision is being forced yet.

**Comes back when** a deployment brings its own key. At that point the person
running it pays per embedded row, and a backfill over a large table is a real
bill they should see before pressing Run, not after.

Things that will need answering then: what a table's backfill would cost
before it starts, whether to embed the whole table or only a sample first,
and whether re-embedding after a model change is a decision or a default.

**Not deferred with it:** even while free, rate limits mean a large backfill
takes time. Run is therefore a long-running operation regardless of money -
it belongs in the background with visible progress, not in a request. That
part is a design constraint now, not a cost question for later.

---

## A "before" development database, and the column-selection heuristic

**The problem.** `seeds/001_schema.sql` already has `embed_*` columns. That
models the state *after* provisioning, so it tests the query path but not the
path that creates those columns. A client's database will not have them.

**What is blocked on it.** Classification reads the embed columns that already
exist, and that part is finished. Deciding *which* columns deserve one on a
database that has none is a different question, and it has no test data.

The measurements taken on the development database say what the rule should
probably be. Average word count separates every identifier from every
semantic column with no overlap:

| column | distinct | rows | avg words |
|---|---|---|---|
| `books.language` | 3 | 420 | 1.0 |
| `books.genre` | 10 | 420 | 1.1 |
| `books.shelf_code` | 399 | 420 | 1.0 |
| `books.isbn` | 420 | 420 | 1.0 |
| `members.email` | 340 | 340 | 1.0 |
| `authors.name_en` | 28 | 28 | 2.2 |
| `authors.bio` | 10 | 28 | 9.8 |
| `books.summary` | 97 | 420 | 14.1 |

Ratio is not the signal and was dropped: the same `city` column reads as 0.015
in a 340-row table and 0.833 in a 12-row one.

**Settled, and not waiting on this:** what the person is asked. Two options,
in their words - "بحث بالمعنى وكلمات مشابهة" and "بحث دقيق" - never "should
this column be embedded". The heuristic decides, shows the result in plain
language, and the person can change it. It never blocks.

**Comes back when** the provisioning path is built - it needs a schema with
ordinary text columns and no vectors.

---

## `get_lsit_values` is misspelled

**The constraint.** It is a dispatch key in `_handlers`, a name in
`get_tool_schemas`, and a method name. Renaming one without the others turns
every call into an `UnknownToolError` at runtime rather than a failure at
startup, so all three move together or none do.

**It has since spread.** The ENUM guidance in `build_guidance` names the tool
in the sentence the model reads, so the rename is now four places, and there
are tests in `tests/unit/test_filter_classifier.py` and
`tests/security/test_untrusted_input.py` asserting the current spelling on
purpose - a "corrected" name in the guidance alone would send the model to a
tool that does not exist.

**Comes back when** something else is already touching that adapter.

---

## Listing a column's values in the filter guidance

**The idea.** For an ENUM column, telling the model `one of: novel, poetry,
drama` is far more useful than telling it the column has few distinct values -
it matches what is stored rather than what it imagines.

**Parked because** it needs a second call per ENUM column at startup and a cap
on how many values to include, and the classifier is worth getting working
first.

**Comes back after** feature 2 runs end to end and the prompts can be judged.

---

## One sample value in the guidance for TEXT columns

**The idea.** For a high-cardinality text column like `shelf_code` (399
distinct of 420) the values are useless to the model, but the *shape* is not.
Telling it `values look like: NO-12.3` lets it build a sensible
`LIKE 'NO-%'` instead of guessing at the format.

**Comes back after** feature 2, alongside listing values for ENUM columns -
they are the same call and the same cap, just different halves of it.

**Related, and not deferred:** the guidance for a TEXT column should already
say *not* to call the distinct-values tool on it. Models call tools
speculatively, and on `isbn` that returns a 420-value sample which is noise
in the context. That wording belongs in feature 2.

---

## Login

**Parked deliberately**, and left absent rather than faked, so nothing in the
console can be mistaken for authentication that is not there.

**Comes back last.** It is users, hashed passwords and sessions - a feature of
its own, not a field on a form.

---

## `get_memory` filters on a column nothing writes

**The bug, and it is not small.** `PostgresHistoryAdapter.get_memory` selects
past turns `WHERE u.valid = true` for its good examples and `WHERE u.valid =
false` for its bad ones. Nothing anywhere writes `valid`. It stays NULL,
`NULL = true` is NULL, and both halves match no rows.

So the memory feature currently costs an embedding call on every single turn
and returns an empty list, and `RunAgentTurn` appends that empty list to the
system prompt as `"History: []"`.

**Why it is deferred rather than fixed.** The column is described in the code
as "manually-assigned", which means the original design had a person marking
turns good or bad - and that is a product decision, not a bug fix. The options
are a human review step, an automatic rule (a turn whose trace contains an
error is invalid), or dropping the distinction. Picking one without knowing
which is worse than leaving it visible.

**Cheap thing to do first, whichever wins:** stop paying for the embedding
when the query cannot match anything. It is still one embedding call per turn
for a query that returns nothing.

**Related, and already done:** a failed memory lookup no longer takes the turn
down with it. It used to - an embedding model the account could not reach
returned AccessDenied, and because the memory lookup is the first thing a turn
does, every question failed with a 500 including the ones needing no memory at
all.

---

## `get_table_records` is offered to the model and cannot work

**The problem.** The handler runs:

```sql
SELECT row_txt FROM {table} ORDER BY embedding <=> $1::vector LIMIT $2
```

Neither `row_txt` nor `embedding` exists in `seeds/001_schema.sql`, or in any
schema this design produces - embeddings live in `embed_<column>` columns
beside the column they describe. It is a survivor from the old codebase, where
each table had one denormalised text column and one vector for the whole row.

**Half done.** It is no longer declared in `get_tool_schemas`, so the model is
no longer told about a tool that fails on every call. The handler stays in
`_handlers`, and `tests/unit/test_tool_schemas.py` now asserts that gap is
exactly this one tool - withheld on purpose rather than lost.

**What is left:** whether whole-row search is rebuilt against the
`embed_<column>` convention or dropped. Dropping is the smaller change and
probably right - `db_execute` already combines a vector distance with ordinary
predicates in one WHERE clause, which is strictly more useful. Rebuilding is
only worth it if "show me what rows look like" turns out to be something the
model actually needs and cannot get from `get_lsit_values` plus a LIMIT.

---

## `lsit_values` is a dead constructor parameter

`SqlToolAdapter.__init__` takes it and stores it as `self._lsit_values`, and
nothing ever reads it.

**Left in place on purpose,** because it is almost certainly the seam for
"listing a column's values in the filter guidance" above - values read once at
startup and handed to the adapter rather than fetched per call. Removing it
now and adding it back later is churn.

**Comes back with** that entry. If that idea is dropped instead, this goes
with it - and the misspelling carried in the name makes it worth doing
alongside the rename.

---

## No vector index on the history tables

`seeds/004_history.sql` creates a btree on `(turn_id, event_type, created_at)`
and no index on `user_message_embed`, so `get_memory`'s ordering is a
sequential scan over the three-day window.

**Deliberate for now.** An ivfflat index built on an empty table is worse than
none - it needs representative data to pick its lists - and the three-day
window keeps the candidate set small while a deployment is young.

**Comes back when** there is real history to measure, and after the entry
above: indexing a query that currently matches nothing would be optimising a
no-op.

---

## The console cannot register an agent

`ui/app.py` collects agents into Streamlit session state and they are gone
when the tab closes. Everything else on that screen is real.

**This is the design, not a gap.** `AgentRegistryPort` is read-only, because
registration means CREATE ROLE and GRANT, and those need privileges nothing
serving requests should hold. The console is a client; a write path from here
would move that boundary by accident.

**Comes back with the provisioner**, which is a phase 3 component with its own
credentials and its own process - see `docs/roadmap.md`. Until then the
honest console is one that shows drafts and says they are drafts.

---

## An empty embedding column is classified SEMANTIC, and answers wrongly

**The worst-shaped bug found so far**, because it produces a confident wrong
answer rather than an error.

`classify_column` calls a column SEMANTIC when an `embed_<name>` partner
*exists*. It does not ask whether that column contains anything. On a
database where the columns have been created but not backfilled - which is
every database before the backfill runs, including `seeds/` today, where
`002_generate_data.py` fills 420 books and zero embeddings - the guidance
tells the model the column is searchable by meaning.

Then this happens, and it is correct pgvector behaviour rather than a bug in
it. `seeds/001_schema.sql` builds an index on each embed_ column, and a
pgvector index does not index NULL rows, so Postgres plans an index scan for
`ORDER BY col <=> $1` and the unembedded rows are simply absent:

```
Index Scan using idx_books_embed_summary on books
```

Zero rows. No error, no warning. The model reports "there are no books about
the sea" when the truth is "nothing has been embedded". Pinned by
`test_ordering_by_distance_on_an_unembedded_table_returns_nothing`.

**The fix, and it is small:** one `count(embed_col)` per embedding column at
startup - the same shape as the distinct-count probe already there - and a
column whose partner is empty classifies as TEXT or ENUM instead. The
guidance then offers exact matching, which works, rather than semantic
search, which silently cannot.

**Comes back with** the backfill, because the two are the same feature seen
from either end: one fills the columns, this one is honest about them until
it has.

---

## The first real question, and what it found

Run on 2026-08-31 against a real deployment with a real key. Three things
came out of it, and only the third was about the model.

**Two bugs in RedisCacheAdapter**, both fatal on the first question of every
session, both invisible to every unit test. The lock and the session window
shared a key namespace, so locking a session destroyed its window; and a
missing session returned None where the interactor slices a list. Fixed, with
`tests/integration/test_redis_cache_live.py` covering them against a real
Redis - the fake in conftest.py could not have caught either, because a fake
keeps its store and its locks in separate attributes and has no shared
namespace to collide in.

**The script reported `HTTP 500` and nothing else**, which sent the reader
looking for the logs rather than telling them where to look. DomainError now
comes back in the response body.

**Three more bugs, all on the vector path**, found afterwards by running the
whole loop against a scripted model rather than a real one:

- the cache could only hold ChatMessages, so `embed_query_tool` could not
  store a vector at all
- asyncpg has no codec for pgvector's type, so every vector reached it as
  "expected str, got list"
- `log_assistant_final` json.dumps'd ChatMessage entities, so every
  successful turn failed to record its own answer

None was reachable from a unit test, and each is one HTTP call deep. That gap
is now covered by `tests/integration/test_agent_loop_live.py`, which drives
the real system against `tests/fakes/scripted_model.py` - an
OpenAI-compatible endpoint that replies from a script and decides nothing.

---

## A real question has never gone through a real model

Every layer up to the model call runs against real Postgres and Redis in
`tests/integration/test_runtime_live.py`: startup, `SET ROLE`, introspection,
GRANT verification, history checks, tool binding, and the HTTP edge. The model
call itself needs an API key.

**Left uncovered rather than skipped.** A test that skips without a key
reports green for a path nobody ran, which is worse than an obvious hole.

**Comes back the moment a key is available**, and it is the first thing to do
then - not last. Everything downstream of it is currently reasoning.
