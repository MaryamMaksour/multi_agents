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

## `get_memory` and `valid`

**Resolved with the automatic rule.** `log_assistant_final` now writes `valid`
on the `assistant_final` row: true when no tool result in the trace is an
`{"error": ...}` dict, false otherwise. `get_memory` reads `f.valid` from that
row and deduplicates on `COALESCE(reason, turn_id)` so untagged turns are not
collapsed into a single example. The human-review option stays open - `reason`
is still only ever written by hand.

**Still open: scope.** `get_memory` is per agent (one history table each) but
not per session. Any session's question and trace can be shown to any other
session's model as a worked example. With a single operator that is the point;
with real users it is a data leak and needs a `session_id` (or tenant) filter
before Login lands.

**Less urgent than it was.** The examples get_memory would have supplied are
now written into the sub-agent prompt as a fixed method - see
`SUB_AGENT_METHOD` in libs/agent_core/prompts.py. That covers the case
get_memory could never have covered anyway: a new deployment has no history,
so an agent was at its worst on the first question anybody asked it.

**Related, and already done:** a failed memory lookup no longer takes the turn
down with it. It used to - an embedding model the account could not reach
returned AccessDenied, and because the memory lookup is the first thing a turn
does, every question failed with a 500 including the ones needing no memory at
all.

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

## Two structurally identical questions, answered differently

Asked "how many Arabic novels under 300 pages" and "how many English novels
under 400 pages" in one session, the model applied the `genre` filter to the
first and not to the third. 10 is right for the first; 127 is
`language = 'English'` with no genre at all, where the answer is 12.

Both remaining failures correlate with position in the session - the language
rule also held on question one and broke on question two - which points at
the conversation window rather than at the prompt wording. That is a
hypothesis and not yet a finding.

**What was missing to tell:** which component was wrong. The orchestrator
rewrites the user's question into a self-contained one before delegating, and
that rewrite can drop a constraint - send "how many English books" and the
agent answers it correctly. From outside, an orchestrator that dropped the
word and an agent that ignored the column look identical.

`/ask` now returns the delegated questions, so the next run says which. Until
then, changing either component is guessing.

---

## The first real answers, and the two things they got wrong

Three questions, all answered from real data, every number checked against
the database and correct: 127 English books under 400 pages, 129 Arabic books
under 300, and 28 authors with more than three books - down to the individual
counts per author.

Two errors, both in wording rather than plumbing, and both now addressed.

**"رواية" was read as "book".** The model filtered on `language` and left
`genre` out of the query, then reported the result as a count of novels. The
number was right for a question nobody asked - 129 books, where 10 are
novels. `novel` is one of ten values in that column, and nothing had ever
shown the model that. Fixed by listing an enum column's actual values in its
guidance, which is the item that used to sit here as "listing a column's
values" and is now done.

**An English question was answered in Arabic**, because the previous question
in the same session had been Arabic and the conversation window outweighed
the instruction. The orchestrator prompt now says the language rule applies
to the question being answered now, and that earlier turns are context rather
than an instruction about wording.

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
