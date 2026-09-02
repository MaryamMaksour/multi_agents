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
model actually needs and cannot get from `get_list_values` plus a LIMIT.

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

# Resolved

What was parked here and has since been done, with what it turned out to be.
Kept rather than deleted: several of these were worse than the entry that
described them, and that is the part worth remembering.

---

## `get_lsit_values` was misspelled — fixed

The entry called it a rename across four places and waited for something else
to be touching that adapter. It was more than a rename.

The name is read by the model and written back by it, and a model that has
seen "list" a great many more times than "lsit" wrote the correct spelling
often enough to lose a step to `UnknownToolError` each time - a paid-for round
trip spent on a typo. The tool is declared as `get_list_values`; the old
spelling still dispatches, so a conversation window recorded before the rename
replays and a model that writes the old name is answered rather than refused.

---

## `get_memory` filtered on a column nothing writes — fixed

Worse than the entry said. `valid` and `reason` were filtered on and written
by nothing, so all 1,084 rows in the development history had `valid` NULL:
`valid = true` matched nothing, `valid = false` matched nothing, and
`get_memory` returned an empty list for **every** question - after paying for
an embedding call to build a vector it then compared against no rows. The
`DISTINCT ON (reason)` was the same, so even with `valid` set the dedupe key
would have collapsed everything to one row.

`judge()` in the history adapter now writes both, from the trace
`log_assistant_final` is already handed. A turn is valid when it produced a
real answer and no tool failed on the way; `reason` is the sequence of tools
it used, so three examples show three approaches rather than the same one
three times. The verdict lives on the `assistant_final` row, because that is
the row that knows how the turn went.

---

## `lsit_values` was a dead constructor parameter — fixed

Not removed, as the entry expected: made optional and documented as ignored.
The seam it was reserving turned out to exist already - the enum values the
model is shown come from the startup probe through `filters`, and
`_get_list_values` reads the rest from the database. Callers that still pass
it are unaffected; new ones need not know about it.

---

## An empty embedding column classified SEMANTIC — fixed

The one on this list that produced confident wrong answers with no error
anywhere, and the reason it survived so long.

`seeds/002_generate_data.py` creates twelve vector columns and fills none of
them. A pgvector index does not index NULL, so a semantic search returned
zero rows and no error, and the model reported there are none - about a table
holding four hundred rows.

Startup now asks each vector column whether it holds a single value
(`SELECT 1 ... IS NOT NULL LIMIT 1`, which stops at the first row), and a
column that holds nothing stops its partner being classified SEMANTIC. The
partner falls through to TEXT, where ILIKE finds what a vector search could
not, and the log says which columns are affected. A probe that *fails* counts
as filled - being unable to check is not evidence of emptiness.

`scripts/backfill_embeddings.py` fills them, discovering which columns exist
by the same rule `TableSchema.embedding_partner` uses.

---

## Found while doing the above, and fixed

None of these were on this list, because nothing had looked.

**`release_lock` raised when the lock had already expired.** `RunAgentTurn`
releases in a `finally`, so a turn that outlived its own 120-second lease had
its correct answer replaced by "Cannot release a lock that's no longer
owned". Logged now, not raised.

**`has_more` was true for every counting question.** `SELECT count(*) AS n
... LIMIT $1` and its count_query both return 12, so total was 12, the page
held one row, and the model was invited to walk eleven more pages that do not
exist. A page shorter than its limit is the end, whatever the count says.

**The agent loop had no iteration budget.** It ran until LangGraph's own
recursion limit raised `GraphRecursionError`, after paying for every call.
Twelve iterations, then it stops and says so.

**Malformed tool arguments ended the turn.** Models emit invalid JSON exactly
when it costs most - long argument lists, truncation at max_tokens - and the
loop already knows how to let a model correct itself.

**The default model could not be called.** `qwen3-14b` in both `config.py`
and `docker-compose.yml`, and every `qwen3-*` model answers "403 Unpurchased"
on this account, so a fresh checkout failed on its first question.

**`log_event` could not log a field called `args`.** `LogRecord` reserves the
name and raises `KeyError`; it only surfaced once something else in the run
had configured logging, so it was logging that worked until logging was
switched on.

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
