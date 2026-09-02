# Logging

Before this, the system logged from one module and configured nothing. Those
four calls went to the root logger at WARNING and nobody ever saw them. A turn
that answered wrongly left no record of how; a turn that raised left a
traceback with no session, no turn id, and no way to tell which of five
containers produced it.

## Reading one question

```bash
docker compose -f deploy/docker-compose.yml logs -f
```

At the default level, one question reads as a sequence. This is a real run,
trimmed:

```
INFO  [catalog] startup.ready ms=877 kind=sub_agent tables=["authors","books","publishers"] model=qwen-plus
INFO  http.request method=POST path=/run
INFO  [415bbe63] run.received question=How many English novels are under 400 pages?
INFO  [415bbe63] llm.response model=qwen-plus ms=20 finish=tool_calls tool_calls=["get_table_schema"] prompt_tokens=1200 cached_tokens=0 cache_hit_pct=0.0
INFO  [415bbe63] tool.call tool=get_table_schema arguments={"tables": ["books"]}
INFO  [415bbe63] tool.result tool=get_table_schema ms=0 keys=["books"]
INFO  [415bbe63] llm.response model=qwen-plus ms=9 finish=tool_calls tool_calls=["get_filter"] prompt_tokens=1500 cached_tokens=1200 cache_hit_pct=80.0
INFO  [415bbe63] tool.call tool=db_execute arguments={"query": "SELECT count(*) AS n FROM books WHERE language=$1 AND genre=$2 AND page_count<$3 ...", "params": ["English", "novel", 400, 10, 0]}
INFO  [415bbe63] tool.result tool=db_execute ms=3 rows=1 row_count=12 has_more=false
INFO  [415bbe63] turn.done ms=97 total_ms=763 tool_calls=["get_table_schema","get_filter","get_list_values","db_execute"]
INFO  [415bbe63] run.answered ms=768 answer=There are 12 English novels under 400 pages.
INFO  http.response method=POST path=/run status=200 ms=771
```

The line that matters most is `tool.call`. Its `arguments` are where a
question about novels becomes a query with no `genre` filter - and that is
the difference between a bad delegation and a bad sub-agent, which no amount
of re-reading the final number will tell you.

On the orchestrator, `ask.answered` carries the delegated questions for the
same reason.

## Settings

| variable | default | what it does |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` adds every SQL statement, the conversation window sizes, cache hits and the model's reasoning |
| `LOG_FORMAT` | `text` | `json` for anything that ships logs somewhere that parses them |

`INFO` is one line per model call, per tool call and per turn. `DEBUG` is for
when a specific turn needs explaining.

## The ids

Every line carries the agent and a short turn id: `[catalog/7e44c9b7]`. They
are bound once at the HTTP edge and travel by `contextvars`, so a record
emitted at any depth - the SQL adapter, the history write - carries them
without any function taking them as an argument.

The request id crosses process boundaries. The orchestrator sends it to a
sub-agent as `X-Request-ID`, so one question and the two delegated calls it
produced share an id across three containers. A caller that supplies its own
keeps it, which makes the id usable from outside as well; the response echoes
it.

## Secrets

A logger here cannot print the API key or the database password, and that
does not depend on any call site getting it right.

Every record passes through a formatter that replaces the known secret values
- read from the environment at startup - in the message, the interpolated
arguments, the `extra` fields, and the traceback text. The last is the one
that matters: nobody writes `logger.info(key)`, but an httpx exception
carries the request headers and an asyncpg connection error carries the DSN,
and a call site cannot scrub an exception it did not raise.

Patterns are the backstop for secrets this process does not hold - a
`scheme://user:secret@host` in an error body it relayed. Weaker by nature.

`tests/security/test_resource_limits.py` asserts the secret is **absent**
rather than that a placeholder is present: a rule that mangles a line while
leaving the secret elsewhere in it would pass the second check.

## Two things this made possible

**Prompt caching is now a number in every run.** `cache_hit_pct` on
`llm.response` is the provider's own count of the prefix it reused. This
system resends about a thousand tokens of tool schemas on every call in the
loop, seven times a question - whether that is being paid for each time used
to be a question for a billing console.

**Silent degradations are audible.** A tool that fails is handed back to the
model as a result rather than raised, so it used to be invisible; a query the
validator rejected left no trace at all, so a run where the model spent four
of its twelve steps being told its LIMIT was missing looked exactly like a
run where it was thinking. Both log now.

## What logging must never do

`log_event` cannot raise. It shipped in a state where it could: `LogRecord`
reserves the attribute name `args`, so `extra={"args": ...}` raised
`KeyError` - and because `log_event` returns early when the level is
disabled, it only crashed once something else in the run had called
`configure_logging`. Logging that works until logging is switched on is the
worst available arrangement. Colliding field names are renamed now, and the
call is wrapped: a log line that takes down the request it was describing is
worse than a log line that is missing.

## Reading a turn after the fact

The logs are the live view. For a turn that already happened, the trace is in
Postgres:

```bash
python3 scripts/show_history.py --agent orchestrator --turns 5
python3 scripts/show_history.py --turn <uuid>     # both sides of one turn
```

Every turn records its own full message list - tool calls with arguments,
results, and the final answer. Where the model returns a chain of thought, it
is stored too and printed as `THINKS`.
