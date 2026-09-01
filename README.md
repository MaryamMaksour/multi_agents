# Multi-agent NL2SQL, on a hexagonal core

A person registers an agent — a name, a description, a prompt, and the tables
it may read. The system creates a Postgres role, grants that role `SELECT` on
exactly those tables, and the agent starts answering questions about them. An
orchestrator routes between agents by their descriptions.

Nothing in the core knows the tables' names, the agents' names, or what any of
them are for. All of that is deployment data.

## The two ideas everything else follows from

**The prompt is untrusted; the GRANT is trusted.** A user writes their own
agent's prompt, and it is safe to let them, because no wording can widen what
its role may read. The security boundary is the database's, not the model's.

**The allowlist derives itself.** `information_schema` reports only the
objects the connected role holds a privilege on. Introspect through an agent's
own role and the answer *is* that agent's scope — there is no list in code that
can drift from the GRANTs.

Both are enforced rather than asserted. `libs/agent_core/agent_startup.py`
compares an agent's declared tables against what its role can really read and
refuses to start on a mismatch, in either direction.

## Running it

```bash
# 1. Infrastructure
docker compose -f deploy/docker-compose.dev.yml up -d

# 2. Schema, data, roles, history tables - in that order
DB="docker exec -i multi_agents_dev_db psql -U dev -d library_dev"
$DB < seeds/001_schema.sql
python3 seeds/002_generate_data.py | $DB
$DB < seeds/003_roles.sql
$DB < seeds/004_history.sql

# 3. The system: one container per agent, plus the orchestrator
export QWEN_API_KEY=...
docker compose -f deploy/docker-compose.yml up --build
```

Ask it something:

```bash
python3 scripts/first_question.py
```

**No key yet?** There is a scripted model that needs none:

```bash
python3 scripts/demo_model.py           # in one terminal
```

It follows a fixed flow - schema, filters, one query, answer - so everything
under it runs for real: tool dispatch, the SQL validator, the least-privilege
role, Postgres, the delegation hop. The number in the answer is the count
Postgres returned; the sentence around it is written in the script. A
demonstration and a development harness, never an evaluation - it cannot tell
you whether a model would choose correctly, only that everything around one
works.

That checks the orchestrator is up and routing before spending a token, asks
three questions, and says where to read the token and cache numbers. A new
DashScope International account gets 1M input and 1M output tokens free for
90 days, which is roughly 190 questions - enough to prove the path works.

Or by hand:

```bash
curl localhost:8000/health          # the orchestrator, and who it routes to
curl localhost:8001/health          # the catalog agent, and its tables

curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"How many Arabic novels under 300 pages?","session_id":"s1"}'
```

`/health` reports the tables each agent actually resolved. A sub-agent that is
up but reading the wrong tables is the failure this design exists to prevent,
so it is visible from outside the process without a query or a log.

### One image, three shapes

`AGENT_KEY` decides what a container becomes. Set it and the process serves
that one agent on `/run`; leave it empty and the process is the orchestrator,
serving `/ask`. Same bytes either way, so there is no per-agent build to keep
in step with the registry.

### The development console

```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

A client like any other — it lives outside the hexagon and calls the same
endpoints. Every function in `ui/backend.py` is labelled REAL or STUB, and a
stub names what will make it real.

## Tests

```bash
pytest                              # unit + security; integration skips
```

Four kinds, answering different questions:

| where | what it checks |
|---|---|
| `tests/unit/` | behaviour against fakes — no services, milliseconds |
| `tests/integration/` | the assumptions those fakes encode, against a real Postgres and real least-privilege roles |
| `tests/security/` | adversarial input against the boundaries, including the claim that a prompt cannot widen scope |
| `tests/unit/test_port_conformance.py` | every adapter still matches its port by name, signature and async-ness; every module imports |

The integration tests are the ones worth the setup, because the security
argument is an assumption about `information_schema` that no fake can check.
They need the database from step 2 above, plus Redis, and skip cleanly without
either.

## Layout

```
domain/          entities, ports, interactors - no I/O, no framework
adapters/
  inbound/http/  FastAPI. Thin: JSON to arguments, exceptions to status codes
  outbound/      Postgres, Redis, the model, HTTP delegation
libs/agent_core/ config, composition root, the classifier, SQL validation
seeds/           schema, data, roles, history - a development deployment
ui/              Streamlit console. A client, outside the hexagon
docs/roadmap.md  what is built, what is next, and what each test kind is for
docs/deferred.md decisions raised, understood, and deliberately put off
```

## Environment

Nothing has a working-looking default. `PG_HOST`, `PG_USER`, `PG_PASSWORD`,
`PG_DBNAME` and `QWEN_API_KEY` are required and checked at startup, together,
so a first deployment names every missing variable at once rather than one
restart at a time.

### Where the model comes from

`QWEN_API_URL` is the seam between the hosted and self-hosted editions. It is
an OpenAI-compatible base URL, so a hosted API, a vLLM server in your own VPC,
or a local Ollama are a configuration change and not a code change.

**DashScope keys are region-bound**, and this is worth knowing before it costs
you an afternoon. A key made in the Beijing console returns
`401 Incorrect API key provided` on the Singapore endpoint and vice versa -
the message says the key is wrong when it is the region that is. The default
here is Singapore; Beijing is `https://dashscope.aliyuncs.com/compatible-mode/v1`
and is substantially cheaper for the same models. To find out which one a key
belongs to:

A key can also authenticate and still reach no models -
`403 AccessDenied.Unpurchased` - which means the account has not activated
Model Studio in that region. `python3 scripts/check_model.py` asks each
candidate directly and reports both whether the key may call it and whether
it returns a tool call at all; a model that answers in prose cannot drive
this system, and fails silently rather than loudly.

```bash
for host in dashscope-intl.aliyuncs.com dashscope.aliyuncs.com; do
  curl -s -o /dev/null -w "$host %{http_code}\n" \
    -H "Authorization: Bearer $QWEN_API_KEY" \
    "https://$host/compatible-mode/v1/models"
done
```

Containers read the environment when they are created, so a corrected key
needs `docker compose up -d --force-recreate`, not a restart.

Embeddings have their own pair, `EMBED_API_URL` and `EMBED_API_KEY`, which
default to the chat endpoint when unset. Anything OpenAI-compatible works on
either side and they need not be the same provider - Cohere's compatibility
endpoint, for instance, serves embeddings under the same shape:

```bash
export EMBED_API_URL=https://api.cohere.ai/compatibility/v1
export QWEN_EMBED_MODEL=embed-multilingual-v3.0    # 1024 dimensions
```

`scripts/check_model.py --embedding <model>` reports the width it returns, so
a mismatch is caught before it becomes a Postgres error mid-question.
 They exist because a vLLM process
serves one model, so self-hosting means two servers - and because the two
workloads have opposite shapes: chat is few calls of many tokens, embeddings
are many calls of few. That also makes the useful hybrid possible: embeddings
local and free, where the call count is high, and a hosted model for
generation, where quality matters most.

#### Prompt caching

Most providers cache by matching the *prefix* of a request, so the system
prompt is sent byte-identical every turn and the volatile parts - the worked
examples from history - go last, next to the question. That is not a
micro-optimisation: the tool schemas alone are ~1,070 tokens resent on every
call in the agent loop, and one question is seven calls. With DashScope's
implicit cache it is about 39% of the bill.

Nothing needs enabling; it is the message order that makes it possible.

`EMBEDDING_DIM` must match the `vector(N)` columns. The seed uses 1024;
changing it is a migration and a re-embedding of every row, not a restart.

### Postgres

`PG_USER` is the *authenticator* — one login for the whole service, holding no
privileges of its own, that each pool turns into an agent with `SET ROLE`. It
is `NOINHERIT`, which is what makes that a narrowing rather than a formality:
before it becomes somebody it can read no agent's data at all.

## Python environment

```bash
conda create -n multi-agents python=3.11
conda activate multi-agents
pip install -r requirements.txt
```
