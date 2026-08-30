# Roadmap

What this is becoming, in the order it gets built, and what is actually
finished today.

Kept next to `deferred.md` for the same reason: a plan that lives only in a
conversation is a plan that gets rebuilt from memory every few weeks, badly.

Status is honest. A feature is **done** only when it has an implementation,
tests, and something that can be run against the development database. A
skeleton with a docstring and a `TODO: implement` is **not started**.

---

## The thing being built

A person registers an agent: a name, a description, a prompt, and the tables
it may read. They press Run. The system creates a Postgres role, grants that
role SELECT on exactly those tables, and the agent starts answering questions
about them.

Nothing in the core knows the tables' names, the agents' names, or what any
of them are for. All of that is deployment data.

### The two ideas everything else follows from

**The prompt is untrusted; the GRANT is trusted.** A user writes an agent's
prompt themselves, and it is safe to let them, because no wording can widen
what the role may read. The security boundary is the database's, not the
model's. (One exception, noted where it matters: `description` steers the
orchestrator's routing, so it is not as free as `prompt`.)

**The allowlist derives itself.** `information_schema` reports only the
objects the connected role holds a privilege on. Introspect through an
agent's own role and the answer *is* that agent's scope - there is no list in
code that can drift from the GRANTs. This is what makes a schema-agnostic
core possible at all.

---

## Phase 1 - one agent, answering from a real database

### Feature 1 - introspection · **done**

Read the schema instead of declaring it.

| file | what it is |
|---|---|
| `domain/entities/table_schema.py` | `TableSchema` / `ColumnSchema`, and `embedding_partner()` |
| `domain/ports/schema_port.py` | `list_tables` / `describe` / `distinct_count` |
| `adapters/outbound/schema/postgres_introspection_adapter.py` | the `information_schema` implementation |

Tests: `tests/unit/test_table_schema.py`,
`tests/unit/test_introspection_adapter.py`, and
`tests/integration/test_introspection_live.py`, which is where the security
claim is actually checked - connect as `app_catalog` and only that agent's
three tables come back.

### Feature 2 - filter classification · **done**

Decide how each column can be searched, and say so in a sentence the model
reads. Replaces four hand-maintained lists that the old codebase rescanned on
every call.

| file | what it is |
|---|---|
| `domain/entities/column_filter.py` | `FilterKind`, `ColumnFilter`, `ENUM_MAX_DISTINCT` |
| `libs/agent_core/filter_classifier.py` | `classify_column` / `classify_table` / `build_guidance` - pure, no I/O |
| `libs/agent_core/schema_bootstrap.py` | the half that talks to the database: introspect, probe, return the adapter's arguments |

The precedence chain: a vector column is storage; a column with an
`embed_<name>` partner is semantic; then numeric, then date; then a text
column with few distinct values is an enum; everything else is text.

`count(DISTINCT ...)` is the only call in startup that reads data rather than
the catalogue, so it runs for the smallest set that could be changed by it -
derived by asking the classifier what a column would be with no count, not by
restating the rules.

Tests: `tests/unit/test_filter_classifier.py`,
`tests/unit/test_schema_bootstrap.py`.

### Feature 3 - the agent registry · **done**

Agents as data rather than as code. The port and the entity exist; the
adapter is a skeleton.

| file | what it is |
|---|---|
| `domain/entities/provider_spec.py` | `ProviderSpec`, `AgentStatus`, `is_routable` |
| `domain/ports/agent_registry_port.py` | `get()` / `list_active()` |
| `adapters/outbound/registry/file_agent_registry_adapter.py` | the JSON implementation |
| `seeds/agents.example.json` | the file format, with two real agents in it |

Everything is read and validated once, in the constructor, and validation is
strict: a duplicate key, an unknown status, an unrecognised field, an empty
`allowed_tables` and a `db_role` that is not a bare identifier all refuse to
load. The alternative is a service that starts fine and then fails on
somebody's question.

Two of those refusals are security-shaped rather than tidy: an unrecognised
field is rejected because a misspelt `allowed_tables` would read as "no
restriction", and a `db_role` beginning `pg_` is rejected because Postgres
reserves that prefix for predefined roles, several of which read and write
files on the database host.

Tests: `tests/unit/test_file_agent_registry.py`.

JSON on disk deliberately, not a table. The port is what makes phase 3's
Postgres-backed version a new adapter rather than a rewrite, and writing it
now costs one file.

Read-only on purpose: registration writes go through the provisioner, which
holds elevated credentials and is not reachable from the request path.

### Feature 4 - startup verification · **done**

`libs/agent_core/agent_startup.py`. Introspects through the agent's own role,
compares the answer against the registry's declared `allowed_tables`, and
refuses to start on a mismatch - in **both** directions, because they fail
differently and both are bad:

- the role reads *more* than declared: the registry is the document a person
  consults to answer "what can this agent see", and it is understating it
- the role reads *less* than declared: the agent will fail partway through
  somebody's question, with an error about SQL rather than configuration

An agent that declares no tables is not a failure - it means "whatever the
role can read", which is the design's own answer stated deliberately.

`start_all` is all-or-nothing. A process that starts with three agents of
four answers as though the fourth does not exist, and the person asking
cannot tell that apart from "no data".

Tests: `tests/unit/test_agent_startup.py`,
`tests/integration/test_agent_startup_live.py`.

**Still to build in this area:** the inbound HTTP adapter and the FastAPI
lifespan that owns the per-agent connection pools. `start_agent` takes a
SchemaPort that must already be connected as the agent's role - that is the
one thing it cannot check for itself, and the composition root owns it.

---

## Testing

Four kinds, and they answer different questions.

| where | what it checks |
|---|---|
| `tests/unit/` | behaviour, against fakes - no database, milliseconds |
| `tests/integration/` | the assumptions the fakes encode, against a real Postgres and real least-privilege roles. Skipped when it is unreachable |
| `tests/security/` | adversarial input against the boundaries: injection through the registry, through identifiers, through the SQL the model writes, and the claim that a prompt cannot widen scope |
| `tests/unit/test_port_conformance.py` | that every adapter still matches its port by name, signature and async-ness, and that every module imports |

The integration tests are the ones worth the setup cost, because the security
argument is an assumption about `information_schema` that no fake can check:

```
docker compose -f deploy/docker-compose.dev.yml up -d
docker exec -i multi_agents_dev_db psql -U dev -d library_dev < seeds/001_schema.sql
python3 seeds/002_generate_data.py | docker exec -i multi_agents_dev_db psql -U dev -d library_dev
docker exec -i multi_agents_dev_db psql -U dev -d library_dev < seeds/003_roles.sql
pytest
```

---

## Phase 2 - the orchestrator

One agent answering questions becomes several, with a main agent that routes.

Already written, from the earlier work: `domain/interactors/run_agent_turn.py`,
`adapters/outbound/agent_loop/langgraph_agent_loop_adapter.py`,
`adapters/outbound/tools/http_delegate_tool_adapter.py`,
`adapters/outbound/llm/qwen_llm_adapter.py`.

Not written: the inbound HTTP adapter (`adapters/inbound/` does not exist),
and the composition root that assembles any of it.

The routing decision reads each agent's `description`. Sub-agents keep no
history of their own - the orchestrator resolves every reference before
delegating, so a sub-agent always receives a self-contained question.

---

## Phase 3 - the platform

Where the project stops being one deployment and starts being a product.

**The provisioner.** `CREATE ROLE`, `GRANT SELECT`, and the reverse. Needs
privileges no service handling requests should hold, so it runs as a separate
component outside the request path - the control plane / data plane split.
This is why `AgentStatus` has a `PENDING` state at all: provisioning is
asynchronous, and the orchestrator must not offer a tool for an agent whose
role does not exist yet.

Dropping an agent is `REASSIGN OWNED`, then `DROP OWNED`, then `DROP ROLE` -
`DROP ROLE` fails while the role still holds a privilege. Which is why the
console offers *disable* instead: it takes effect immediately, keeps the
history, and is reversible.

**Embedding provisioning.** A client's database will not have `embed_*`
columns. Creating them, choosing which columns deserve one, and backfilling
is its own feature - and rate limits alone make it long-running, whatever it
costs. See `deferred.md`.

**A registry table**, replacing the JSON file behind the same port. The
adapter it replaces is deliberately naive - it reads once at construction and
has no cache, because the file cannot change under a running process in a way
worth supporting. The Postgres one will need real invalidation, and not a TTL
alone: a newly registered agent that stays invisible for the length of a TTL
looks like a bug to the person who just registered it.

**Login.** Users, hashed passwords, sessions. Deliberately absent rather than
faked, so nothing in the console can be mistaken for authentication that is
not there.

---

## Not on the roadmap, on purpose

The Streamlit console in `ui/`. It is a client, like any other caller of the
orchestrator, and lives outside the hexagon. Every function in `ui/backend.py`
is labelled REAL or STUB, and a stub names the feature that will make it
real - a console that quietly fakes an answer is worse than no console,
because it teaches you the system works when it does not.

Today: the Tables screen is real, the Agents screen keeps drafts in session
state, and the Ask screen returns a hand-written trace.
