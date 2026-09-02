---
name: testing-multi-agents
description: How to bring up and end-to-end test the multi-agents NL2SQL stack (Postgres+pgvector, Redis, sub-agent, orchestrator) locally, including a scripted fake LLM when no model credentials exist.
---

# Local end-to-end testing of multi-agents

## Infrastructure
- `docker compose -f deploy/docker-compose.dev.yml up -d` -> pgvector Postgres on :55432 (dev/dev, db library_dev), Redis on :56379.
- Seed in this order (all via `docker exec -i multi_agents_dev_db psql -q -U dev -d library_dev < file`):
  `seeds/001_schema.sql`, `python seeds/002_generate_data.py > /tmp/002_data.sql` then that file,
  `seeds/003_roles.sql`, `seeds/004_history.sql`. The roles file is idempotent; the history tables are only
  granted to `app_authenticator` after 004 runs, so a process fails at startup ("history table") if you skip it.
- Agent roles: app_catalog (authors, publishers, books), app_circulation (branches, members, loans, books).
  Services connect as `app_authenticator` / `dev_authenticator` and `SET ROLE` per pool.

## Running processes
Env needed (config.validate requires PG_* and QWEN_API_KEY to be non-empty; any string works for the key):
```
export PG_HOST=localhost PG_PORT=55432 PG_DBNAME=library_dev PG_USER=app_authenticator PG_PASSWORD=dev_authenticator
export REDIS_URL=redis://localhost:56379/1 AGENTS_REGISTRY_PATH=seeds/agents.example.json
export QWEN_API_KEY=... QWEN_API_URL=http://host/v1 AGENT_URL_TEMPLATE=http://127.0.0.1:8001/run
```
- Sub-agent: `AGENT_KEY=catalog .venv/bin/uvicorn main:app --port 8001` ; orchestrator: no AGENT_KEY, port 8000.
- Startup does NOT call the LLM/embeddings (only introspection + history-table check), so /health works with a
  dummy key. /run, /ask need a chat+embeddings endpoint (embedding dim must be 1024).
- `AGENT_URL_TEMPLATE` has `{key}`; pointing it at a fixed URL routes every delegate to one sub-agent (fine for tests).

## No LLM credentials? Use a scripted OpenAI-compatible stub
A ~60-line FastAPI app serving `POST /v1/chat/completions` and `POST /v1/embeddings` is enough: decide the
scripted tool_call from the tool names in the request (`db_execute` => sub-agent, `catalog` => orchestrator)
and from the count of `role: tool` messages so far. Return 1024-dim random embeddings (not all zeros – cosine
distance on a zero vector errors). Point QWEN_API_URL at it. Be explicit in reports that the model was stubbed.

## Useful checks
- History rows: `SELECT event_type, session_id, turn_id, valid FROM history_catalog ORDER BY id`.
  Sub-agent rows called from the orchestrator have `session_id LIKE 'delegate:%'` and the orchestrator's turn_id.
- `pytest.ini` already has `-q`; run `pytest -o addopts="" -q` to get the "N passed" summary line
  (with `-q -q` it is suppressed). Integration tests skip without the DB.
- Role-leak probe: pool with `max_size=1`, `init=SET ROLE app_catalog`; run the set_config query via
  `PostgresDatabaseAdapter.fetch`, then `SELECT current_user` and `SELECT count(*) FROM members` on the same pool.

## Devin Secrets Needed
- Real model testing: QWEN_API_KEY (DashScope, region-bound to QWEN_API_URL) or any OpenAI-compatible endpoint
  serving chat + 1024-dim embeddings (QWEN_API_URL/QWEN_MODEL/QWEN_EMBED_MODEL, optional EMBED_API_URL/KEY).
