-- Per-agent history tables, and the service's access to them.
--
-- Created here rather than by the service at startup, and that is the whole
-- point of the file. Creating a table is DDL, and the process that answers
-- questions must not hold DDL rights - it is the same argument that keeps
-- CREATE ROLE and GRANT in the provisioner. A service that can CREATE TABLE
-- can also, given the right bug, create one somewhere it should not.
--
-- In a real deployment the provisioner runs this for an agent at the moment
-- it registers one, alongside CREATE ROLE and the GRANTs. Here it is a seed,
-- because the two development agents are fixed.
--
-- One table per agent, not one shared table with an agent column. History is
-- read back by get_memory, which is a vector search over past turns, so a
-- shared table means the catalogue agent learning from the circulation
-- agent's questions. Isolation by table needs nothing to be remembered
-- correctly.
--
-- Run after 003_roles.sql:
--   psql -d library_dev -f seeds/004_history.sql

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
DECLARE
    agent_table text;
BEGIN
    -- history_orchestrator is here too: the orchestrator keeps the
    -- conversation with the person, which is the history that has to survive
    -- between their questions.
    FOREACH agent_table IN ARRAY ARRAY[
        'history_catalog', 'history_circulation', 'history_orchestrator'
    ]
    LOOP
        EXECUTE format($ddl$
            CREATE TABLE IF NOT EXISTS %I (
                id                 BIGSERIAL PRIMARY KEY,
                session_id         TEXT NOT NULL,
                turn_id            UUID NOT NULL,
                event_type         TEXT NOT NULL,
                payload            JSONB NOT NULL DEFAULT '{}'::jsonb,
                -- Must match EMBEDDING_DIM. Changing it is a migration and a
                -- re-embedding of every row, not a restart.
                user_message_embed vector(1024),
                valid              BOOLEAN,
                reason             TEXT,
                time               DOUBLE PRECISION,
                created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
            )$ddl$, agent_table);

        -- get_memory joins user rows to their assistant_final by turn_id and
        -- filters on a three-day window, so this is the index it actually
        -- uses. No vector index: the window keeps the candidate set small,
        -- and an ivfflat index built on an empty table is worse than none.
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON %I (turn_id, event_type, created_at)',
            agent_table || '_turn_idx', agent_table);

        -- The service writes and reads history; the agent roles do not. An
        -- agent role holds SELECT on its own tables and nothing else, and
        -- giving it INSERT anywhere would widen exactly the privilege this
        -- design spends its effort narrowing.
        --
        -- No DELETE and no UPDATE. History is a record of what happened, and
        -- a process that can rewrite it is a process whose audit trail means
        -- less than it appears to.
        EXECUTE format('GRANT SELECT, INSERT ON %I TO app_authenticator', agent_table);
        EXECUTE format('GRANT USAGE ON SEQUENCE %I TO app_authenticator',
                       agent_table || '_id_seq');
    END LOOP;
END
$$;

-- The service needs to resolve the table names it was granted.
GRANT USAGE ON SCHEMA public TO app_authenticator;

-- Verify:
--
--   psql -U app_authenticator -d library_dev -c "
--     SELECT table_name, privilege_type FROM information_schema.table_privileges
--     WHERE grantee = 'app_authenticator' ORDER BY 1, 2;"
--
-- shows SELECT and INSERT on the three history tables, and nothing else.
