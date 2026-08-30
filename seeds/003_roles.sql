-- One least-privilege Postgres role per agent, for the development schema.
--
-- This is the security boundary. The SQL validator in application code
-- rejects non-SELECT statements and tables outside an agent's list, but that
-- is application code and application code has bugs. A role that physically
-- cannot see another agent's tables still cannot see them when the validator
-- is wrong.
--
-- It also removes a source of drift. information_schema only reports tables
-- the current user holds some privilege on, so introspecting through the
-- agent's own role returns exactly its readable tables - the GRANTs below are
-- the only place the agent-to-table mapping is written down. There is no
-- second list in Python to fall out of sync with this one.
--
-- Run after 001_schema.sql and 002_data.sql:
--   psql -d library_dev -f seeds/003_roles.sql

-- Development passwords. Real deployments provision roles through the
-- provisioner component, which generates its own credentials.
--
-- DROP ROLE alone is not enough and this file used to get that wrong: it
-- worked once, on a fresh database, and failed on every re-run with
--
--   ERROR: role "app_catalog" cannot be dropped because some objects
--   depend on it
--   DETAIL: privileges for table authors
--
-- A role cannot be dropped while it still holds a privilege, and the GRANTs
-- below are privileges. DROP OWNED BY revokes every grant made *to* the role
-- and drops everything it owns, which is what makes this idempotent.
--
-- The same sequence, for the same reason, is what the provisioner will need
-- to remove an agent for real - there it is REASSIGN OWNED first, because a
-- production role may own objects worth keeping. These roles own nothing:
-- they hold SELECT and nothing else.
DO $$
DECLARE
    role_name text;
BEGIN
    FOREACH role_name IN ARRAY ARRAY['app_catalog', 'app_circulation', 'app_authenticator']
    LOOP
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
            EXECUTE format('DROP OWNED BY %I', role_name);
            EXECUTE format('DROP ROLE %I', role_name);
        END IF;
    END LOOP;
END
$$;

CREATE ROLE app_catalog     WITH LOGIN PASSWORD 'dev_catalog';
CREATE ROLE app_circulation WITH LOGIN PASSWORD 'dev_circulation';

-- The authenticator. One login for the whole service, holding no privileges
-- of its own; each connection does SET ROLE to the agent it is serving.
--
-- NOINHERIT is the entire point and not a detail. Without it the
-- authenticator would hold the union of every agent's privileges the moment
-- it connects, and every agent would be able to read every other agent's
-- tables before SET ROLE narrowed anything. With it, membership grants only
-- the *right to become* those roles - so a connection that has not run SET
-- ROLE can read nothing, and one that runs RESET ROLE goes back to being
-- able to read nothing.
--
-- The alternative is one password per agent role, which means one secret per
-- registered agent. That does not scale past a handful and makes registering
-- an agent a credential-distribution problem.
CREATE ROLE app_authenticator WITH LOGIN NOINHERIT PASSWORD 'dev_authenticator';

-- WITH INHERIT FALSE is stated rather than left to the role's NOINHERIT, and
-- the difference is not cosmetic. Since PostgreSQL 16 each membership carries
-- its own inherit_option, captured from the member role's NOINHERIT at the
-- moment of the GRANT. Relying on that means the security property depends on
-- the order of two lines in this file: move the GRANT above the CREATE, or
-- add an agent with a plain GRANT later, and you get an inheriting membership
-- and an authenticator that reads everything - with nothing in the SQL
-- looking wrong.
--
-- Requires PostgreSQL 16 or newer, which deploy/docker-compose.dev.yml pins.
GRANT app_catalog, app_circulation TO app_authenticator WITH INHERIT FALSE;

GRANT USAGE ON SCHEMA public TO app_catalog, app_circulation;

-- catalog agent
GRANT SELECT ON authors, publishers, books TO app_catalog;

-- circulation agent
-- Note it also gets books: answering "which titles are overdue" needs the
-- join, and this is exactly the kind of overlap a registry has to express.
-- It does NOT get authors or publishers.
GRANT SELECT ON branches, members, loans, books TO app_circulation;

-- A runaway query guard at the database level, independent of any limit the
-- application applies.
ALTER ROLE app_catalog     SET statement_timeout = '30s';
ALTER ROLE app_circulation SET statement_timeout = '30s';
ALTER ROLE app_authenticator SET statement_timeout = '30s';

-- Verify: connect as each role and list what it can see.
--
--   psql -U app_catalog -d library_dev -c "
--     SELECT table_name FROM information_schema.tables
--     WHERE table_schema='public' ORDER BY 1;"
--
-- app_catalog sees authors, books, publishers.
-- app_circulation sees books, branches, loans, members.
-- Neither sees the other's tables, and neither needed a list in code.
--
-- And the authenticator, before it becomes anybody:
--
--   psql -U app_authenticator -d library_dev -c "
--     SELECT table_name FROM information_schema.tables
--     WHERE table_schema='public' ORDER BY 1;"
--
-- returns nothing at all. That empty result is the NOINHERIT above doing its
-- job, and it is what makes SET ROLE a narrowing rather than a formality.
