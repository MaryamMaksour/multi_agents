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
DROP ROLE IF EXISTS app_catalog;
DROP ROLE IF EXISTS app_circulation;

CREATE ROLE app_catalog     WITH LOGIN PASSWORD 'dev_catalog';
CREATE ROLE app_circulation WITH LOGIN PASSWORD 'dev_circulation';

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

-- Verify: connect as each role and list what it can see.
--
--   psql -U app_catalog -d library_dev -c "
--     SELECT table_name FROM information_schema.tables
--     WHERE table_schema='public' ORDER BY 1;"
--
-- app_catalog sees authors, books, publishers.
-- app_circulation sees books, branches, loans, members.
-- Neither sees the other's tables, and neither needed a list in code.
