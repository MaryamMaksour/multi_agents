"""The prompt must not be able to widen an agent's scope through the database.

The validator checks table names; the GRANT checks table names. Neither, on
its own, stops a query that names only allowed tables and changes *who is
asking*: `SELECT set_config('role', 'app_circulation', false) FROM books`.
The authenticator holds membership of every agent role, so the SET succeeds,
and asyncpg's `RESET ALL` on release does not undo `role` - the pooled
connection stays escalated for whoever gets it next.

Two layers, tested separately, each sufficient on its own:

1. The validator refuses the functions that could do it (and the ones that
   read files, sleep, or open other connections while it is at it).
2. Every fetch runs in a READ ONLY transaction that is always rolled back,
   so a GUC change that gets past the validator dies with the transaction.
"""

from __future__ import annotations

import pytest

from adapters.outbound.db.postgres_db_adapter import PostgresDatabaseAdapter
from domain.exceptions import DatabaseError
from libs.agent_core.sql_validation import validate_readonly_query

ALLOWED = {"books", "authors"}


# --------------------------------------------------------------------------
# layer 1: the validator
# --------------------------------------------------------------------------

ESCALATIONS = [
    ("set_config on role",
     "SELECT set_config('role', 'app_circulation', false) FROM books LIMIT $1 OFFSET $2"),
    ("set_config, upper-cased",
     "SELECT SET_CONFIG('role', 'app_circulation', false) FROM books LIMIT $1 OFFSET $2"),
    ("set_config with a schema prefix",
     "SELECT pg_catalog.set_config('role', 'app_circulation', false) FROM books"),
    ("set_config inside a subquery",
     "SELECT id FROM books WHERE id = (SELECT length(set_config('role', 'x', false)))"),
    ("set_config inside a CTE",
     "WITH x AS (SELECT set_config('role', 'x', false) AS r) SELECT * FROM books, x"),
    ("set_config wrapped in another function",
     "SELECT lower(set_config('role', 'x', false)) FROM books"),
    ("set_role", "SELECT set_role('app_circulation') FROM books"),
    ("current_setting", "SELECT current_setting('is_superuser') FROM books"),
    ("pg_sleep", "SELECT pg_sleep(30) FROM books"),
    ("pg_read_file", "SELECT pg_read_file('/etc/passwd') FROM books"),
    ("pg_ls_dir", "SELECT pg_ls_dir('/') FROM books"),
    ("dblink", "SELECT * FROM dblink('dbname=x', 'SELECT 1') AS t(a int)"),
    ("lo_import", "SELECT lo_import('/etc/passwd') FROM books"),
    ("pg_terminate_backend", "SELECT pg_terminate_backend(1) FROM books"),
]


@pytest.mark.parametrize("what,query", ESCALATIONS, ids=[e[0] for e in ESCALATIONS])
def test_the_validator_refuses_functions_that_change_who_is_asking(what, query):
    reason = validate_readonly_query(query, ALLOWED)
    assert reason is not None, query
    assert "not allowed" in reason.lower()


def test_the_validator_still_allows_the_functions_a_query_needs():
    for query in (
        "SELECT count(*) FROM books",
        "SELECT lower(title_en), coalesce(page_count, 0) FROM books LIMIT $1 OFFSET $2",
        "SELECT title_en FROM books WHERE date_part('year', added_at) = $1",
        "SELECT name_en FROM authors ORDER BY name_embed <=> $1::vector LIMIT $2 OFFSET $3",
        "SELECT string_agg(title_en, ', ') FROM books GROUP BY genre",
    ):
        assert validate_readonly_query(query, ALLOWED) is None, query


# --------------------------------------------------------------------------
# comments cannot hide a table from the validator
# --------------------------------------------------------------------------

HIDDEN = [
    ("block comment before the join target", "SELECT b.id FROM books b JOIN /*c*/ members m ON 1=1"),
    ("line comment before the join target", "SELECT b.id FROM books b JOIN -- c\n members m ON 1=1"),
    ("block comment in a comma list", "SELECT * FROM books, /*c*/ members"),
    ("block comment right after FROM", "SELECT * FROM /*c*/ members"),
    ("comment split across the keyword", "SELECT * FROM books JOIN/**/members ON 1=1"),
]


@pytest.mark.parametrize("what,query", HIDDEN, ids=[h[0] for h in HIDDEN])
def test_a_comment_does_not_hide_a_table(what, query):
    reason = validate_readonly_query(query, ALLOWED)
    assert reason is not None, query
    assert "members" in reason


def test_a_comment_on_an_allowed_query_is_harmless():
    assert validate_readonly_query(
        "SELECT id /* the id */ FROM books -- all of them\n LIMIT $1 OFFSET $2", ALLOWED
    ) is None


# --------------------------------------------------------------------------
# layer 2: the transaction
# --------------------------------------------------------------------------


class RecordingTransaction:
    def __init__(self, log: list, readonly: bool):
        self.log = log
        self.readonly = readonly

    async def start(self):
        self.log.append(("start", self.readonly))

    async def rollback(self):
        self.log.append(("rollback",))

    async def commit(self):
        self.log.append(("commit",))


class RecordingConnection:
    def __init__(self, log: list, rows=None, fail: Exception | None = None):
        self.log = log
        self.rows = rows or []
        self.fail = fail

    def transaction(self, *, readonly: bool = False, **_):
        return RecordingTransaction(self.log, readonly)

    async def fetch(self, query, *params):
        self.log.append(("fetch", query, params))
        if self.fail:
            raise self.fail
        return self.rows


class RecordingPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *exc):
                return False

        return _Acquire()


class Row(dict):
    pass


async def test_fetch_runs_inside_a_read_only_transaction_and_rolls_it_back():
    log: list = []
    conn = RecordingConnection(log, rows=[Row(n=1)])

    rows = await PostgresDatabaseAdapter(RecordingPool(conn)).fetch("SELECT 1", 7)

    assert rows == [{"n": 1}]
    assert log == [
        ("start", True),
        ("fetch", "SELECT 1", (7,)),
        ("rollback",),
    ]


async def test_fetch_rolls_back_even_when_the_query_fails():
    """The failure path matters more than the success path: a query that
    changed `role` and then errored must not leave the connection changed."""
    log: list = []
    conn = RecordingConnection(log, fail=RuntimeError("boom"))

    with pytest.raises(DatabaseError):
        await PostgresDatabaseAdapter(RecordingPool(conn)).fetch("SELECT 1")

    assert log[0] == ("start", True)
    assert log[-1] == ("rollback",)
    assert ("commit",) not in log
