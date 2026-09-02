"""SQL safety checks, separated from the tool that uses them.

These were defined inside sql_tool_adapter.py, which meant testing them
required constructing an adapter with a database, a cache and an embedding
client. They depend on none of that: given a string, they return a verdict.
Pulling them out makes them directly testable, which matters more here than
almost anywhere else in the codebase - this is the layer that decides whether
a model-written query is allowed to run.

It is the second line of defence, not the first. The agent's Postgres role is
granted SELECT on its own tables and nothing else, so a table that slips past
these checks is still refused by the database. These exist so that a mistake
is reported clearly to the model instead of arriving as a permission error,
and so an implementation bug does not become the only thing standing between
an agent and another agent's data.
"""

from __future__ import annotations

import re

import sqlparse
from sqlparse.sql import Comment, Function, Identifier, IdentifierList, Parenthesis, TokenList
from sqlparse.tokens import Keyword
from sqlparse.tokens import Comment as CommentToken

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Keywords after which a table reference appears. JOIN needs its variants
# spelled out: sqlparse reports "LEFT JOIN" as a single keyword token, so
# matching only the bare word would let `LEFT JOIN members` through.
_TABLE_KEYWORDS = frozenset({
    "FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
    "CROSS JOIN", "LEFT OUTER JOIN", "RIGHT OUTER JOIN", "FULL OUTER JOIN",
    "STRAIGHT_JOIN", "NATURAL JOIN",
})

# Functions a SELECT may not call. None of them reads a table, so the table
# allowlist never sees them, and each one reaches past the query: `set_config`
# and `set_role` change who the connection is (the authenticator is a member
# of every agent role, so `set_config('role', ...)` is a scope change);
# `pg_sleep` holds a pooled connection; the `pg_read_*`/`pg_ls_*`/`lo_*`
# families touch the database host's filesystem; `dblink`/`pg_*_query` open
# a second connection with the server's own credentials. The rolled-back
# transaction in the database adapter would contain the first; the rest are
# refused here because the model has no legitimate reason to call any of
# them and a clear message beats a permission error.
_DENIED_FUNCTIONS = frozenset({
    "set_config", "set_role", "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "lo_import", "lo_export", "lo_get", "lo_put", "dblink", "dblink_exec",
    "dblink_connect", "query_to_xml", "current_setting",
})


def validate_identifier(name: str) -> str:
    """Return `name` if it is a bare SQL identifier, else raise.

    Table and column names cannot be passed as query parameters, so anywhere
    one is interpolated into SQL it has to come through here first. Raising
    rather than returning a flag is deliberate: an invalid identifier at an
    interpolation site is a bug or an injection attempt, and neither should
    continue quietly.
    """
    if not _IDENTIFIER_RE.match(name or ""):
        raise ValueError(f"Invalid identifier: {name!r}")
    return name


def _cte_names(statement: TokenList) -> set[str]:
    """Names defined by a WITH clause.

    A CTE name is not a table - it refers to a subquery that is itself walked
    below - so referencing one must not be treated as reading an unknown
    table. Without this, every aggregate written as a CTE is refused, which
    for an analytics agent is most of them.
    """
    names: set[str] = set()
    seen_with = False
    for token in statement.tokens:
        if token.ttype is Keyword.CTE:          # the WITH itself
            seen_with = True
            continue
        if not seen_with or token.is_whitespace:
            continue
        if isinstance(token, IdentifierList):
            for ident in token.get_identifiers():
                if isinstance(ident, Identifier) and ident.get_real_name():
                    names.add(ident.get_real_name().lower())
        elif isinstance(token, Identifier) and token.get_real_name():
            names.add(token.get_real_name().lower())
        else:
            # Past the CTE definitions and into the main SELECT.
            break
    return names


def _is_derived_table(token) -> bool:
    """Whether this token is a subquery standing where a table name would.

    `FROM (SELECT ...) x` does not arrive as a bare Parenthesis: sqlparse
    wraps it in an Identifier whose real name is the alias, so checking for
    Parenthesis alone reports "table not allowed: x". What is inside gets
    checked by the recursion, so the wrapper itself is skipped.
    """
    if isinstance(token, Parenthesis):
        return True
    if isinstance(token, Identifier):
        for inner in token.tokens:
            if inner.is_whitespace:
                continue
            return isinstance(inner, Parenthesis)
    return False


def _next_meaningful(tokens: list, start: int):
    """The first token after `start` that is neither whitespace nor a comment.

    sqlparse hands a comment back as a `sql.Comment` group (ttype None), and
    a bare comment token carries a sub-type (`Comment.Multiline`), so neither
    is caught by `ttype is Comment` - which let `JOIN /* */ members` past the
    allowlist with the comment standing where the table name was checked.
    """
    for token in tokens[start + 1:]:
        if token.is_whitespace or isinstance(token, Comment) or token.ttype in CommentToken:
            continue
        return token
    return None


def _denied_function(tokens: list) -> str | None:
    """The first denied function called anywhere in the tree, or None."""
    for token in tokens:
        if isinstance(token, Function):
            name = (token.get_real_name() or "").lower()
            if name in _DENIED_FUNCTIONS:
                return name
        if token.is_group:
            found = _denied_function(token.tokens)
            if found:
                return found
    return None


def _check_tokens(tokens: list, allowed) -> str | None:
    """Check one level of the parse tree, then descend into every group.

    Descending is the whole point. An earlier version inspected only the
    statement's top-level tokens, so a disallowed table inside a subquery -
    `... WHERE id IN (SELECT book_id FROM members)` - was never looked at and
    passed as valid.
    """
    for i, token in enumerate(tokens):
        if token.ttype is Keyword and token.value.upper() in _TABLE_KEYWORDS:
            target = _next_meaningful(tokens, i)
            if target is None:
                continue

            # A derived table - FROM (SELECT ...) x. Not a table name; the
            # recursion below checks what is inside it.
            if _is_derived_table(target):
                continue

            candidates = (
                list(target.get_identifiers())
                if isinstance(target, IdentifierList)
                else [target]
            )
            for candidate in candidates:
                if _is_derived_table(candidate):
                    continue
                name = (
                    candidate.get_real_name()
                    if hasattr(candidate, "get_real_name")
                    else candidate.value
                )
                if name and name.lower() not in allowed:
                    return f"Table not allowed: {name}"

        if token.is_group:
            error = _check_tokens(token.tokens, allowed)
            if error:
                return error

    return None


def validate_readonly_query(query: str, allowed_tables) -> str | None:
    """Return None if the query is allowed, else a message explaining why not.

    The message is written to be read by the model, since that is where it
    goes - it should say what was wrong clearly enough to be fixed on the next
    attempt.

    Three things are checked:

      - exactly one statement, so a stacked `SELECT ...; DROP ...` cannot pass
      - the statement is a SELECT
      - every table named after FROM or a JOIN is in `allowed_tables`, at any
        depth: subqueries, derived tables and CTE bodies included
      - no call to a function in `_DENIED_FUNCTIONS`, at any depth

    `allowed_tables` may be any container of lowercase names; membership is
    tested against lowercased table names.
    """
    # Comments carry no meaning to Postgres but they do to sqlparse's grouper:
    # `FROM books, /* */ members` stops being one IdentifierList and the second
    # table is never looked at. Strip them before parsing so the tree that is
    # checked has the shape the rules were written for.
    query = sqlparse.format(query, strip_comments=True)
    statements = [s for s in sqlparse.parse(query) if str(s).strip(" \t\n;")]
    if len(statements) != 1:
        return "Multiple statements are not allowed."

    statement = statements[0]
    if statement.get_type() != "SELECT":
        return "Only SELECT queries are allowed."

    denied = _denied_function(statement.tokens)
    if denied:
        return f"Function not allowed: {denied}"

    scope = {t.lower() for t in allowed_tables} | _cte_names(statement)
    return _check_tokens(statement.tokens, scope)
