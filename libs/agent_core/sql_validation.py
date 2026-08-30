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
from sqlparse.sql import IdentifierList

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def validate_readonly_query(query: str, allowed_tables) -> str | None:
    """Return None if the query is allowed, else a message explaining why not.

    The message is written to be read by the model, since that is where it
    goes - it should say what was wrong clearly enough to be fixed on the next
    attempt.

    Three things are checked:

      - exactly one statement, so a stacked `SELECT ...; DROP ...` cannot pass
      - the statement is a SELECT
      - every table after FROM or JOIN is in `allowed_tables`

    `allowed_tables` may be any container of lowercase names; membership is
    tested against lowercased table names.
    """
    statements = sqlparse.parse(query)
    if len(statements) != 1:
        return "Multiple statements are not allowed."

    statement = statements[0]
    if statement.get_type() != "SELECT":
        return "Only SELECT queries are allowed."

    for i, token in enumerate(statement.tokens):
        if token.is_keyword and token.value.upper() in ("FROM", "JOIN"):
            next_token = statement.token_next(i)[1]
            if next_token is None:
                continue

            # `FROM table_a, table_b` arrives as one IdentifierList rather than
            # as separate tokens; without this branch only the first name is
            # checked and the rest go unexamined.
            if isinstance(next_token, IdentifierList):
                candidates = list(next_token.get_identifiers())
            else:
                candidates = [next_token]

            for candidate in candidates:
                table_name = (
                    candidate.get_real_name()
                    if hasattr(candidate, "get_real_name")
                    else candidate.value
                )
                if table_name and table_name.lower() not in allowed_tables:
                    return f"Table not allowed: {table_name}"

    return None
