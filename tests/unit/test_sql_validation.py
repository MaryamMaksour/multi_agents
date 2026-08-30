"""SQL validation - the layer that decides whether model-written SQL runs.

Worth the most care in this suite. Every case below is something a model
could plausibly produce, either by mistake or because a question steered it
there, and each one has to be refused before it reaches the database.

It is still the second line of defence: an agent's Postgres role is granted
SELECT on its own tables and nothing else, so anything slipping past here is
refused anyway. These tests protect the quality of the refusal - a clear
message the model can act on rather than a permission error - and make sure
a bug here is caught in CI instead of relying on the grant to save it.
"""

from __future__ import annotations

import pytest

from libs.agent_core.sql_validation import validate_identifier, validate_readonly_query

ALLOWED = {"books", "authors", "publishers"}


def allowed(query: str) -> bool:
    return validate_readonly_query(query, ALLOWED) is None


# --------------------------------------------------------------------------
# queries that must be allowed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "SELECT id, title_en FROM books LIMIT $1 OFFSET $2",
    "SELECT count(*) FROM books",
    "SELECT b.title_en FROM books b WHERE b.page_count < 300",
    "SELECT b.title_en, a.name_en FROM books b JOIN authors a ON a.id = b.author_id",
    "SELECT * FROM books, authors",
    "select id from books",
    "SELECT genre, count(*) FROM books GROUP BY genre HAVING count(*) > 5",
    "SELECT id FROM books WHERE id IN (SELECT id FROM authors)",
    "SELECT embed_summary <=> $1::vector AS d FROM books ORDER BY d LIMIT $2 OFFSET $3",
])
def test_allows_legitimate_reads(query):
    assert allowed(query), f"should have been allowed: {query}"


# --------------------------------------------------------------------------
# statements that are not reads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "DELETE FROM books",
    "UPDATE books SET price = 0",
    "INSERT INTO books (id) VALUES (1)",
    "DROP TABLE books",
    "TRUNCATE books",
    "ALTER TABLE books ADD COLUMN x int",
    "CREATE TABLE evil (id int)",
    "GRANT SELECT ON books TO PUBLIC",
])
def test_rejects_anything_that_is_not_a_select(query):
    assert not allowed(query), f"should have been rejected: {query}"


def test_rejection_says_only_selects_are_allowed():
    assert "SELECT" in validate_readonly_query("DELETE FROM books", ALLOWED)


# --------------------------------------------------------------------------
# stacked statements
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "SELECT 1; DROP TABLE books",
    "SELECT id FROM books; DELETE FROM books",
    "SELECT 1; SELECT 2",
])
def test_rejects_stacked_statements(query):
    assert not allowed(query), f"should have been rejected: {query}"


def test_a_trailing_semicolon_is_still_one_statement():
    """Otherwise a perfectly ordinary query gets refused for no reason."""
    assert allowed("SELECT id FROM books;")


# --------------------------------------------------------------------------
# table scoping - the part that keeps agents apart
# --------------------------------------------------------------------------


def test_rejects_a_table_outside_the_allowlist():
    assert not allowed("SELECT * FROM members")


def test_names_the_offending_table_so_the_model_can_correct_itself():
    assert "members" in validate_readonly_query("SELECT * FROM members", ALLOWED)


def test_rejects_a_disallowed_table_second_in_a_from_list():
    """Regression: `FROM a, b` arrives as one IdentifierList, and an earlier
    version only inspected the first name - so the second went unchecked."""
    assert not allowed("SELECT * FROM books, members")


def test_rejects_a_disallowed_table_behind_a_join():
    assert not allowed("SELECT * FROM books b JOIN members m ON m.id = b.id")


def test_rejects_a_disallowed_table_in_a_second_join():
    assert not allowed(
        "SELECT * FROM books b JOIN authors a ON a.id = b.author_id "
        "JOIN loans l ON l.book_id = b.id"
    )


def test_allowlist_matching_is_case_insensitive():
    assert allowed("SELECT * FROM BOOKS")
    assert not allowed("SELECT * FROM MEMBERS")


# --------------------------------------------------------------------------
# identifiers, which cannot be parameterised
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["books", "book_id", "_private", "t1", "B"])
def test_accepts_bare_identifiers(name):
    assert validate_identifier(name) == name


@pytest.mark.parametrize("name", [
    "books; DROP TABLE authors",
    'books"',
    "books--comment",
    "books authors",
    "1books",
    "book-id",
    "public.books",
    "",
    None,
    "books)",
    "*",
])
def test_rejects_anything_that_is_not_a_bare_identifier(name):
    with pytest.raises(ValueError):
        validate_identifier(name)


def test_identifier_rejection_raises_rather_than_returning():
    """It is only called where a value is about to be interpolated into SQL,
    so a falsy return could be used by mistake; raising cannot be ignored."""
    with pytest.raises(ValueError):
        validate_identifier("books; --")


# --------------------------------------------------------------------------
# nesting
#
# These are the cases the first implementation got wrong. It walked only the
# statement's top-level tokens, so a disallowed table inside a subquery was
# never looked at and the query was reported valid - an agent could read
# another agent's tables through a WHERE ... IN (SELECT ...).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "SELECT id FROM books WHERE id IN (SELECT book_id FROM members)",
    "SELECT (SELECT count(*) FROM members) AS c FROM books",
    "SELECT * FROM books WHERE EXISTS (SELECT 1 FROM members)",
    "SELECT * FROM (SELECT * FROM members) x",
    "WITH m AS (SELECT * FROM members) SELECT * FROM m",
    "SELECT * FROM books UNION SELECT * FROM members",
    "SELECT * FROM books WHERE id IN (SELECT id FROM books WHERE id IN (SELECT id FROM members))",
])
def test_rejects_a_disallowed_table_at_any_depth(query):
    assert not allowed(query), f"should have been rejected: {query}"


@pytest.mark.parametrize("query", [
    "SELECT id FROM books WHERE id IN (SELECT id FROM authors)",
    "SELECT * FROM (SELECT * FROM books) x",
    "WITH t AS (SELECT * FROM books) SELECT * FROM t",
    "WITH t AS (SELECT id FROM books), u AS (SELECT id FROM authors) "
    "SELECT * FROM t JOIN u ON t.id = u.id",
])
def test_allows_nesting_when_every_table_is_permitted(query):
    """A CTE or derived-table name is not a table, and refusing it would rule
    out most aggregate queries."""
    assert allowed(query), f"should have been allowed: {query}"


@pytest.mark.parametrize("join", [
    "LEFT JOIN", "RIGHT JOIN", "INNER JOIN", "FULL OUTER JOIN", "CROSS JOIN",
])
def test_rejects_a_disallowed_table_after_any_join_variant(join):
    """sqlparse reports "LEFT JOIN" as one keyword, so matching the bare word
    JOIN would have let every qualified form through."""
    assert not allowed(f"SELECT * FROM books b {join} members m ON m.id = b.id")
