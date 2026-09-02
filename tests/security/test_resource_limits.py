"""What a caller can spend on this system's behalf, and where a secret can go.

The existing security tests attack the boundaries that decide *what may be
read*: identifiers reaching SQL, the model's queries, the GRANT underneath
them. This file covers the two that decide *what may be spent* and *what may
escape*, which fail quietly rather than loudly and so are easier to leave out.

Cost is a security property here in a way it is not in an ordinary web
service. One request is an embedding call, several model calls priced per
token, and rows that stay for the retention window - so an endpoint with a
floor and no ceiling is not merely unvalidated input, it is somebody else's
bill. The limits are generous; the point is that they exist.

Secrets are the other half. The service holds an API key and a database
password, and the ways they reach an output are not ones a review catches: an
httpx exception carries request headers, a connection error carries the DSN,
and a 500 body carries whatever the exception said.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from adapters.inbound.http.schemas import (
    MAX_CURSOR_CHARS,
    MAX_QUESTION_CHARS,
    MAX_SESSION_ID_CHARS,
    AskRequest,
    DelegateContext,
    RunRequest,
)
from libs.agent_core import logging_setup
from libs.agent_core.logging_setup import JsonFormatter, TextFormatter, redact


@pytest.fixture(autouse=True)
def _isolated_secrets(monkeypatch):
    monkeypatch.setattr(logging_setup, "_known_secrets", set())
    yield


# --------------------------------------------------------------------------
# what a caller may spend
# --------------------------------------------------------------------------

def test_an_enormous_question_is_refused_at_the_edge():
    """An empty question is a mistake; a four-megabyte one is a bill.

    It would be embedded, sent to the model as the last message of a prompt
    that already carries the tool schemas, kept in the conversation window and
    written to history. The endpoint had a floor and no ceiling.
    """
    with pytest.raises(ValidationError):
        AskRequest(question="x" * (MAX_QUESTION_CHARS + 1), session_id="s1")


def test_the_same_limit_applies_to_the_sub_agent_endpoint():
    """/run is reachable in its own right - in the development compose file
    its port is published - so a limit only on /ask guards the front door of a
    building with two."""
    with pytest.raises(ValidationError):
        RunRequest(user_input="x" * (MAX_QUESTION_CHARS + 1), session_id="s1")


def test_a_real_question_is_nowhere_near_the_limit():
    """A limit that rejects real use gets raised until it stops protecting
    anything. Several pages of Arabic, well inside it."""
    question = "كم رواية عربية عندنا أقل من ٣٠٠ صفحة؟ " * 100
    assert len(question) < MAX_QUESTION_CHARS
    AskRequest(question=question, session_id="s1")


def test_an_unbounded_session_id_is_refused():
    """It becomes a Redis key and a history column."""
    with pytest.raises(ValidationError):
        AskRequest(question="q", session_id="s" * (MAX_SESSION_ID_CHARS + 1))


def test_an_unbounded_cursor_is_refused():
    """Issued by this system and handed back unchanged - so the cap is on what
    a caller may return, not on what was ever produced."""
    with pytest.raises(ValidationError):
        DelegateContext(cursor="c" * (MAX_CURSOR_CHARS + 1))


def test_an_empty_question_is_still_refused():
    """The floor that was already there, kept."""
    with pytest.raises(ValidationError):
        AskRequest(question="", session_id="s1")


# --------------------------------------------------------------------------
# what may escape
# --------------------------------------------------------------------------

def test_the_api_key_cannot_be_logged_by_any_route(monkeypatch):
    """Message, interpolated argument, extra field, traceback.

    Four routes because in practice the key arrives by the fourth: nobody
    writes logger.info(key), but an httpx exception carries the request
    headers and a call site cannot scrub what it did not raise.
    """
    key = "sk-a-real-looking-key-000111222"
    monkeypatch.setenv("QWEN_API_KEY", key)
    monkeypatch.setattr(logging_setup, "_configured", False)
    logging_setup.configure_logging(force=True)

    formatter = JsonFormatter()

    def render(**kw):
        record = logging.LogRecord("t", logging.ERROR, __file__, 1,
                                   kw.pop("msg", "m"), kw.pop("args", ()), kw.pop("exc", None))
        for name, value in kw.items():
            setattr(record, name, value)
        return formatter.format(record)

    assert key not in render(msg=f"authorization: Bearer {key}")
    assert key not in render(msg="provider said %s", args=(key,))
    assert key not in render(api_key=key)

    try:
        raise RuntimeError(f"401 with key {key}")
    except RuntimeError:
        import sys
        assert key not in render(msg="call failed", exc=sys.exc_info())


def test_the_database_password_cannot_be_logged(monkeypatch):
    """The DSN in an asyncpg connection error is the realistic path."""
    monkeypatch.setenv("PG_PASSWORD", "dev_authenticator")
    monkeypatch.setattr(logging_setup, "_configured", False)
    logging_setup.configure_logging(force=True)

    line = ("connection to server failed: "
            "postgresql://app_authenticator:dev_authenticator@postgres:5432/library_dev")
    assert "dev_authenticator" not in redact(line)


def test_an_unregistered_credential_is_still_caught():
    """A secret this process does not hold - another service's DSN in an error
    body it relayed. Weaker by nature, which is why it is the backstop and not
    the defence."""
    assert "SomeOtherPassword1" not in redact(
        "upstream said: mysql://svc:SomeOtherPassword1@db.internal:3306/x")


def test_redaction_does_not_eat_the_schema():
    """Redaction that mangles ordinary output is redaction someone turns off,
    and then none of the above holds."""
    line = ("db.query sql=SELECT count(*) FROM books WHERE language='Arabic' "
            "AND genre='novel' AND page_count<300")
    assert redact(line) == line


@pytest.mark.parametrize("formatter", [TextFormatter(), JsonFormatter()])
def test_both_formatters_redact(formatter):
    """One of the two forgetting is the failure mode a single-formatter test
    would miss, and the deployment picks the format."""
    logging_setup.register_secret("sk-a-real-looking-key-000111222")
    record = logging.LogRecord("t", logging.INFO, __file__, 1,
                               "key sk-a-real-looking-key-000111222", (), None)
    assert "sk-a-real-looking-key-000111222" not in formatter.format(record)


def test_a_cursor_this_system_issues_is_accepted_back():
    """A cap that rejects what the system itself produced.

    A cursor carries the query's resolved parameters, and a semantic query's
    parameter is a 1024-dimension vector: 21,463 characters as a pgvector
    literal, still ~13,600 after compression, because real embeddings are
    random floats and do not compress. At 8,000 the second page of any
    semantic search came back 422 from /run - and only the second page, since
    nothing smaller ever reached the limit.
    """
    import random

    from adapters.outbound.tools.sql_tool_adapter import _encode_cursor
    from libs.agent_core.pgvector import to_vector_literal

    random.seed(1)
    vector = to_vector_literal([random.gauss(0, 0.05) for _ in range(1024)])
    cursor = _encode_cursor({
        "offset": 10,
        "resolved_params": [vector, 10, 0],
        "query": "SELECT id FROM books ORDER BY embed_summary <=> $1 "
                 "LIMIT $2 OFFSET $3",
        "count_query": "SELECT count(*) FROM books",
        "count_params": [],
    })

    assert len(cursor) > 8_000, "the case this guards has stopped being real"
    DelegateContext(cursor=cursor)   # must not raise


def test_the_cursor_cap_still_stops_an_arbitrarily_large_one():
    """Raised to match the decoder's own ceiling, not removed."""
    with pytest.raises(ValidationError):
        DelegateContext(cursor="c" * (MAX_CURSOR_CHARS + 1))
