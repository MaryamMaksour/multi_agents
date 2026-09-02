"""Logging, and the one property of it that is a security control.

Most of this file is about redaction, and the reason is that redaction is the
only part of logging whose failure is not merely annoying. A missing field
makes debugging slower; a printed API key makes the key worthless. So the
tests here are adversarial: they assert the secret is *absent*, not that the
placeholder is present, because a rule that mangles a line while leaving the
secret elsewhere in it would pass the second check and fail the first.

The rest covers what the formatters promise - context ids on every record,
`extra` fields rendered in both formats, and a formatter that survives values
JSON cannot encode, since the line most likely to carry one is the line
reporting an error.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import uuid
from datetime import datetime, timezone

import pytest

from libs.agent_core import logging_setup
from libs.agent_core.logging_setup import (
    JsonFormatter,
    TextFormatter,
    Timer,
    bind,
    configure_logging,
    context,
    current_context,
    log_event,
    new_request_id,
    redact,
    register_secret,
)


@pytest.fixture(autouse=True)
def _isolated_secrets(monkeypatch):
    """A fresh secret set per test.

    The registry is process-global on purpose - a secret registered anywhere
    must be unprintable everywhere - which makes it exactly the kind of state
    that leaks between tests if it is not reset.
    """
    monkeypatch.setattr(logging_setup, "_known_secrets", set())
    yield


def emit(formatter: logging.Formatter, level: int = logging.INFO,
         msg: str = "event", args=(), exc_info=None, **extra) -> str:
    """Format one record without touching global logging state."""
    # logging resolves exc_info=True to sys.exc_info() before a record is
    # built, so a helper that constructs records directly has to do the same
    # or hand the formatter a bool where it expects a triple.
    if exc_info is True:
        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="test", level=level, pathname=__file__, lineno=1,
        msg=msg, args=args, exc_info=exc_info,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return formatter.format(record)


# --- redaction: known values ---------------------------------------------

def test_registered_secret_is_removed_wherever_it_appears():
    register_secret("sk-abcdef0123456789")
    for line in ("key=sk-abcdef0123456789",
                 "the value sk-abcdef0123456789 in prose",
                 '{"authorization": "Bearer sk-abcdef0123456789"}'):
        assert "sk-abcdef0123456789" not in redact(line)


def test_short_values_are_not_registered():
    """Registering "dev" would redact that substring from every line forever.

    The protection is worse than the exposure: a three-character password is
    a password to change, not one to hide from our own logs at the cost of
    making them unreadable.
    """
    register_secret("dev")
    assert redact("user=dev host=development") == "user=dev host=development"


def test_overlapping_secrets_leave_no_readable_tail():
    """Longest first, or the shorter replacement splits the longer secret and
    leaves the remainder of it printed."""
    register_secret("abcdefgh")
    register_secret("abcdefgh_ijklmnop")
    assert "ijklmnop" not in redact("token=abcdefgh_ijklmnop")


# --- redaction: patterns, with nothing registered -------------------------
# These are the backstop: a secret this process does not hold, in a string it
# did not construct - a provider's error body, another service's DSN.

@pytest.mark.parametrize("line, secret", [
    ("postgresql://app:UnregisteredPw99@postgres:5432/db", "UnregisteredPw99"),
    ("redis://:AnotherPw123@redis:6379/1", "AnotherPw123"),
    ("mongodb+srv://u:p4ssw0rd@cluster0.net/db", "p4ssw0rd"),
    ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghij", "eyJhbGciOiJIUzI1NiJ9"),
    ('{"api_key":"sk-live-9f8e7d6c5b4a3210"}', "sk-live-9f8e7d6c5b4a3210"),
    ("PASSWORD = hunter2secret", "hunter2secret"),
    ("api-key: cohere_2LNRK7WuKILbnGz5LVxc8", "cohere_2LNRK7WuKILbnGz5LVxc8"),
])
def test_pattern_removes_unregistered_secret(line, secret):
    assert secret not in redact(line)


def test_url_password_rule_keeps_the_url_intact():
    """The rule that replaced group 2 with the placeholder while dropping
    group 3 shipped once: it ate the '@' and, for an unregistered password,
    left the password itself in place."""
    out = redact("postgresql://app_user:UnregisteredPw99@postgres:5432/library")
    assert "UnregisteredPw99" not in out
    assert out == "postgresql://app_user:[redacted]@postgres:5432/library"


@pytest.mark.parametrize("line", [
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "SELECT count(*) FROM books WHERE genre='novel' AND page_count<300",
    "postgres://postgres:5432/library_dev",
    "connected to redis://redis:6379/1",
    "answered in 812ms with 1071 prompt tokens",
])
def test_ordinary_lines_survive_untouched(line):
    """Redaction that eats normal output is redaction people turn off."""
    assert redact(line) == line


def test_redaction_applies_to_the_traceback(caplog):
    """Where a secret most often is, and the one place a call site cannot
    scrub it: the exception was raised by a library."""
    register_secret("dev_authenticator")
    try:
        raise RuntimeError("dsn=postgresql://u:dev_authenticator@h/db refused")
    except RuntimeError:
        rendered = emit(JsonFormatter(), level=logging.ERROR,
                        msg="startup failed", exc_info=True)
    assert "dev_authenticator" not in rendered
    assert "startup failed" in json.loads(rendered)["msg"]


def test_redaction_applies_to_interpolated_arguments():
    register_secret("sk-abcdef0123456789")
    rendered = emit(TextFormatter(), msg="provider said: %s",
                    args=("sk-abcdef0123456789",))
    assert "sk-abcdef0123456789" not in rendered


def test_redaction_applies_to_extra_fields():
    """A field is as printable as a message, and `extra={"key": key}` is the
    easiest of all these mistakes to make."""
    register_secret("sk-abcdef0123456789")
    for formatter in (TextFormatter(), JsonFormatter()):
        assert "sk-abcdef0123456789" not in emit(formatter, api_key="sk-abcdef0123456789")


def test_redact_handles_empty_input():
    assert redact("") == ""


# --- context -------------------------------------------------------------

def test_context_ids_reach_the_record():
    with context(agent="catalog", turn_id="abc123", session_id="s1"):
        rendered = json.loads(emit(JsonFormatter()))
    assert rendered["agent"] == "catalog"
    assert rendered["turn_id"] == "abc123"
    assert rendered["session_id"] == "s1"


def test_context_restores_the_outer_values():
    """Sub-agent inside orchestrator: the inner bind must not relabel the
    orchestrator's remaining lines."""
    with context(agent="orchestrator"):
        with context(agent="catalog"):
            assert current_context()["agent"] == "catalog"
        assert current_context()["agent"] == "orchestrator"
    assert "agent" not in current_context()


def test_empty_values_are_not_bound():
    """An id that is not known yet must be absent, not present and blank -
    `turn_id=""` in a log line reads as a turn whose id is the empty string."""
    with context(agent="", turn_id="t1"):
        assert current_context() == {"turn_id": "t1"}


def test_unset_context_leaves_the_keys_out():
    assert "turn_id" not in json.loads(emit(JsonFormatter()))


def test_new_request_id_is_unique_and_short():
    ids = {new_request_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(len(i) == 12 for i in ids)


# --- formatters ----------------------------------------------------------

def test_json_line_is_one_object_per_line():
    rendered = emit(JsonFormatter(), msg="turn.start", question_chars=42)
    assert "\n" not in rendered
    assert json.loads(rendered)["question_chars"] == 42


def test_json_encodes_values_json_cannot():
    """asyncpg hands back UUIDs and datetimes. A formatter that raises on one
    loses the line that was reporting the problem."""
    rendered = emit(JsonFormatter(), turn=uuid.uuid4(),
                    at=datetime.now(timezone.utc))
    assert isinstance(json.loads(rendered)["turn"], str)


def test_text_shows_agent_and_short_turn():
    with context(agent="catalog", turn_id="7e44c9b7-74bc-4b25-88c8-3acae768c2f6"):
        rendered = emit(TextFormatter(), msg="llm.call", model="qwen-max")
    assert "[catalog/7e44c9b7]" in rendered
    assert "model=qwen-max" in rendered


def test_text_field_values_stay_on_one_line():
    """A newline inside a field would split one event across two log lines,
    and the second would look like an event with no context."""
    rendered = emit(TextFormatter(), sql="SELECT 1\nFROM books")
    assert rendered.count("\n") == 0


def test_text_truncates_a_long_field():
    rendered = emit(TextFormatter(), rows="x" * 5000)
    assert len(rendered) < 500


def test_no_colour_when_not_a_terminal():
    assert "\033[" not in emit(TextFormatter(colour=False), level=logging.ERROR)


# --- setup ---------------------------------------------------------------

def test_configure_logging_is_idempotent(monkeypatch):
    """Called from the app lifespan and from every script; twice must not
    double every line."""
    monkeypatch.setattr(logging_setup, "_configured", False)
    configure_logging(force=True)
    before = len(logging.getLogger().handlers)
    configure_logging()
    configure_logging()
    assert len(logging.getLogger().handlers) == before == 1


def test_configure_logging_registers_environment_secrets(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "sk-from-the-environment-1234")
    monkeypatch.setenv("PG_PASSWORD", "pg-password-from-env")
    monkeypatch.setenv("REDIS_URL", "redis://:redis-pw-from-env@redis:6379/1")
    monkeypatch.setattr(logging_setup, "_configured", False)
    configure_logging(force=True)

    for secret in ("sk-from-the-environment-1234", "pg-password-from-env",
                   "redis-pw-from-env"):
        assert secret not in redact(f"value is {secret} here")


def test_log_event_respects_the_level(monkeypatch):
    logger = logging.getLogger("test.level")
    logger.setLevel(logging.WARNING)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(TextFormatter())
    logger.handlers = [handler]
    logger.propagate = False

    log_event(logger, "debug.event", level=logging.DEBUG)
    log_event(logger, "warning.event", level=logging.WARNING)

    assert "debug.event" not in stream.getvalue()
    assert "warning.event" in stream.getvalue()


def test_bind_returns_tokens_that_undo_it():
    tokens = bind(agent="catalog")
    assert current_context()["agent"] == "catalog"
    logging_setup.unbind(tokens)
    assert "agent" not in current_context()


# --- timing --------------------------------------------------------------

def test_timer_measures_a_block():
    with Timer() as timer:
        sum(range(10_000))
    assert timer.ms > 0


def test_timer_is_zero_before_the_block_ends():
    timer = Timer()
    assert timer.ms == 0.0


# --- the failure mode that logging itself can cause ----------------------

def test_a_reserved_field_name_does_not_raise():
    """`extra={"args": ...}` raises KeyError inside logging, and the words it
    reserves are the words a call site wants: a tool call has args, a tool has
    a name, a module has a module.

    This shipped. It was survivable only because log_event returns early when
    the level is disabled, so the crash appeared once something else in the
    run had called configure_logging - logging that works until logging is
    switched on. The rename happens here so no call site has to know the list.
    """
    logger = logging.getLogger("test.reserved")
    logger.setLevel(logging.INFO)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(TextFormatter())
    logger.handlers = [handler]
    logger.propagate = False

    log_event(logger, "tool.call", args={"query": "SELECT 1"}, name="db_execute",
              module="sql", message="x", levelname="nope")

    out = stream.getvalue()
    assert "tool.call" in out
    assert "SELECT 1" in out


def test_log_event_never_raises_even_on_an_unencodable_value():
    """Observability must not be able to break the request it describes."""
    class Hostile:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    logger = logging.getLogger("test.hostile")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(io.StringIO())
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.propagate = False

    log_event(logger, "tool.result", value=Hostile())  # must not raise


def test_disabled_level_skips_the_work_entirely():
    """The early return is why the reserved-name bug hid for as long as it
    did, so it is worth pinning: a disabled level must not format anything."""
    logger = logging.getLogger("test.disabled")
    logger.setLevel(logging.ERROR)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(TextFormatter())
    logger.handlers = [handler]
    logger.propagate = False

    log_event(logger, "tool.call", level=logging.DEBUG, args={"a": 1})
    assert stream.getvalue() == ""
