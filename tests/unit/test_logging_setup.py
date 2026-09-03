"""The redacting formatter, which is the whole of the "secrets can't be
logged" claim.

Every case here is a place a secret reaches a log line without anyone
writing it there: substituted into a message through %s, attached as an
extra field, or carried inside an exception that was raised while making a
request. The last is the one that matters - nobody writes logger.info(key),
but an HTTP client's exception carries the request it was making.
"""

from __future__ import annotations

import importlib
import io
import json
import logging

import pytest

from libs.agent_core import config as config_module
from libs.agent_core import logging_setup


@pytest.fixture
def secrets(monkeypatch):
    """A config whose secret values are recognisable in a log line."""
    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen-secret")
    monkeypatch.setenv("EMBED_API_KEY", "sk-embed-secret")
    monkeypatch.setenv("PG_PASSWORD", "pg-secret")
    monkeypatch.setenv("REDIS_URL", "redis://:redis-secret@redis:6379/1")
    reloaded = importlib.reload(config_module)
    monkeypatch.setattr(logging_setup, "config", reloaded)
    return reloaded


@pytest.fixture
def log(secrets):
    """Configure logging onto a buffer and hand back a reader for it."""
    stream = io.StringIO()

    def configure(**kwargs):
        logging_setup.configure_logging(stream=stream, **kwargs)
        return logging.getLogger("test.logger")

    yield configure, stream

    logging.getLogger().handlers.clear()
    importlib.reload(config_module)


def test_a_secret_substituted_into_the_message_is_removed(log):
    configure, stream = log
    logger = configure()

    logger.info("calling the model with %s", "sk-qwen-secret")

    assert "sk-qwen-secret" not in stream.getvalue()
    assert logging_setup.REDACTED in stream.getvalue()


def test_a_secret_in_an_extra_field_is_removed(log):
    configure, stream = log
    logger = configure()

    logger.info("tool called", extra={"tool_args": {"password": "pg-secret"}})

    written = stream.getvalue()
    assert "pg-secret" not in written
    assert "tool_args" in written, "extra fields are printed, not silently dropped"


def test_a_secret_inside_a_traceback_is_removed(log):
    """The case the posture section leads with: nobody logs the key, but an
    exception raised while using it carries it."""
    configure, stream = log
    logger = configure()

    try:
        raise RuntimeError("401 from https://api/v1 with key sk-embed-secret")
    except RuntimeError:
        logger.exception("model call failed")

    written = stream.getvalue()
    assert "sk-embed-secret" not in written
    assert "RuntimeError" in written, "the traceback itself still reaches the log"


def test_a_password_inside_the_redis_url_is_removed(log):
    """It is not a variable of its own - it has to be parsed out of the URL,
    which is exactly the sort of secret a redactor forgets."""
    configure, stream = log
    logger = configure()

    logger.error("cannot reach redis://:redis-secret@redis:6379/1")

    assert "redis-secret" not in stream.getvalue()


def test_json_format_emits_one_object_per_record(log):
    configure, stream = log
    logger = configure(fmt="json")

    logger.info("tool called", extra={"tool": "db_execute", "turn_id": "t-1"})

    payload = json.loads(stream.getvalue().strip())
    assert payload["message"] == "tool called"
    assert payload["level"] == "INFO"
    assert payload["tool"] == "db_execute"
    assert payload["turn_id"] == "t-1"


def test_json_format_redacts_too(log):
    configure, stream = log
    logger = configure(fmt="json")

    logger.info("using %s", "sk-qwen-secret", extra={"key": "sk-embed-secret"})

    written = stream.getvalue()
    assert "sk-qwen-secret" not in written and "sk-embed-secret" not in written
    json.loads(written.strip())  # still valid JSON after redaction


def test_the_level_is_honoured(log):
    configure, stream = log
    logger = configure(level="WARNING")

    logger.info("not this one")
    logger.warning("this one")

    assert "not this one" not in stream.getvalue()
    assert "this one" in stream.getvalue()


def test_configuring_twice_does_not_double_every_line(log):
    """The app factory configures logging, and so does a test. Handlers are
    replaced rather than added, or every line appears twice."""
    configure, stream = log
    configure()
    logger = configure()

    logger.info("once")

    assert stream.getvalue().count("once") == 1


def test_an_empty_secret_is_not_treated_as_one(monkeypatch):
    """An unset key is the empty string, and replacing it would redact the
    space between every character."""
    monkeypatch.setenv("QWEN_API_KEY", "")
    monkeypatch.setenv("EMBED_API_KEY", "")
    monkeypatch.setenv("PG_PASSWORD", "")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/1")
    reloaded = importlib.reload(config_module)
    monkeypatch.setattr(logging_setup, "config", reloaded)

    assert logging_setup.secret_values() == ()
    assert logging_setup.redact("nothing to hide", ()) == "nothing to hide"

    importlib.reload(config_module)


def test_longer_secrets_are_replaced_first(monkeypatch):
    """A key that contains another key must not be left half visible by the
    shorter replacement running first."""
    monkeypatch.setenv("QWEN_API_KEY", "sk-abc")
    monkeypatch.setenv("EMBED_API_KEY", "sk-abcdef")
    monkeypatch.setenv("PG_PASSWORD", "")
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/1")
    reloaded = importlib.reload(config_module)
    monkeypatch.setattr(logging_setup, "config", reloaded)

    redacted = logging_setup.redact("sk-abcdef", logging_setup.secret_values())

    assert "abc" not in redacted

    importlib.reload(config_module)
