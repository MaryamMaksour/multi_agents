"""Configuration reading and its startup check.

Two properties matter here. Importing must not raise, because a module that
fails on import cannot be imported by a test and produces an error before
there is any logging to report it through. And validate() must refuse to let
a service start half-configured, since the failure mode it prevents - an
unset host silently resolving somewhere unintended - is the kind that is
noticed late and from the wrong direction.
"""

from __future__ import annotations

import importlib

import pytest

from libs.agent_core import config as config_module

FULL_ENV = {
    "PG_DBNAME": "library_dev",
    "PG_USER": "dev",
    "PG_PASSWORD": "dev",
    "PG_HOST": "localhost",
    "QWEN_API_KEY": "test-key",
}


@pytest.fixture
def config(monkeypatch):
    """A freshly imported config module with a complete environment."""
    for key, value in FULL_ENV.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config_module)


@pytest.fixture(autouse=True)
def _restore_module():
    """Reload once more afterwards so a mutated module cannot leak into the
    next test - these tests deliberately overwrite module-level values."""
    yield
    importlib.reload(config_module)


# --------------------------------------------------------------------------
# import safety
# --------------------------------------------------------------------------


def test_importing_with_an_empty_environment_does_not_raise(monkeypatch):
    for key in FULL_ENV:
        monkeypatch.delenv(key, raising=False)
    importlib.reload(config_module)  # must not raise


def test_importing_opens_no_connections(config):
    """Nothing here should hold a pool, a client or a socket - building those
    belongs in the composition root's lifespan.

    Imported modules and __future__ flags are expected; anything else that is
    neither a plain value nor a function is the kind of live object this file
    must not create at import time.
    """
    import __future__
    import inspect

    suspicious = [
        name for name, value in vars(config).items()
        if not name.startswith("_")
        and not isinstance(value, (str, int, float, bool, tuple, frozenset, type(None)))
        and not callable(value)
        and not inspect.ismodule(value)
        and not isinstance(value, __future__._Feature)
    ]
    assert suspicious == [], f"config holds non-configuration objects: {suspicious}"


# --------------------------------------------------------------------------
# validate()
# --------------------------------------------------------------------------


def test_passes_with_a_complete_environment(config):
    config.validate()


def test_reports_every_missing_variable_at_once(config):
    """Finding them one restart at a time is the small friction that makes a
    first deployment take an hour."""
    config.PG_HOST = ""
    config.QWEN_API_KEY = ""
    config.PG_PASSWORD = ""

    with pytest.raises(RuntimeError) as excinfo:
        config.validate()

    message = str(excinfo.value)
    for name in ("PG_HOST", "QWEN_API_KEY", "PG_PASSWORD"):
        assert name in message


@pytest.mark.parametrize("name", sorted(FULL_ENV))
def test_each_required_variable_is_checked(config, name):
    setattr(config, name, "")
    with pytest.raises(RuntimeError, match=name):
        config.validate()


def test_secrets_have_no_working_default(monkeypatch):
    """An unset password that falls back to a real one fails in the worst
    possible way: silently, against the wrong system."""
    for key in FULL_ENV:
        monkeypatch.delenv(key, raising=False)
    fresh = importlib.reload(config_module)

    for name in ("PG_PASSWORD", "PG_HOST", "PG_DBNAME", "PG_USER", "QWEN_API_KEY"):
        assert getattr(fresh, name) == "", f"{name} must not default to anything usable"


# --------------------------------------------------------------------------
# values that reach SQL
# --------------------------------------------------------------------------


@pytest.mark.parametrize("operator", ["<=>", "<->", "<#>"])
def test_accepts_the_three_pgvector_operators(config, operator):
    config.DIST_OP = operator
    config.validate()


@pytest.mark.parametrize("operator", ["; DROP TABLE books", "=", "", "<=> --", "OR 1=1"])
def test_rejects_any_other_distance_operator(config, operator):
    """DIST_OP is interpolated into SQL and cannot be parameterised, so the
    allowlist is the only thing guarding it."""
    config.DIST_OP = operator
    with pytest.raises(RuntimeError, match="DIST_OP"):
        config.validate()


# --------------------------------------------------------------------------
# defaults that carry meaning
# --------------------------------------------------------------------------


def test_context_sent_is_smaller_than_what_is_retained(config):
    """They are separate on purpose: one bounds what is kept, the other what
    is sent. Sending more than is kept would be incoherent."""
    assert config.CONTEXT_MESSAGES_SENT <= config.MAX_SESSION_MESSAGES


def test_context_window_can_hold_several_whole_turns(config):
    """A turn spans user, tool call, tool result and final answer, so a small
    slice would deliver turns cut in half."""
    assert config.CONTEXT_MESSAGES_SENT >= 8


def test_pagination_limit_is_set_and_bounded(config):
    assert 1 <= config.MAX_PAGES_PER_TOOL <= 20


def test_environment_overrides_a_default(monkeypatch):
    monkeypatch.setenv("MAX_PAGES_PER_TOOL", "3")
    assert importlib.reload(config_module).MAX_PAGES_PER_TOOL == 3


def test_no_per_agent_urls_are_configured(config):
    """Which agents exist is registry data. A *_AGENT_URL here would mean the
    core knows its deployment's agents, which is what the registry exists to
    prevent."""
    leaked = [n for n in vars(config) if n.endswith("_AGENT_URL")]
    assert leaked == []


# --------------------------------------------------------------------------
# the embedding endpoint, split from the chat one
# --------------------------------------------------------------------------


def test_the_embedding_endpoint_defaults_to_the_chat_one(monkeypatch):
    """A deployment using one endpoint for both sets nothing and nothing
    changes. The split has to be free for the people who do not need it."""
    monkeypatch.delenv("EMBED_API_URL", raising=False)
    monkeypatch.delenv("EMBED_API_KEY", raising=False)
    monkeypatch.setenv("QWEN_API_URL", "https://example.test/v1")
    monkeypatch.setenv("QWEN_API_KEY", "chat-key")

    config = importlib.reload(config_module)
    assert config.EMBED_API_URL == "https://example.test/v1"
    assert config.EMBED_API_KEY == "chat-key"


def test_the_embedding_endpoint_can_point_somewhere_else(monkeypatch):
    """The hybrid this exists for: embeddings local and free, where the call
    count is high, and a hosted model for generation."""
    monkeypatch.setenv("QWEN_API_URL", "https://hosted.test/v1")
    monkeypatch.setenv("QWEN_API_KEY", "chat-key")
    monkeypatch.setenv("EMBED_API_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("EMBED_API_KEY", "local")

    config = importlib.reload(config_module)
    assert config.EMBED_API_URL == "http://localhost:8001/v1"
    assert config.QWEN_API_URL == "https://hosted.test/v1"


def test_a_separate_endpoint_without_a_key_is_refused_at_startup(monkeypatch):
    """A local server usually ignores the key, but the client still requires
    one. Caught here rather than on the first question."""
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EMBED_API_URL", "http://localhost:8001/v1")

    config = importlib.reload(config_module)
    config.EMBED_API_KEY = ""
    with pytest.raises(RuntimeError, match="EMBED_API_KEY"):
        config.validate()


def test_a_model_name_in_a_url_setting_is_refused(monkeypatch):
    """Four related settings sit together and two of them take URLs. Putting
    a model name in one otherwise fails as an HTTP error against a hostname
    that is a model name, which names neither the setting nor the mistake."""
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("EMBED_API_URL", "embed-multilingual-v3.0")

    config = importlib.reload(config_module)
    with pytest.raises(RuntimeError, match="is not a URL"):
        config.validate()


def test_a_chat_url_that_is_not_a_url_is_refused(monkeypatch):
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QWEN_API_URL", "qwen-plus")

    config = importlib.reload(config_module)
    with pytest.raises(RuntimeError, match="QWEN_API_URL"):
        config.validate()


def test_real_urls_pass(monkeypatch):
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QWEN_API_URL", "https://example.test/v1")
    monkeypatch.setenv("EMBED_API_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBED_API_KEY", "local")

    importlib.reload(config_module).validate()


# --------------------------------------------------------------------------
# enable_thinking policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model, setting, expected", [
    ("qwen3-14b", "", {"enable_thinking": False}),     # hybrid model: must send false
    ("Qwen3-Coder-480B", "", {"enable_thinking": False}),
    ("qwen-plus", "", None),                            # not hybrid: nothing sent
    ("gpt-4.1", "", None),                              # strict endpoint: nothing sent
    ("gpt-4.1", "false", {"enable_thinking": False}),   # explicit override wins
    ("qwen3-14b", "true", {"enable_thinking": True}),
    ("qwen3-14b", "none", None),                        # explicit opt-out
])
def test_enable_thinking_is_inferred_from_the_model_unless_forced(model, setting, expected):
    assert config_module.llm_extra_body(model, setting) == expected


def test_enable_thinking_defaults_come_from_the_environment(monkeypatch):
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QWEN_MODEL", "qwen3-14b")
    monkeypatch.delenv("QWEN_ENABLE_THINKING", raising=False)

    config = importlib.reload(config_module)
    assert config.llm_extra_body() == {"enable_thinking": False}

    monkeypatch.setenv("QWEN_MODEL", "gpt-4.1")
    config = importlib.reload(config_module)
    assert config.llm_extra_body() is None


def test_an_unknown_enable_thinking_value_is_refused_at_startup(monkeypatch):
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QWEN_ENABLE_THINKING", "flase")

    config = importlib.reload(config_module)
    with pytest.raises(RuntimeError, match="QWEN_ENABLE_THINKING"):
        config.validate()


@pytest.mark.parametrize("value", ["", "true", "False", "none"])
def test_documented_enable_thinking_values_pass(monkeypatch, value):
    for name, value_ in FULL_ENV.items():
        monkeypatch.setenv(name, value_)
    monkeypatch.setenv("QWEN_ENABLE_THINKING", value)

    importlib.reload(config_module).validate()


# --------------------------------------------------------------------------
# QWEN_EXTRA_BODY, the provider escape hatch
# --------------------------------------------------------------------------


def test_extra_body_is_merged_into_the_request(config):
    """A model that wants one extra parameter should not mean editing an
    adapter and rebuilding an image."""
    assert config.llm_extra_body(
        "gpt-4.1", "none", '{"top_k": 20}'
    ) == {"top_k": 20}


def test_extra_body_is_merged_alongside_enable_thinking(config):
    body = config.llm_extra_body("qwen3-14b", "", '{"top_k": 20}')
    assert body == {"enable_thinking": False, "top_k": 20}


def test_extra_body_wins_over_the_inferred_value(config):
    """Explicit beats inferred: the escape hatch is what you reach for when
    the inference is wrong for your provider."""
    body = config.llm_extra_body("qwen3-14b", "", '{"enable_thinking": true}')
    assert body == {"enable_thinking": True}


def test_an_empty_extra_body_changes_nothing(config):
    assert config.llm_extra_body("gpt-4.1", "none", "") is None
    assert config.llm_extra_body("gpt-4.1", "none", "   ") is None


@pytest.mark.parametrize("value", ['{"top_k": 20', "[1, 2]", '"a string"', "7"])
def test_a_malformed_extra_body_is_refused_at_startup(monkeypatch, value):
    """Next to the variable that caused it, rather than inside the first
    question."""
    for name, value_ in FULL_ENV.items():
        monkeypatch.setenv(name, value_)
    monkeypatch.setenv("QWEN_EXTRA_BODY", value)

    config = importlib.reload(config_module)
    with pytest.raises(RuntimeError, match="QWEN_EXTRA_BODY"):
        config.validate()


def test_extra_body_defaults_come_from_the_environment(monkeypatch):
    for name, value in FULL_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("QWEN_MODEL", "gpt-4.1")
    monkeypatch.setenv("QWEN_ENABLE_THINKING", "none")
    monkeypatch.setenv("QWEN_EXTRA_BODY", '{"top_k": 20}')

    config = importlib.reload(config_module)
    config.validate()
    assert config.llm_extra_body() == {"top_k": 20}


# --------------------------------------------------------------------------
# logging and limits
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("LOG_FORMAT", "yaml", "LOG_FORMAT"),
        ("LOG_LEVEL", "VERBOSE", "LOG_LEVEL"),
        ("MAX_QUESTION_CHARS", "0", "MAX_QUESTION_CHARS"),
        ("AGENT_MAX_STEPS", "0", "AGENT_MAX_STEPS"),
    ],
)
def test_a_bad_operational_value_is_refused_at_startup(monkeypatch, name, value, message):
    for key, value_ in FULL_ENV.items():
        monkeypatch.setenv(key, value_)
    monkeypatch.setenv(name, value)

    config = importlib.reload(config_module)
    with pytest.raises(RuntimeError, match=message):
        config.validate()


@pytest.mark.parametrize("fmt", ["text", "json", "JSON"])
@pytest.mark.parametrize("level", ["debug", "INFO", "Warning"])
def test_documented_logging_values_pass(monkeypatch, fmt, level):
    for key, value in FULL_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("LOG_FORMAT", fmt)
    monkeypatch.setenv("LOG_LEVEL", level)

    importlib.reload(config_module).validate()


def test_a_500_does_not_name_the_exception_unless_asked(monkeypatch):
    """The unconfigured deployment is the one most likely to be facing someone
    who is not operating it."""
    for key, value in FULL_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("EXPOSE_ERRORS", raising=False)

    assert importlib.reload(config_module).EXPOSE_ERRORS is False


@pytest.mark.parametrize("value,expected", [("true", True), ("false", False), ("TRUE", True)])
def test_naming_the_exception_is_opt_in_from_the_environment(monkeypatch, value, expected):
    for key, env_value in FULL_ENV.items():
        monkeypatch.setenv(key, env_value)
    monkeypatch.setenv("EXPOSE_ERRORS", value)

    assert importlib.reload(config_module).EXPOSE_ERRORS is expected
