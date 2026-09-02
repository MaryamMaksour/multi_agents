"""Logging, configured in one place, with secrets that cannot escape.

Before this file the system logged from exactly one module and configured
nothing, which meant the four calls it did make went to the root logger at
WARNING and were never seen. A turn that answered wrongly left no record of
how, and a turn that failed left a traceback with no session, no turn id, and
no way to tell which of five processes produced it.

Three ideas, and the third is the one that matters.

**Context travels with the work, not with the call site.** A log line is only
useful if it says which turn it belongs to, and threading a turn_id through
every function signature down to the SQL adapter would be a change to every
signature. `contextvars` carries it instead: the HTTP edge binds the ids once
per request, every record emitted underneath it - in this task, at any depth -
carries them, and a concurrent request in the same process gets its own copy.

**Formatting is a deployment choice.** LOG_FORMAT=text is for reading during
development; LOG_FORMAT=json is for anything that ships logs somewhere that
parses them. Same records, same fields, one variable.

**A logger must not be able to print a secret.** This system holds an API key
and a database password, and the ways they reach a log line are not ones you
can review for: an httpx exception carries the request headers, a DSN in a
connection error carries the password, and `logger.debug("config: %s", vars)`
is one careless line away at any time. So redaction does not depend on any
call site getting it right. Every record passes through a formatter that knows
the actual secret values and replaces them - in the message, in the arguments,
and in the traceback text, which is where they most often appear. Add a secret
to _register_secret and it is unprintable from that moment, everywhere.

    configure_logging()             once, at process start
    bind(turn_id=..., agent=...)    once per unit of work
    log_event(logger, "turn.start", question_chars=len(q))
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import time
import uuid
from typing import Any, Iterator
from contextlib import contextmanager

# --- request-scoped context ----------------------------------------------
# Set once at the edge, read by every record emitted underneath. Defaults are
# empty strings rather than None so a formatter never has to special-case
# them, and so a line logged outside any request still formats.
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
_turn_id: contextvars.ContextVar[str] = contextvars.ContextVar("turn_id", default="")
_agent: contextvars.ContextVar[str] = contextvars.ContextVar("agent", default="")

_CONTEXT_VARS = {
    "request_id": _request_id,
    "session_id": _session_id,
    "turn_id": _turn_id,
    "agent": _agent,
}


def bind(**values: str) -> dict[str, contextvars.Token]:
    """Attach ids to everything logged from here on in this task.

    Returns the tokens needed to undo it. Prefer `context()`, which does that
    for you; this is for the edge, where the bind outlives the function that
    made it.
    """
    tokens = {}
    for key, value in values.items():
        var = _CONTEXT_VARS.get(key)
        if var is not None and value:
            tokens[key] = var.set(str(value))
    return tokens


def unbind(tokens: dict[str, contextvars.Token]) -> None:
    for key, token in tokens.items():
        _CONTEXT_VARS[key].reset(token)


@contextmanager
def context(**values: str) -> Iterator[None]:
    """Bind ids for the duration of a block, then put back what was there.

    Reset rather than clear, because these nest: a sub-agent turn inside an
    orchestrator turn must not leave the orchestrator's remaining log lines
    labelled with the sub-agent's id.
    """
    tokens = bind(**values)
    try:
        yield
    finally:
        unbind(tokens)


def current_context() -> dict[str, str]:
    return {key: var.get() for key, var in _CONTEXT_VARS.items() if var.get()}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


# --- redaction ------------------------------------------------------------
# Two layers, because they fail differently.
#
# Known values catch what we hold: the API key and the database password are
# read from the environment at startup and registered here, so any line
# containing one has it replaced no matter how it got there - including out of
# a library's exception, which is the case no reviewer catches.
#
# Patterns catch what we do not hold: a key belonging to some other service,
# in a config dump or a provider's error body. Weaker by nature - a pattern
# can only match shapes it was told about - so it is the backstop, not the
# defence.
_REDACTED = "[redacted]"
_known_secrets: set[str] = set()

# Each pattern carries its own replacement rather than the code guessing one
# from the group count. That guess was a real hole: the two three-group
# patterns have opposite shapes - (key)(sep)(SECRET) for key=value but
# (prefix)(SECRET)(@) for a URL - so one rule for both rebuilt the URL with
# the password still in it and the '@' eaten. An explicit template per pattern
# cannot be wrong in that direction.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer tokens, however they are cased.
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{12,}"), r"\1" + _REDACTED),
    # OpenAI-style and Cohere-style keys, as literals anywhere in a line.
    (re.compile(r"\bsk-[A-Za-z0-9._\-]{12,}"), _REDACTED),
    (re.compile(r"\bcohere_[A-Za-z0-9]{12,}"), _REDACTED),
    # key=value and "key": "value" for anything named like a secret.
    (re.compile(r"(?i)\b(api[_-]?key|password|passwd|secret|token|authorization)"
                r"(\"?\s*[:=]\s*\"?)([^\s,;\"'}\)]{4,})"), r"\1\2" + _REDACTED),
    # A password inside a connection URL: scheme://user:secret@host, and the
    # userless scheme://:secret@host that a Redis URL uses.
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^:/\s]*:)([^@/\s]+)(@)"),
     r"\1" + _REDACTED + r"\3"),
]


def register_secret(value: str | None) -> None:
    """Make a literal value unprintable from now on, everywhere.

    Short values are ignored on purpose. Registering a two-character password
    would redact every occurrence of those two characters in every log line
    the process ever writes, which destroys the logs to protect a password
    that should be changed instead.
    """
    if value and len(value) >= 8:
        _known_secrets.add(value)


def redact(text: str) -> str:
    if not text:
        return text
    # Longest first, so a secret that contains another is not left with a
    # readable tail after the shorter one is replaced inside it.
    for secret in sorted(_known_secrets, key=len, reverse=True):
        if secret in text:
            text = text.replace(secret, _REDACTED)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def register_config_secrets() -> None:
    """Register everything this process holds, read straight from the env.

    From os.environ rather than from libs.agent_core.config, so importing this
    module never imports configuration - logging has to be safe to set up
    before anything else exists, including in a script that sets no config at
    all.
    """
    for name in ("QWEN_API_KEY", "EMBED_API_KEY", "PG_PASSWORD",
                 "REDIS_PASSWORD", "OPENAI_API_KEY"):
        register_secret(os.getenv(name))

    # A Redis URL may carry its password inline; register the password itself
    # so it is caught even when the URL is not printed whole.
    redis_url = os.getenv("REDIS_URL", "")
    match = re.match(r"[a-z]+://[^:/\s]*:([^@/\s]+)@", redis_url)
    if match:
        register_secret(match.group(1))


# --- formatters -----------------------------------------------------------
# Both subclass the same redacting base, so neither can be the one that
# forgets. format() is final in spirit: subclasses implement _render.

# The attributes logging puts on every record. Used twice: to pick the caller's
# own fields back out of a record when formatting, and to rename a field that
# would collide before it ever reaches one (see _safe_fields).
_RESERVED_FIELD_NAMES = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Whatever the call site passed as extra=, and nothing else."""
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED_FIELD_NAMES}


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(self._render(record))

    def _render(self, record: logging.LogRecord) -> str:  # pragma: no cover
        raise NotImplementedError

    def _exception_text(self, record: logging.LogRecord) -> str:
        if record.exc_info:
            return self.formatException(record.exc_info)
        return ""


class JsonFormatter(_RedactingFormatter):
    """One JSON object per line, for anything that parses logs."""

    def _render(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(current_context())
        payload.update(_extra_fields(record))

        exception = self._exception_text(record)
        if exception:
            payload["exception"] = exception

        # default=str so a value that is not JSON-native - a UUID, a datetime,
        # a Decimal out of asyncpg - degrades to its string form instead of
        # taking down the log line that was trying to report a problem.
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(_RedactingFormatter):
    """Human-readable, for reading during development."""

    _LEVEL_COLOUR = {
        "DEBUG": "2", "INFO": "36", "WARNING": "33",
        "ERROR": "1;31", "CRITICAL": "1;41",
    }

    def __init__(self, colour: bool = False):
        super().__init__()
        self._colour = colour

    def _paint(self, text: str, level: str) -> str:
        if not self._colour:
            return text
        return f"\033[{self._LEVEL_COLOUR.get(level, '0')}m{text}\033[0m"

    def _render(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        head = self._paint(f"{record.levelname:<8}", record.levelname)

        ctx = current_context()
        # The agent name and the turn are what tell five containers apart in
        # one `docker compose logs`, so they lead. Truncated: a full uuid per
        # line pushes the message off the screen and the first characters
        # already distinguish concurrent turns.
        label = ctx.get("agent", "")
        if ctx.get("turn_id"):
            label = f"{label}/{ctx['turn_id'][:8]}" if label else ctx["turn_id"][:8]

        parts = [f"{stamp} {head}", f"[{label}]" if label else "", record.getMessage()]

        fields = _extra_fields(record)
        if fields:
            parts.append(" ".join(f"{k}={_compact(v)}" for k, v in fields.items()))

        line = " ".join(p for p in parts if p)

        exception = self._exception_text(record)
        if exception:
            line = f"{line}\n{exception}"
        return line


def _compact(value: Any) -> str:
    """A field value on one line, short enough to sit beside the message."""
    if isinstance(value, float):
        return f"{value:.3f}"
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = str(text).replace("\n", "\\n")
    return text if len(text) <= 200 else text[:200] + "..."


# --- setup ----------------------------------------------------------------
# Libraries whose DEBUG output is about their own internals rather than about
# this system. Matched as a prefix, so httpcore2, httpx._client and anything
# else under these roots is covered without naming each one.
_NOISY_ROOTS = ("httpx", "httpcore", "urllib3", "openai", "anyio", "asyncio",
                "aiohttp", "redis", "asyncpg", "langchain", "langgraph",
                "langsmith", "watchfiles", "multipart")


class _ThirdPartyNoiseFilter(logging.Filter):
    """Drop DEBUG and INFO from noisy libraries; keep their warnings.

    Warnings and errors are kept deliberately. "Retrying request" from the
    OpenAI client and "connection pool exhausted" from asyncpg are exactly the
    lines that explain a slow turn, and dropping a whole library is how those
    get lost.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        root = record.name.split(".")[0]
        return not any(root.startswith(noisy) for noisy in _NOISY_ROOTS)


_configured = False


def configure_logging(force: bool = False) -> None:
    """Set up logging for this process. Safe to call more than once.

    Idempotent because it is called from the app's lifespan and from every
    script, and doubling the handlers would double every line.
    """
    global _configured
    if _configured and not force:
        return

    register_config_secrets()

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "text").lower()
    colour = sys.stderr.isatty() and os.getenv("NO_COLOR") is None

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter(colour))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))

    # uvicorn installs its own handlers and marks its loggers non-propagating,
    # so without this the access log bypasses the formatter entirely - and
    # with it, the one place a URL containing a token could print unredacted.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    # Third-party chatter is filtered by prefix rather than by exact logger
    # name, and that is the whole reason it is a filter.
    #
    # setLevel on a list of names only works for the names on the list.
    # LOG_LEVEL=DEBUG on this system produced eight lines per model call from
    # `httpcore2` - connect_tcp.started, send_request_headers.complete,
    # response_closed.complete - because the list said "httpcore" and the
    # package had been renamed. A filter on the handler catches whatever the
    # next version calls itself, and applies to loggers created long after
    # this function has run.
    handler.addFilter(_ThirdPartyNoiseFilter())

    _configured = True


# LogRecord builds its own attributes first and refuses to let `extra`
# overwrite any of them - `logger.info(msg, extra={"args": ...})` raises
# KeyError rather than logging. Several of the reserved names are exactly the
# words a call site wants: a tool call has `args`, a tool has a `name`, a
# message has a `module`.
#
# This shipped and was caught by the test suite, but only just: log_event
# returns early when the level is disabled, so the crash appeared only once
# something else in the run had called configure_logging. Logging that works
# until logging is switched on is the worst possible arrangement, so the fix
# is here rather than at the call sites - a field that collides is renamed,
# and no call site has to know the list.
def _safe_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        (f"{key}_" if key in _RESERVED_FIELD_NAMES else key): value
        for key, value in fields.items()
    }


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO,
              **fields: Any) -> None:
    """Log a named event with structured fields.

    The event name is the message, so text output stays readable and JSON
    output has a stable key to filter on. Fields go through `extra`, which
    both formatters render - so a call site never has to know which format
    the deployment chose.

    Never raises. A log line that takes down the request it was describing is
    a worse outcome than a log line that is missing, and the failure would
    arrive in production - where logging is on - rather than in a test run
    where it is off.
    """
    if not logger.isEnabledFor(level):
        return
    try:
        logger.log(level, event, extra=_safe_fields(fields))
    except Exception:  # noqa: BLE001 - observability must not break the caller
        try:
            logger.log(level, event)
        except Exception:
            pass


class Timer:
    """Elapsed milliseconds for a block, for logging how long something took.

        with Timer() as t:
            ...
        log_event(logger, "llm.call", ms=t.ms)

    Uses a monotonic clock: a wall clock can step backwards and report a
    negative duration, which then reads as a bug in the thing being measured.
    """

    def __init__(self) -> None:
        self.ms = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.ms = (time.monotonic() - self._start) * 1000
