"""Process-wide logging: one format, one level, and no secret in any record.

Configured once, from the entry point, before anything opens a connection.
Every handler gets a formatter that removes the values of the known secrets
from the message, the arguments, the extra fields and the traceback. The
traceback is the one that matters: nobody writes `logger.info(api_key)`, but
an HTTP client exception carries the request it was making, headers included.
"""

from __future__ import annotations

import json
import logging
import sys
from urllib.parse import urlsplit

from libs.agent_core import config

REDACTED = "***"

# logging.LogRecord's own attributes; anything else on a record arrived
# through `extra=` and is ours to print.
_STANDARD_FIELDS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}


def secret_values() -> tuple[str, ...]:
    """Every configured value that must never appear in a log line.

    Read at call time rather than import time, so a test that sets the
    environment after importing this module still gets its values redacted.
    """
    values = [config.QWEN_API_KEY, config.EMBED_API_KEY, config.PG_PASSWORD]
    redis_password = urlsplit(config.REDIS_URL).password
    if redis_password:
        values.append(redis_password)
    # Longest first, so a key that is a prefix of another is not left half
    # visible by the shorter replacement running first.
    return tuple(sorted({v for v in values if v}, key=len, reverse=True))


def redact(text: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    return text


class SecretRedactingFormatter(logging.Formatter):
    """A formatter that never emits a configured secret.

    Redaction runs over the fully rendered line - message with its arguments
    substituted, the extra fields, and the formatted exception - because that
    is the only string that is guaranteed to be everything that reaches the
    handler. Redacting the message template alone would miss a secret that
    arrived through `%s`.
    """

    def __init__(self, fmt: str | None = None, *, json_lines: bool = False):
        super().__init__(fmt)
        self._json = json_lines

    def format(self, record: logging.LogRecord) -> str:
        secrets = secret_values()
        if self._json:
            rendered = self._format_json(record)
        else:
            rendered = super().format(record)
            extras = self._extra_fields(record)
            if extras:
                rendered += " " + json.dumps(extras, ensure_ascii=False, default=str)
        return redact(rendered, secrets)

    def _format_json(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            **self._extra_fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _extra_fields(record: logging.LogRecord) -> dict:
        return {
            key: value for key, value in vars(record).items()
            if key not in _STANDARD_FIELDS
        }


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
    stream=None,
) -> None:
    """Install the redacting formatter on the root logger.

    Idempotent: replaces the root handlers rather than adding one, so calling
    it twice (the app factory and a test, say) does not print every line
    twice. Uvicorn's own loggers are pointed at the same handler, so its
    access log carries the same guarantee as ours.
    """
    level_name = (config.LOG_LEVEL if level is None else level).upper()
    format_name = (config.LOG_FORMAT if fmt is None else fmt).lower()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        SecretRedactingFormatter(_TEXT_FORMAT, json_lines=format_name == "json")
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level_name)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        for existing in list(uvicorn_logger.handlers):
            uvicorn_logger.removeHandler(existing)
        uvicorn_logger.propagate = True
