"""Environment configuration, read in one place.

Two rules shape this file.

Nothing here connects to anything. Reading an environment variable at import
time is safe; opening a pool is not, and that is what the composition root's
lifespan is for. Keeping the split means importing a module never reaches the
network, so tests import freely.

And no secret or host gets a working-looking default. An unset PG_PASSWORD
that quietly falls back to a real password, or an unset PG_HOST that resolves
to some other network's database, fails in the worst possible way: silently
and against the wrong system. Those default to empty and are caught by
validate() instead - loudly, at startup, naming what is missing.

What is deliberately absent: per-agent URLs. Which agents exist and where
they answer is deployment data that arrives through AgentRegistryPort, not
configuration compiled into every service.
"""

from __future__ import annotations

import os

# --- Qwen / any OpenAI-compatible endpoint -------------------------------
# QWEN_API_URL is the seam between the hosted and self-hosted editions.
# Pointing it at a local vLLM or Ollama server is a configuration change,
# not a code change - the adapter is the same either way.
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_API_URL = os.getenv("QWEN_API_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-14b")
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.1"))
QWEN_MAX_TOKENS = int(os.getenv("QWEN_MAX_TOKENS", "32000"))
# Qwen3 hybrid-thinking models (qwen3-*) reject non-streaming calls unless
# enable_thinking is explicitly false, while strict OpenAI-compatible
# endpoints reject the parameter altogether. Unset means: infer from the
# model name - qwen3-* gets false, anything else gets nothing. "true" /
# "false" force it; "none" forces it off for a qwen3-* model.
QWEN_ENABLE_THINKING = os.getenv("QWEN_ENABLE_THINKING", "")


def llm_extra_body(model: str | None = None, enable_thinking: str | None = None) -> dict | None:
    """Provider-specific request fields for the chat call, or None."""
    model = QWEN_MODEL if model is None else model
    setting = (QWEN_ENABLE_THINKING if enable_thinking is None else enable_thinking).strip().lower()
    if setting in ("true", "false"):
        return {"enable_thinking": setting == "true"}
    if setting == "" and model.lower().startswith("qwen3"):
        return {"enable_thinking": False}
    return None


QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3")

# Embeddings can come from somewhere else entirely, and often have to.
#
# A vLLM process serves one model, so self-hosting means two servers - one
# generating, one embedding - on different ports at least. And the two have
# opposite shapes: chat is few calls of many tokens, embedding is many calls
# of few, so they suit different hardware and different bills. Splitting them
# also makes the sensible hybrid possible: embeddings local and free, where
# the call count is high, and a hosted model for generation, where quality
# matters most.
#
# Defaults to the chat endpoint, so a deployment using one endpoint for both
# sets nothing and nothing changes.
EMBED_API_URL = os.getenv("EMBED_API_URL", "") or QWEN_API_URL
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "") or QWEN_API_KEY

# Fixed in the column type as vector(N), so changing it is a schema migration
# and a re-embedding of every row, not a restart. Kept here so the value the
# code sends and the value the columns accept come from one place.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# --- PostgreSQL / pgvector -----------------------------------------------
PG_DBNAME = os.getenv("PG_DBNAME", "")
PG_USER = os.getenv("PG_USER", "")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")
PG_HOST = os.getenv("PG_HOST", "")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_SSL = os.getenv("PG_SSL", "false").lower() == "true"

DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "1"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
DB_COMMAND_TIMEOUT = float(os.getenv("DB_COMMAND_TIMEOUT", "60"))

# pgvector distance operator: '<->' L2, '<#>' inner product, '<=>' cosine.
DIST_OP = os.getenv("DIST_OP", "<=>")

# --- Redis ----------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/1")

# --- Agent registry -------------------------------------------------------
# Phase 2 reads agents from a file; phase 4 reads them from a table. Both sit
# behind AgentRegistryPort, so this selects an adapter and nothing more.
AGENTS_REGISTRY_PATH = os.getenv("AGENTS_REGISTRY_PATH", "seeds/agents.example.json")

# Which agent an agent_runtime process serves. One image, many containers,
# differing only in this value. Empty in the orchestrator, which serves none.
AGENT_KEY = os.getenv("AGENT_KEY", "")

# Where a sub-agent answers, when its registry entry does not say. `{key}` is
# the agent's own name, which is also its service name in a compose file or a
# Kubernetes Service - so the default is right for the common deployment and
# an entry's own `endpoint` overrides it for anything else.
#
# Still not per-agent configuration: this is one template, and which agents
# exist comes from the registry.
AGENT_URL_TEMPLATE = os.getenv("AGENT_URL_TEMPLATE", "http://agent-{key}:8000/run")

# What this process listens on.
HTTP_HOST = os.getenv("HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))

# --- Limits ---------------------------------------------------------------
# How many pages one tool may be walked through in a single turn. Without it
# a model that keeps seeing has_more=true will page until the context fills.
MAX_PAGES_PER_TOOL = int(os.getenv("MAX_PAGES_PER_TOOL", "5"))

# How much history is kept per session.
MAX_SESSION_MESSAGES = int(os.getenv("MAX_SESSION_MESSAGES", "40"))

# How much of that history is actually sent to the model each turn. Separate
# from MAX_SESSION_MESSAGES on purpose: one turn already spans several
# messages (user, tool call, tool result, ..., final answer), so this has to
# be a multiple of that rather than a small slice, or a turn arrives cut in
# half.
CONTEXT_MESSAGES_SENT = int(os.getenv("CONTEXT_MESSAGES_SENT", "20"))

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 24 * 3)))

# A vector token only has to outlive the turn that made it.
VECTOR_TTL_SECONDS = int(os.getenv("VECTOR_TTL_SECONDS", "900"))

TOOLS_HTTP_TIMEOUT_SECS = int(os.getenv("TOOLS_HTTP_TIMEOUT_SECS", "60"))


_REQUIRED = ("PG_DBNAME", "PG_USER", "PG_PASSWORD", "PG_HOST", "QWEN_API_KEY")


def validate() -> None:
    """Fail at startup if anything required is unset.

    Called from the composition root, not at import: a module that raises on
    import cannot be imported by a test, and the error would arrive before
    there is any logging to report it through.

    Reports every missing name at once. Finding them one restart at a time is
    the kind of small friction that makes a first deployment take an hour.
    """
    missing = [name for name in _REQUIRED if not globals().get(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". These have no safe default - an unset database host or password "
            "would otherwise fail by connecting somewhere unintended."
        )

    # Not in _REQUIRED: it defaults to QWEN_API_KEY, which is required, so a
    # deployment sharing one endpoint can never reach here with it empty. It
    # is only empty when someone set EMBED_API_URL and forgot the key - which
    # would otherwise fail on the first question rather than at startup.
    if not EMBED_API_KEY:
        raise RuntimeError(
            "EMBED_API_URL is set but EMBED_API_KEY is empty. A local server "
            "usually ignores the key, but it still has to be present - pass any "
            "non-empty value."
        )

    # A URL where a model name was meant, or the reverse. Easy to do when
    # four related settings sit together, and the failure otherwise arrives
    # as an HTTP error against a hostname that is a model name - which names
    # neither the setting nor the mistake.
    for name, value in (("QWEN_API_URL", QWEN_API_URL), ("EMBED_API_URL", EMBED_API_URL)):
        if value and not value.startswith(("http://", "https://")):
            raise RuntimeError(
                f"{name}={value!r} is not a URL. It is a base URL like "
                "'https://host/v1'; the model name goes in QWEN_MODEL or "
                "QWEN_EMBED_MODEL."
            )

    if DIST_OP not in ("<=>", "<->", "<#>"):
        raise RuntimeError(
            f"DIST_OP must be one of '<=>', '<->', '<#>' - got {DIST_OP!r}. "
            "It is interpolated into SQL and cannot be parameterised."
        )
