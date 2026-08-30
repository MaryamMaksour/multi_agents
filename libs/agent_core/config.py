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

QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v3")

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

    if DIST_OP not in ("<=>", "<->", "<#>"):
        raise RuntimeError(
            f"DIST_OP must be one of '<=>', '<->', '<#>' - got {DIST_OP!r}. "
            "It is interpolated into SQL and cannot be parameterised."
        )
