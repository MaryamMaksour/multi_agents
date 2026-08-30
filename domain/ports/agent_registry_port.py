"""Where agent definitions come from.

This port is what makes the core deployment-agnostic. Behind it, phase 1 puts
a JSON file and phase 3 puts a Postgres table - the core never learns which,
and moving between them is a new adapter rather than a change here.

Writing this port now, before it is strictly needed, is what keeps phase 3
from being a rewrite. It costs one file today.
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.provider_spec import ProviderSpec


class AgentRegistryPort(Protocol):
    """Reads agent definitions.

    Methods to define:

        async def get(key: str) -> ProviderSpec
            Raises rather than returning None for an unknown key - a missing
            agent is a real error, and the caller cannot proceed without one.

        async def list_active() -> tuple[ProviderSpec, ...]
            The orchestrator builds its tool list from this. Only agents with
            status 'active' - a pending agent has no role granted yet, so
            routing to it would fail at the database.

    TODO: read-only on purpose. Registration writes go through the
    provisioner, which holds elevated credentials and is not reachable from
    the request path. Do not add create/update here later without moving that
    boundary deliberately.

    TODO: both methods are called per turn once composition moves to
    request time, so the adapter needs caching with explicit invalidation -
    not a TTL alone, or a newly registered agent stays invisible for the
    length of the TTL.
    """

    ...
