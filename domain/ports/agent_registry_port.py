"""Where agent definitions come from.

This port is what makes the core deployment-agnostic. Behind it, phase 1 puts
a JSON file and phase 3 puts a Postgres table - the core never learns which,
and moving between them is a new adapter rather than a change here.

Writing this port before it was strictly needed is what kept phase 3 from
being a rewrite. It cost one file.
"""

from __future__ import annotations

from typing import Protocol

from domain.entities.provider_spec import ProviderSpec


class AgentRegistryPort(Protocol):
    """Reads agent definitions.

    Read-only, and deliberately so. Registration writes go through the
    provisioner, which holds credentials that can CREATE ROLE and GRANT -
    privileges nothing serving requests should hold. Adding create/update
    here later would move that boundary, and moving it should be a decision
    rather than a convenience.
    """

    async def get(self, key: str) -> ProviderSpec:
        """The agent registered under `key`.

        Raises rather than returning None for an unknown key: every caller
        needs an agent to continue, so None would only be checked and
        re-raised at each call site.

        Raises:
            UnknownAgentError: if no agent is registered under that key.
        """
        ...

    async def list_active(self) -> tuple[ProviderSpec, ...]:
        """Every agent the orchestrator may route to.

        Only routable agents. A pending one has no database role granted yet,
        so offering it as a tool produces a database error in the middle of
        answering somebody's question; a disabled one was taken out of
        service deliberately.

        Ordering is stable, because the orchestrator's tool list is built
        from this and a list that reshuffles between restarts is a prompt
        that changes for no reason.
        """
        ...
