"""What an agent is, as a value.

Replaces the previous provider_spec.py. The change that matters is the
removal of AgentType(Enum): an enum is for a closed set of labels, and the
set of agents is open by design - a deployment defines its own agents, and
the core cannot know them. The key becomes a validated string instead.

Nothing here names a table, a domain, or an agent. An AgentSpec is data that
arrives from an AgentRegistryPort; the same code path serves every agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# TODO: agent keys become Postgres role names and history rows, so they need
# the same identifier discipline the SQL tool applies to table names.
# Suggest: ^[a-z][a-z0-9_]{2,30}$  (lowercase, no leading digit, no dashes).
AGENT_KEY_PATTERN = ...


@dataclass(frozen=True)
class AgentSpec:
    """One agent's complete definition.

    Fields to define:

        key            str   - stable identifier, matches AGENT_KEY_PATTERN
        display_name   str   - shown to people, not to the model
        description    str   - what the ORCHESTRATOR sees when deciding
                               whether to route here. This is the routing
                               signal, so it is more sensitive than `prompt`
                               and should not be fully user-controlled.
        prompt         str   - the agent's own system prompt. Untrusted:
                               the GRANT is the boundary, not this text.
        allowed_tables tuple - a redundancy check, NOT the source of truth.
                               The real list comes from the role's GRANTs via
                               introspection. Compare the two and refuse to
                               start on a mismatch rather than trusting this.
        db_role        str   - the Postgres role this agent connects as
        status         str   - pending | active | disabled. The orchestrator
                               must never route to a non-active agent.
    """

    # TODO: implement
    ...


@dataclass(frozen=True)
class AgentDescriptor:
    """The subset the orchestrator needs to route - key, display_name,
    description. Deliberately excludes `prompt` and `db_role` so routing
    logic cannot accidentally depend on them.

    TODO: decide whether this is worth a separate type or whether the
    orchestrator should just read three fields off AgentSpec. Separate type
    wins if the registry ever serves the orchestrator over HTTP.
    """

    # TODO: implement
    ...
