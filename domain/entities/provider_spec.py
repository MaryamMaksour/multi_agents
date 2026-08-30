from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class AgentType(Enum):
    SUB_AGENT = "sub_agent"
    ORCHESTRATOR = "orchestrator"


class AgentStatus(Enum):
    """Whether an agent may be routed to yet.

    A closed set, like AgentType - these three states are a property of the
    code, not of a deployment. (The set of agents is the opposite: open by
    design, which is why `name` is a plain string.)

    PENDING is the default because provisioning is asynchronous. An agent is
    registered before its database role exists, and routing to it in that
    window fails at the database. Defaulting to PENDING means a spec that
    forgets to say is treated as not-ready rather than as ready.
    """

    PENDING = "pending"
    ACTIVE = "active"
    DISABLED = "disabled"


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


@dataclass
class ProviderSpec:
    name: str
    type: AgentType
    system_prompt: str
    history_table: str
    tools: List[str]

    # What the orchestrator reads when deciding whether to route here. Note
    # this is a different kind of text from system_prompt: the prompt is
    # untrusted, because the GRANT is the security boundary and no wording
    # can widen it - but the description steers routing, so an agent that
    # claims to handle everything can draw questions away from the agent that
    # should answer them.
    description: str

    # The Postgres role this agent connects as. The real limit on what it can
    # read, and the source `allowed_tables` is checked against rather than
    # trusted over.
    db_role: str

    status: AgentStatus = AgentStatus.PENDING

    # A redundancy check against what introspection reports through db_role,
    # not the source of truth. None means "whatever the role can read".
    tables: Optional[List[str]] = None

    def __post_init__(self) -> None:
        # `name` identifies the agent in routes, history rows and log lines,
        # and `db_role` is interpolated into SET LOCAL ROLE, where it cannot
        # be passed as a parameter. Both are checked here so a bad value is
        # rejected when the spec is built rather than at the point it reaches
        # SQL.
        if not _NAME_RE.match(self.name or ""):
            raise ValueError(
                f"Invalid agent name: {self.name!r}. Expected lowercase letters, "
                "digits and underscores, starting with a letter."
            )
        if not _NAME_RE.match(self.db_role or ""):
            raise ValueError(
                f"Invalid db_role: {self.db_role!r}. It is used as a SQL identifier."
            )

    @property
    def is_routable(self) -> bool:
        """Whether the orchestrator may send work here.

        Read this rather than comparing to ACTIVE at each call site: when a
        fourth status appears, the rule changes in one place.
        """
        return self.status is AgentStatus.ACTIVE
