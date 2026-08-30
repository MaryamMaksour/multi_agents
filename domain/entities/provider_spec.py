from __future__ import annotations

import re
from dataclasses import dataclass, field
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

# Postgres reserves the pg_ prefix for its own roles, so `CREATE ROLE pg_x`
# is refused and a db_role beginning with it can only name a *predefined*
# role - pg_read_server_files, pg_execute_server_program, pg_write_server_files
# among them. Those are real escalation targets for SET ROLE: they read and
# write files on the database host. Reaching one still requires the
# connecting role to be a member, which is a misconfiguration rather than a
# vulnerability, but a registry file is the wrong place to be one edit away
# from it.
RESERVED_ROLE_PREFIX = "pg_"

# One shared history table, with row-level security keyed on the agent,
# rather than a table per agent. A table per agent means the provisioner runs
# DDL every time somebody registers one, and DDL is the privilege you least
# want reachable from a registration form.
DEFAULT_HISTORY_TABLE = "agent_history"

# The orchestrator reads every routable agent's description on every routing
# decision, so their combined length is a per-turn cost paid by all of them.
# The cap bounds that. It is not a content control - it cannot tell a fair
# description from one that oversells - and the real defence there is that
# descriptions are registry data, written by whoever operates the deployment,
# never by the person asking questions.
MAX_DESCRIPTION_CHARS = 1000


@dataclass
class ProviderSpec:
    name: str

    # Untrusted, and safe to be: the GRANT is the security boundary and no
    # wording can widen what the role may read. This is what makes it
    # reasonable to let a user write their own agent's prompt.
    system_prompt: str

    # What the orchestrator reads when deciding whether to route here. A
    # different kind of text from system_prompt - the prompt cannot widen an
    # agent's reach, but a description that claims to handle everything can
    # draw questions away from the agent that should answer them.
    description: str

    # The Postgres role this agent connects as. The real limit on what it can
    # read, and the source `allowed_tables` is checked against rather than
    # trusted over.
    db_role: str

    # A role, not an identity. The set is closed because these two behave
    # differently in code; the set of *agents* is open by design.
    type: AgentType = AgentType.SUB_AGENT

    # Infrastructure, not registry data. Both default rather than being
    # required, so registering an agent is a name, a prompt, a description
    # and a set of tables - and nobody filling in a form has to know what a
    # history table is.
    history_table: str = DEFAULT_HISTORY_TABLE
    tools: List[str] = field(default_factory=list)

    # What a person sees. Falls back to `name`, which is machine-shaped.
    display_name: str = ""

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
        if self.db_role.startswith(RESERVED_ROLE_PREFIX):
            raise ValueError(
                f"Invalid db_role: {self.db_role!r}. Postgres reserves the "
                f"{RESERVED_ROLE_PREFIX!r} prefix, so this can only name a "
                "predefined role - several of which read or write files on the "
                "database host. An agent's role is one the provisioner created."
            )
        # Interpolated into INSERT INTO for the same reason: a table name is
        # not a parameter.
        if not _NAME_RE.match(self.history_table or ""):
            raise ValueError(
                f"Invalid history_table: {self.history_table!r}. It is used as a "
                "SQL identifier."
            )

        if not (self.description or "").strip():
            raise ValueError(
                f"Agent {self.name!r} has no description. The orchestrator routes "
                "by description alone, so an agent without one can never be chosen."
            )
        if len(self.description) > MAX_DESCRIPTION_CHARS:
            raise ValueError(
                f"Agent {self.name!r} has a description of {len(self.description)} "
                f"characters, over the {MAX_DESCRIPTION_CHARS} limit. Every "
                "routable agent's description is read on every routing decision."
            )

        if not self.display_name:
            self.display_name = self.name

    @property
    def is_routable(self) -> bool:
        """Whether the orchestrator may send work here.

        Read this rather than comparing to ACTIVE at each call site: when a
        fourth status appears, the rule changes in one place.
        """
        return self.status is AgentStatus.ACTIVE
