"""Bringing one agent up: check its scope, then read its schema.

This is where the design's central claim gets enforced rather than described.
An agent's scope is its Postgres role's GRANTs. The registry also carries a
table list, and that list is a second opinion - never the source of truth.
When the two disagree, one of them is wrong, and this refuses to start rather
than picking a winner.

Refusing in *both* directions is deliberate, because the two failures are
different and both are bad:

    the role reads more than declared
        Somebody wrote down a narrower agent than the one that exists. The
        registry is the document a person reads to answer "what can this
        agent see", and it is understating the answer.

    the role reads less than declared
        The agent was promised tables it cannot read. It will fail partway
        through somebody's question instead of at startup, and the message
        they get will be about SQL rather than about configuration.

An agent with no declared list is not a failure - it means "whatever the role
can read", which is the design's own answer, stated deliberately.

Nothing here holds elevated privileges. It reads through the agent's own
connection and compares two lists. Creating roles and granting is the
provisioner's job, in a separate component, out of the request path.
"""

from __future__ import annotations

from dataclasses import dataclass

from domain.entities.provider_spec import ProviderSpec
from domain.exceptions import GrantMismatchError
from domain.ports.schema_port import SchemaPort
from libs.agent_core.filter_classifier import DEFAULT_DIST_OP
from libs.agent_core.schema_bootstrap import AgentSchema, load_agent_schema
import logging
from libs.agent_core.logging_setup import Timer, log_event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReadyAgent:
    """One agent, verified and with its schema loaded.

    `schema.tables` is what the agent may actually read - taken from the
    database, not from the registry - so a caller passing this to
    SqlToolAdapter is passing the GRANTs, whatever the file said.
    """

    spec: ProviderSpec
    schema: AgentSchema

    @property
    def allowed_tables(self) -> tuple[str, ...]:
        return self.schema.tables


def verify_grants(spec: ProviderSpec, readable: tuple[str, ...] | set[str]) -> None:
    """Check an agent's declared tables against what its role can read.

    `readable` comes from introspecting through the agent's own role, so it
    is the GRANTs restated. Raises GrantMismatchError naming both directions
    separately - "which tables and which way" is the whole diagnostic value,
    and a message saying only that the sets differ leaves the reader to work
    out what to change.

    Does nothing when the agent declares no tables. That is the design's
    default position, not an oversight to be warned about.
    """
    if spec.tables is None:
        return

    declared = {t.lower() for t in spec.tables}
    granted = {t.lower() for t in readable}

    if declared == granted:
        return

    ungranted = sorted(declared - granted)
    undeclared = sorted(granted - declared)

    problems = []
    if undeclared:
        problems.append(
            "its role can read " + ", ".join(undeclared)
            + " which the registry does not list (the registry understates what "
            "this agent can see)"
        )
    if ungranted:
        problems.append(
            "the registry lists " + ", ".join(ungranted)
            + " which the role cannot read (queries against these would fail at "
            "the database)"
        )

    raise GrantMismatchError(
        f"Agent {spec.name!r} connecting as {spec.db_role!r}: "
        + "; and ".join(problems)
        + ". The GRANTs are the source of truth - fix the registry to match "
        "them, or change the GRANTs deliberately. Starting either way would "
        "leave the two disagreeing."
    )


async def start_agent(
    spec: ProviderSpec,
    schema_port: SchemaPort,
    *,
    dist_op: str = DEFAULT_DIST_OP,
    probe_cardinality: bool = True,
    require_routable: bool = True,
) -> ReadyAgent:
    """Verify an agent's scope and load everything its tools need.

    `schema_port` must already be connected **as this agent's role**. That is
    the one thing this cannot check for itself, and it is what the whole
    verification rests on: introspecting through an administrative connection
    would return every table in the database and the comparison below would
    be meaningless. The composition root owns that connection.

    `require_routable=False` is for a console that wants to inspect an agent
    before provisioning it. It is off by default because in a serving process
    a pending agent's role does not exist yet, and connecting as it will have
    failed long before this.
    """
    if require_routable and not spec.is_routable:
        raise GrantMismatchError(
            f"Agent {spec.name!r} has status {spec.status.value!r} and is not "
            "routable. A pending agent has no role granted yet; a disabled one "
            "was taken out of service deliberately."
        )

    # What the database says this role can read, which is the fact the whole
    # design rests on - the registry's table list is only ever checked against
    # it. Logged because "the agent answers about no tables" and "the GRANT
    # was never run" look identical from a health check.
    readable = tuple(await schema_port.list_tables())
    log_event(logger, "startup.grants", agent=spec.name, db_role=spec.db_role,
              readable=list(readable), declared=list(spec.tables or []))

    verify_grants(spec, readable)

    tables = tuple(spec.tables) if spec.tables is not None else readable
    with Timer() as timer:
        schema = await load_agent_schema(
            schema_port,
            tables=tables,
            dist_op=dist_op,
            probe_cardinality=probe_cardinality,
        )

    log_event(logger, "startup.schema", agent=spec.name, ms=timer.ms,
              tables=list(schema.tables),
              columns=sum(len(f) for f in schema.filters.values()))
    return ReadyAgent(spec=spec, schema=schema)


async def start_all(
    registry,
    schema_port_for,
    *,
    dist_op: str = DEFAULT_DIST_OP,
    probe_cardinality: bool = True,
) -> tuple[ReadyAgent, ...]:
    """Bring up every routable agent, or raise on the first one that fails.

    `schema_port_for` is an async callable taking a ProviderSpec and
    returning a SchemaPort connected as that agent's role. Passed in rather
    than built here so this stays free of asyncpg and testable without a
    database.

    All-or-nothing on purpose. A process that starts with three of four
    agents answers questions as though the fourth does not exist, and the
    person asking has no way to tell the difference between "no data" and
    "that agent failed to start".
    """
    ready = []
    for spec in await registry.list_active():
        port = await schema_port_for(spec)
        ready.append(await start_agent(
            spec, port, dist_op=dist_op, probe_cardinality=probe_cardinality,
        ))
    return tuple(ready)
