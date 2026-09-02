"""Startup verification - where the design's central claim is enforced.

The claim is that an agent's scope is its role's GRANTs, and that the table
list in the registry is a second opinion rather than the source of truth.
That is easy to state and easy to quietly not do: introspect through an admin
connection instead of the agent's role and every test below still passes
while the property is gone. So the fake here is built around *whose*
connection is being read, and the tests assert on both directions of
disagreement separately.

Both directions refuse, and they refuse differently on purpose. A role that
reads more than declared means the document a person consults to answer "what
can this agent see" is understating it. A role that reads less means an agent
promised tables it cannot read, failing partway through somebody's question
with an error about SQL rather than about configuration.
"""

from __future__ import annotations

import pytest

from domain.entities.provider_spec import AgentStatus, ProviderSpec
from domain.entities.table_schema import ColumnSchema, TableSchema
from domain.exceptions import GrantMismatchError
from libs.agent_core.agent_startup import (
    ReadyAgent,
    start_agent,
    start_all,
    verify_grants,
)

# Shaped so every branch of the classifier is exercised during startup: a
# semantic column with its vector partner, a numeric that needs no probe, and
# a plain text column that does.
TABLES = {
    name: TableSchema(name=name, columns=(
        ColumnSchema("id", "integer", nullable=False),
        ColumnSchema("title", "text"),
        ColumnSchema("embed_title", "vector", is_vector=True),
        ColumnSchema("notes", "text"),
    ))
    for name in ("authors", "publishers", "books", "members", "loans")
}


def spec(**overrides) -> ProviderSpec:
    kwargs = dict(
        name="catalog",
        system_prompt="You answer questions about the catalogue.",
        description="Books, authors and publishers.",
        db_role="app_catalog",
        status=AgentStatus.ACTIVE,
        tables=["authors", "publishers", "books"],
    )
    kwargs.update(overrides)
    return ProviderSpec(**kwargs)


class RoleScopedSchemaPort:
    """A SchemaPort that only reports what one role was granted.

    Modelled on the real thing rather than on the interface: information_schema
    reports only objects the connected role holds a privilege on, so a table
    outside `granted` is absent rather than an error. Tests that ask for one
    are testing that absence.
    """

    def __init__(self, granted=("authors", "publishers", "books"), counts=None):
        self.granted = tuple(granted)
        self._counts = counts or {}
        self.probes: list[tuple[str, str]] = []

    async def list_tables(self):
        return self.granted

    async def describe(self, tables):
        return {t: TABLES[t] for t in tables if t in self.granted and t in TABLES}

    async def distinct_count(self, table, column):
        self.probes.append((table, column))
        return self._counts.get((table, column), 1)

    async def distinct_values(self, table, column, limit):
        self.probes.append((table, column))
        return ("one", "two", "three")[:limit]

    async def has_any_value(self, table, column):
        # A database whose embeddings were filled. The empty case has its own
        # tests in test_schema_bootstrap.py; here it would only obscure what
        # these tests are about, which is grants.
        return True


# --------------------------------------------------------------------------
# verify_grants, on its own
# --------------------------------------------------------------------------


def test_matching_sets_pass_quietly():
    verify_grants(spec(), ("authors", "publishers", "books"))


def test_order_does_not_matter():
    verify_grants(spec(), ("books", "authors", "publishers"))


def test_case_does_not_matter():
    """Unquoted SQL identifiers fold to lowercase; a registry written in
    title case describes the same tables."""
    verify_grants(spec(tables=["Authors", "PUBLISHERS", "books"]),
                  ("authors", "publishers", "books"))


def test_a_role_reading_more_than_declared_is_refused():
    """The dangerous direction. The registry is what a person reads to answer
    "what can this agent see", and here it is understating the answer."""
    with pytest.raises(GrantMismatchError, match="members"):
        verify_grants(spec(), ("authors", "publishers", "books", "members"))


def test_a_role_reading_less_than_declared_is_refused():
    """The agent was promised tables it cannot read. Without this it fails
    partway through a question, with an error about SQL."""
    with pytest.raises(GrantMismatchError, match="publishers"):
        verify_grants(spec(), ("authors", "books"))


def test_the_message_separates_the_two_directions():
    """"The sets differ" leaves the reader to work out what to change."""
    with pytest.raises(GrantMismatchError) as raised:
        verify_grants(spec(tables=["authors", "loans"]), ("authors", "books"))

    message = str(raised.value)
    assert "books" in message and "does not list" in message
    assert "loans" in message and "cannot read" in message


def test_the_message_names_the_agent_and_the_role():
    """Several agents start in one process; "a mismatch" is not actionable
    without knowing which."""
    with pytest.raises(GrantMismatchError) as raised:
        verify_grants(spec(), ("authors",))

    assert "catalog" in str(raised.value)
    assert "app_catalog" in str(raised.value)


def test_the_message_says_which_side_wins():
    """Someone reading this has to change one of the two. Saying the GRANTs
    are the truth makes that a decision rather than a coin toss."""
    with pytest.raises(GrantMismatchError, match="source of truth"):
        verify_grants(spec(), ("authors",))


def test_declaring_no_tables_defers_to_the_grants():
    """Not an oversight to warn about - it is the design's own default."""
    verify_grants(spec(tables=None), ("anything", "at", "all"))


# --------------------------------------------------------------------------
# start_agent
# --------------------------------------------------------------------------


async def test_a_verified_agent_comes_back_ready():
    ready = await start_agent(spec(), RoleScopedSchemaPort())

    assert isinstance(ready, ReadyAgent)
    assert ready.spec.name == "catalog"
    assert set(ready.allowed_tables) == {"authors", "publishers", "books"}


async def test_the_allowed_tables_come_from_the_database_not_the_registry():
    """What reaches SqlToolAdapter is what the role can read. Even with the
    registry agreeing, the value passed on is the introspected one."""
    ready = await start_agent(spec(), RoleScopedSchemaPort())
    assert ready.allowed_tables == ready.schema.tables


async def test_an_agent_with_no_declared_tables_gets_what_its_role_can_read():
    port = RoleScopedSchemaPort(granted=("books", "loans"))
    ready = await start_agent(spec(tables=None), port)

    assert set(ready.allowed_tables) == {"books", "loans"}


async def test_a_mismatch_stops_the_agent_starting():
    port = RoleScopedSchemaPort(granted=("authors", "publishers", "books", "members"))
    with pytest.raises(GrantMismatchError):
        await start_agent(spec(), port)


async def test_the_schema_is_classified_during_startup():
    """The filters the model reads are built here, not on the first
    question - a first question that pays for every distinct count would be
    slow for one user and fast for everyone after them."""
    ready = await start_agent(spec(), RoleScopedSchemaPort())

    assert "books" in ready.schema.filters
    assert isinstance(ready.schema.filters["books"]["title"], str)


async def test_probing_can_be_skipped():
    """The escape hatch for a large database: count(DISTINCT) is the only
    call in startup that scans data rather than the catalogue."""
    port = RoleScopedSchemaPort()
    await start_agent(spec(), port, probe_cardinality=True)
    assert port.probes  # there is something to skip

    port = RoleScopedSchemaPort()
    await start_agent(spec(), port, probe_cardinality=False)
    assert port.probes == []


async def test_the_distance_operator_is_threaded_through():
    """From the composition root, through startup and the classifier, into
    the sentence the model reads. A deployment that sets DIST_OP and has the
    model told a different operator would query an index that cannot serve
    it."""
    ready = await start_agent(spec(), RoleScopedSchemaPort(), dist_op="<->")
    guidance = ready.schema.filters["books"]["title"]

    assert "<->" in guidance and "<=>" not in guidance
    assert "embed_title" in guidance


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


async def test_a_pending_agent_is_refused():
    """Its role does not exist yet. In a serving process the connection
    would already have failed; refusing here is what makes the console's
    "not provisioned" state honest."""
    with pytest.raises(GrantMismatchError, match="pending"):
        await start_agent(spec(status=AgentStatus.PENDING), RoleScopedSchemaPort())


async def test_a_disabled_agent_is_refused():
    with pytest.raises(GrantMismatchError, match="disabled"):
        await start_agent(spec(status=AgentStatus.DISABLED), RoleScopedSchemaPort())


async def test_the_status_check_can_be_waived_for_inspection():
    """A console showing an agent before provisioning it needs this. Off by
    default, because in a serving process it is never right."""
    ready = await start_agent(
        spec(status=AgentStatus.PENDING), RoleScopedSchemaPort(),
        require_routable=False,
    )
    assert ready.spec.status is AgentStatus.PENDING


# --------------------------------------------------------------------------
# start_all
# --------------------------------------------------------------------------


class FakeRegistry:
    def __init__(self, *specs):
        self._specs = specs

    async def list_active(self):
        return self._specs


def ports_for(mapping):
    async def factory(agent_spec):
        return RoleScopedSchemaPort(granted=mapping[agent_spec.name])
    return factory


async def test_every_agent_is_brought_up():
    registry = FakeRegistry(
        spec(),
        spec(name="circulation", db_role="app_circulation",
             description="Loans and members.", tables=["members", "loans"]),
    )
    ready = await start_all(registry, ports_for({
        "catalog": ("authors", "publishers", "books"),
        "circulation": ("members", "loans"),
    }))

    assert [r.spec.name for r in ready] == ["catalog", "circulation"]


async def test_each_agent_reads_through_its_own_connection():
    """The property the whole verification rests on. One shared admin
    connection would report every table to every agent, and every comparison
    above would pass while meaning nothing."""
    registry = FakeRegistry(
        spec(),
        spec(name="circulation", db_role="app_circulation",
             description="Loans and members.", tables=["members", "loans"]),
    )
    ready = await start_all(registry, ports_for({
        "catalog": ("authors", "publishers", "books"),
        "circulation": ("members", "loans"),
    }))

    scopes = {r.spec.name: set(r.allowed_tables) for r in ready}
    assert scopes["catalog"] == {"authors", "publishers", "books"}
    assert scopes["circulation"] == {"members", "loans"}
    assert scopes["catalog"] & scopes["circulation"] == set()


async def test_one_bad_agent_stops_the_whole_startup():
    """All-or-nothing. A process that starts with three of four answers as
    though the fourth does not exist, and the person asking cannot tell that
    apart from "no data"."""
    registry = FakeRegistry(
        spec(),
        spec(name="circulation", db_role="app_circulation",
             description="Loans and members.", tables=["members", "loans"]),
    )
    with pytest.raises(GrantMismatchError, match="circulation"):
        await start_all(registry, ports_for({
            "catalog": ("authors", "publishers", "books"),
            "circulation": ("members",),
        }))
