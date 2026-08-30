"""The registry adapter - agents as data, and the validation that guards it.

Almost every test here is about a malformed file, which is the point. The
happy path is a JSON load; the value is in refusing to start on input that
would otherwise produce a service that looks fine and answers wrongly.

Two of those refusals are security-shaped rather than tidiness-shaped, and
they are worth naming: an unrecognised field is rejected because a misspelt
`allowed_tables` would read as "no restriction", and an empty
`allowed_tables` is rejected because it cannot be told apart from "I meant to
fill this in" while silently meaning "this agent may read nothing".

The rest is written against a real file on disk rather than a mocked open(),
because the thing being tested is the parsing of a file somebody edits by
hand.
"""

from __future__ import annotations

import json

import pytest

from adapters.outbound.registry.file_agent_registry_adapter import (
    FileAgentRegistryAdapter,
)
from domain.entities.provider_spec import AgentStatus, AgentType
from domain.exceptions import RegistryError, UnknownAgentError

AGENT = {
    "key": "catalog",
    "display_name": "Catalogue",
    "description": "Books, their authors and publishers.",
    "db_role": "app_catalog",
    "allowed_tables": ["authors", "publishers", "books"],
    "status": "active",
    "prompt": "You answer questions about a library's catalogue.",
}


def write(tmp_path, *agents, extra=None):
    payload = {"agents": list(agents)}
    if extra:
        payload.update(extra)
    path = tmp_path / "agents.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def registry(tmp_path, *agents, **kw):
    return FileAgentRegistryAdapter(write(tmp_path, *(agents or (AGENT,)), **kw))


def agent(**overrides):
    return {**AGENT, **overrides}


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


async def test_an_agent_is_read_back_field_by_field(tmp_path):
    spec = await registry(tmp_path).get("catalog")

    assert spec.name == "catalog"
    assert spec.display_name == "Catalogue"
    assert spec.db_role == "app_catalog"
    assert spec.tables == ["authors", "publishers", "books"]
    assert spec.status is AgentStatus.ACTIVE
    assert spec.system_prompt.startswith("You answer")


async def test_an_unknown_key_raises_rather_than_returning_none(tmp_path):
    """Every caller needs an agent to continue, so None would only be
    checked and re-raised at each call site."""
    with pytest.raises(UnknownAgentError):
        await registry(tmp_path).get("nope")


async def test_the_error_names_what_is_registered(tmp_path):
    """A typo in AGENT_KEY is the likeliest cause, and the fix is visible
    the moment the real names are printed."""
    with pytest.raises(UnknownAgentError, match="catalog"):
        await registry(tmp_path).get("catalogue")


async def test_only_routable_agents_are_listed(tmp_path):
    """A pending agent's Postgres role does not exist yet. Offering it as a
    tool produces a database error in the middle of someone's question."""
    reg = registry(
        tmp_path,
        agent(key="catalog", status="active"),
        agent(key="pending_one", db_role="app_pending", status="pending"),
        agent(key="disabled_one", db_role="app_disabled", status="disabled"),
    )
    assert [s.name for s in await reg.list_active()] == ["catalog"]


async def test_a_disabled_agent_is_still_readable_by_key(tmp_path):
    """Disabling takes an agent out of routing; it does not delete it. The
    console still has to be able to show it."""
    reg = registry(tmp_path, agent(status="disabled"))
    assert (await reg.get("catalog")).status is AgentStatus.DISABLED


async def test_the_active_list_is_ordered_by_name(tmp_path):
    """The orchestrator's tool list is built from this. Stable order means a
    prompt that does not change between restarts."""
    reg = registry(
        tmp_path,
        agent(key="zebra", db_role="app_zebra"),
        agent(key="alpha", db_role="app_alpha"),
    )
    assert [s.name for s in await reg.list_active()] == ["alpha", "zebra"]


async def test_the_file_is_read_once_at_construction(tmp_path):
    """No TTL, no invalidation. If the file changes, the process restarts -
    which also means a later edit cannot change a running agent's tables."""
    path = write(tmp_path, AGENT)
    reg = FileAgentRegistryAdapter(path)
    path.write_text(json.dumps({"agents": []}), encoding="utf-8")

    assert (await reg.get("catalog")).name == "catalog"


# --------------------------------------------------------------------------
# defaults - what a registry entry does not have to say
# --------------------------------------------------------------------------


async def test_status_defaults_to_pending(tmp_path):
    """A spec that forgets to say is treated as not-ready. The safe way
    round: provisioning is asynchronous."""
    spec = await registry(tmp_path, {k: v for k, v in AGENT.items() if k != "status"}).get("catalog")
    assert spec.status is AgentStatus.PENDING


async def test_type_defaults_to_sub_agent(tmp_path):
    assert (await registry(tmp_path).get("catalog")).type is AgentType.SUB_AGENT


async def test_display_name_falls_back_to_the_key(tmp_path):
    entry = {k: v for k, v in AGENT.items() if k != "display_name"}
    assert (await registry(tmp_path, entry).get("catalog")).display_name == "catalog"


async def test_the_history_table_is_infrastructure_not_registry_data(tmp_path):
    """Nobody registering an agent should have to know what a history table
    is, so it defaults rather than being required."""
    assert (await registry(tmp_path).get("catalog")).history_table == "agent_history"


async def test_omitting_allowed_tables_means_whatever_the_role_can_read(tmp_path):
    """The design's own default. None is not the same as an empty list."""
    entry = {k: v for k, v in AGENT.items() if k != "allowed_tables"}
    assert (await registry(tmp_path, entry).get("catalog")).tables is None


async def test_declared_tables_are_lowercased(tmp_path):
    """Unquoted SQL identifiers fold to lowercase, and the comparison against
    introspection has to be on the same footing."""
    spec = await registry(tmp_path, agent(allowed_tables=["Authors", "BOOKS"])).get("catalog")
    assert spec.tables == ["authors", "books"]


# --------------------------------------------------------------------------
# refusing a malformed file
# --------------------------------------------------------------------------


def test_a_missing_file_names_the_path_and_the_fix(tmp_path):
    with pytest.raises(RegistryError, match="AGENTS_REGISTRY_PATH"):
        FileAgentRegistryAdapter(tmp_path / "not_here.json")


def test_invalid_json_names_the_line(tmp_path):
    """These files are edited by hand and a trailing comma is the usual
    cause, so the position is most of the fix."""
    path = tmp_path / "agents.json"
    path.write_text('{"agents": [ {"key": "a"}, ]}', encoding="utf-8")

    with pytest.raises(RegistryError, match="line"):
        FileAgentRegistryAdapter(path)


def test_a_file_without_an_agents_list_is_refused(tmp_path):
    path = tmp_path / "agents.json"
    path.write_text('{"stuff": []}', encoding="utf-8")
    with pytest.raises(RegistryError, match="agents"):
        FileAgentRegistryAdapter(path)


def test_a_registry_with_no_agents_is_refused(tmp_path):
    """A service with none can answer nothing. Better at startup than on the
    first question."""
    with pytest.raises(RegistryError, match="no agents"):
        FileAgentRegistryAdapter(write(tmp_path))


def test_a_duplicate_key_is_refused(tmp_path):
    """Keeping the last silently would make an agent's tables depend on the
    order of a file nobody reads twice."""
    with pytest.raises(RegistryError, match="more than once"):
        registry(tmp_path, agent(), agent(allowed_tables=["members"]))


def test_an_entry_without_a_key_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="key"):
        registry(tmp_path, {k: v for k, v in AGENT.items() if k != "key"})


def test_an_entry_without_a_prompt_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="prompt"):
        registry(tmp_path, {k: v for k, v in AGENT.items() if k != "prompt"})


def test_an_entry_without_a_description_is_refused(tmp_path):
    """The orchestrator routes by description alone, so an agent without one
    can never be chosen - it would sit in the tool list, unreachable."""
    with pytest.raises(RegistryError, match="description"):
        registry(tmp_path, {k: v for k, v in AGENT.items() if k != "description"})


def test_an_unknown_status_is_refused_rather_than_defaulted(tmp_path):
    """Defaulting would turn a typo into a silently non-routable agent."""
    with pytest.raises(RegistryError, match="active"):
        registry(tmp_path, agent(status="enabled"))


def test_an_unrecognised_field_is_refused(tmp_path):
    """The security-shaped one. `allowed_table` ignored would read as "no
    restriction" - the agent would quietly get every table its role can
    read, which is the opposite of what the author was writing."""
    with pytest.raises(RegistryError, match="allowed_table"):
        registry(tmp_path, agent(allowed_table=["books"]))


async def test_a_comment_field_is_allowed(tmp_path):
    """seeds/agents.example.json explains the format inside the format, so
    an underscore-prefixed key has to survive the unknown-field check."""
    reg = registry(tmp_path, agent(_note="why this agent exists"))
    assert (await reg.get("catalog")).name == "catalog"


def test_an_empty_allowed_tables_is_refused(tmp_path):
    """Cannot be told apart from "I meant to fill this in", and silently
    means an agent that may read nothing."""
    with pytest.raises(RegistryError, match="empty"):
        registry(tmp_path, agent(allowed_tables=[]))


def test_allowed_tables_must_be_a_list_of_strings(tmp_path):
    with pytest.raises(RegistryError, match="list of table names"):
        registry(tmp_path, agent(allowed_tables="books"))


def test_a_non_object_entry_is_refused(tmp_path):
    with pytest.raises(RegistryError, match="not an object"):
        registry(tmp_path, "catalog")


# --------------------------------------------------------------------------
# identifiers that reach SQL
# --------------------------------------------------------------------------


def test_a_db_role_that_is_not_an_identifier_is_refused(tmp_path):
    """db_role is interpolated into SET LOCAL ROLE, where it cannot be a
    parameter. This is the only thing between a registry file and that
    statement."""
    with pytest.raises(RegistryError):
        registry(tmp_path, agent(db_role="app_catalog; DROP TABLE books"))


def test_a_key_that_is_not_an_identifier_is_refused(tmp_path):
    with pytest.raises(RegistryError):
        registry(tmp_path, agent(key="Catalog Agent"))


def test_a_history_table_that_is_not_an_identifier_is_refused(tmp_path):
    with pytest.raises(RegistryError):
        registry(tmp_path, agent(history_table='history"; --'))


def test_an_overlong_description_is_refused(tmp_path):
    """Every routable agent's description is read on every routing decision,
    so their combined length is a per-turn cost paid by all of them."""
    with pytest.raises(RegistryError, match="description"):
        registry(tmp_path, agent(description="x" * 1001))


# --------------------------------------------------------------------------
# the file that ships with the repo
# --------------------------------------------------------------------------


async def test_the_example_registry_loads():
    """seeds/agents.example.json is documentation people copy. If it stopped
    parsing, the first thing anyone tried would fail."""
    reg = FileAgentRegistryAdapter("seeds/agents.example.json")
    assert [s.name for s in await reg.list_active()] == ["catalog", "circulation"]


async def test_the_example_registry_matches_the_roles_seed():
    """seeds/003_roles.sql grants these exact tables. The two files are a
    deployment's two halves and drift between them is what verify_grants
    exists to catch - catching it here as well means catching it without a
    database."""
    reg = FileAgentRegistryAdapter("seeds/agents.example.json")

    assert set((await reg.get("catalog")).tables) == {"authors", "publishers", "books"}
    assert set((await reg.get("circulation")).tables) == {
        "branches", "members", "loans", "books",
    }
