"""ProviderSpec - identity, status, and the validation that guards SQL.

db_role ends up interpolated into SET LOCAL ROLE, where it cannot be passed
as a parameter, so the check in __post_init__ is the only thing between a
registry entry and that statement. Rejecting at construction means a bad
value cannot be carried around and reach SQL somewhere far from here.
"""

from __future__ import annotations

import pytest

from domain.entities.provider_spec import AgentStatus, AgentType, ProviderSpec


def spec(**overrides) -> ProviderSpec:
    kwargs = dict(
        name="catalog",
        type=AgentType.SUB_AGENT,
        system_prompt="You answer questions about the catalogue.",
        history_table="history_catalog",
        tools=["db_execute", "get_table_schema"],
        description="Books, authors and publishers.",
        db_role="app_catalog",
    )
    kwargs.update(overrides)
    return ProviderSpec(**kwargs)


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------


def test_defaults_to_pending_not_active():
    """Provisioning is asynchronous: an agent is registered before its role
    exists. A spec that omits status must be treated as not-ready, or the
    orchestrator routes to an agent whose role has not been granted yet."""
    assert spec().status is AgentStatus.PENDING


def test_only_an_active_agent_is_routable():
    assert not spec(status=AgentStatus.PENDING).is_routable
    assert not spec(status=AgentStatus.DISABLED).is_routable
    assert spec(status=AgentStatus.ACTIVE).is_routable


def test_status_values_are_the_strings_a_registry_stores():
    assert {s.value for s in AgentStatus} == {"pending", "active", "disabled"}


# --------------------------------------------------------------------------
# agent type
# --------------------------------------------------------------------------


def test_agent_type_describes_a_role_not_an_identity():
    """A closed set - an agent either orchestrates or is delegated to. The set
    of agents is the opposite, open by design, which is why name is a string
    and not another enum member."""
    assert {t.value for t in AgentType} == {"sub_agent", "orchestrator"}


def test_any_name_is_acceptable_as_long_as_it_is_well_formed():
    for name in ("catalog", "circulation", "hr_payroll", "team_9"):
        assert spec(name=name).name == name


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "Catalog",              # uppercase
    "cat-alog",             # dash
    "cat alog",             # space
    "9catalog",             # leading digit
    "c",                    # too short
    "",
    None,
    "catalog; DROP ROLE x",
    "catalog--",
])
def test_rejects_a_malformed_name(name):
    with pytest.raises(ValueError):
        spec(name=name)


@pytest.mark.parametrize("role", [
    "app catalog",
    "app-catalog",
    "app_catalog; DROP ROLE postgres",
    "x'; DROP TABLE books; --",
    "",
    None,
    "APP_CATALOG",
])
def test_rejects_a_db_role_that_could_not_be_used_as_an_identifier(role):
    with pytest.raises(ValueError):
        spec(db_role=role)


def test_the_error_names_the_offending_value():
    with pytest.raises(ValueError, match="db_role"):
        spec(db_role="app catalog")
    with pytest.raises(ValueError, match="name"):
        spec(name="Bad Name")


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def test_tables_is_optional_because_the_grants_are_the_real_list():
    """None means "whatever the role can read". The value here is a
    redundancy check against introspection, never the source of truth."""
    assert spec().tables is None


def test_tables_can_be_declared_for_cross_checking():
    s = spec(tables=["books", "authors"])
    assert s.tables == ["books", "authors"]


def test_prompt_and_description_are_separate_fields():
    """They are trusted differently: the prompt cannot widen access because
    the GRANT is the boundary, but the description steers routing."""
    s = spec(system_prompt="P", description="D")
    assert s.system_prompt == "P"
    assert s.description == "D"
