"""AgentRegistryPort backed by a JSON file. Phase 1.

Deliberately the simplest thing that satisfies the port. It exists so the
core is deployment-agnostic from the first commit without also requiring the
registry table, the provisioner, and the admin API on day one.

The phase 3 adapter reads the same shape from Postgres. Because both sit
behind AgentRegistryPort, swapping them touches only the composition root.

Everything is read and validated once, in the constructor. The file cannot
change under a running process in any way worth supporting - if it changes,
the process restarts - so there is no cache, no TTL and no invalidation here.
That machinery belongs in the Postgres adapter, where the data really does
change underneath.

Validation is strict and happens at load, because the alternative is a
service that starts fine and then fails on somebody's question. A malformed
registry is a deployment mistake, and deployment mistakes should be loud and
early.

See seeds/agents.example.json for the file format.
"""

from __future__ import annotations

import json
from pathlib import Path

from domain.entities.provider_spec import AgentStatus, AgentType, ProviderSpec
from domain.exceptions import RegistryError, UnknownAgentError

# Keys the loader understands. Anything else is a typo until proven
# otherwise: a file saying "allowed_table" would silently grant the agent
# every table its role can read, which is the wrong way for a typo to fail.
_KNOWN_FIELDS = frozenset({
    "key", "display_name", "description", "prompt", "db_role",
    "allowed_tables", "status", "type", "history_table", "tools",
})

# Anything starting with _ is a comment. seeds/agents.example.json uses
# "_comment" to explain the format inside the format.
_COMMENT_PREFIX = "_"


class FileAgentRegistryAdapter:
    """Loads agent definitions from a JSON file on disk.

    Read-only, and not by accident. Registration writes go through the
    provisioner, which holds credentials that can CREATE ROLE and GRANT -
    privileges nothing serving requests should have. Adding a write method
    here later would move that boundary, and moving it should be a decision
    rather than a convenience.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._agents: dict[str, ProviderSpec] = self._load()

    # -- reading ----------------------------------------------------------

    async def get(self, key: str) -> ProviderSpec:
        """The agent registered under `key`.

        Raises UnknownAgentError rather than returning None: every caller
        needs an agent to continue, so None would only be checked and
        re-raised at each call site.
        """
        try:
            return self._agents[key]
        except KeyError:
            raise UnknownAgentError(
                f"No agent registered as {key!r} in {self._path}. "
                f"Registered: {', '.join(sorted(self._agents)) or 'none'}."
            ) from None

    async def list_active(self) -> tuple[ProviderSpec, ...]:
        """Every agent the orchestrator may route to, by name.

        Filtered on `is_routable` rather than on `status == ACTIVE`, so a
        fourth status changes the rule in the entity and not here.

        A pending agent is excluded because its Postgres role does not exist
        yet - offering it as a tool would produce a database error in the
        middle of answering somebody's question.
        """
        return tuple(
            spec for spec in sorted(self._agents.values(), key=lambda s: s.name)
            if spec.is_routable
        )

    # -- loading ----------------------------------------------------------

    def _load(self) -> dict[str, ProviderSpec]:
        raw = self._read_file()
        entries = raw.get("agents")

        if not isinstance(entries, list):
            raise RegistryError(
                f"{self._path} has no 'agents' list. Expected "
                '{"agents": [ ... ]} - see seeds/agents.example.json.'
            )

        agents: dict[str, ProviderSpec] = {}
        for position, entry in enumerate(entries):
            spec = self._build(entry, position)
            if spec.name in agents:
                # Silently keeping the last would mean an agent's tables and
                # prompt depend on the order of a file nobody reads twice.
                raise RegistryError(
                    f"{self._path} registers {spec.name!r} more than once. "
                    "Agent keys identify routes and history rows; they have to "
                    "be unique."
                )
            agents[spec.name] = spec

        if not agents:
            raise RegistryError(
                f"{self._path} registers no agents. A service with none can "
                "answer nothing, which is worth failing at startup rather than "
                "on the first question."
            )
        return agents

    def _read_file(self) -> dict:
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise RegistryError(
                f"No agent registry at {self._path}. Set AGENTS_REGISTRY_PATH, "
                "or copy seeds/agents.example.json and edit it."
            ) from None
        except OSError as e:
            raise RegistryError(f"Cannot read {self._path}: {e}") from e

        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            # Naming the line and column matters: these files are edited by
            # hand and a trailing comma is the usual cause.
            raise RegistryError(
                f"{self._path} is not valid JSON: {e.msg} at line {e.lineno} "
                f"column {e.colno}."
            ) from e

        if not isinstance(raw, dict):
            raise RegistryError(
                f"{self._path} should contain a JSON object, not a "
                f"{type(raw).__name__}."
            )
        return raw

    def _build(self, entry, position: int) -> ProviderSpec:
        where = f"{self._path} agents[{position}]"

        if not isinstance(entry, dict):
            raise RegistryError(f"{where} is a {type(entry).__name__}, not an object.")

        unknown = {
            k for k in entry
            if k not in _KNOWN_FIELDS and not k.startswith(_COMMENT_PREFIX)
        }
        if unknown:
            raise RegistryError(
                f"{where} has unrecognised field(s): {', '.join(sorted(unknown))}. "
                "Rejected rather than ignored - a misspelt 'allowed_tables' would "
                "otherwise read as 'no restriction'."
            )

        key = entry.get("key")
        if not isinstance(key, str) or not key:
            raise RegistryError(f"{where} has no 'key'.")

        tables = self._tables(entry.get("allowed_tables"), where)

        try:
            # ProviderSpec validates name, db_role and history_table as SQL
            # identifiers, and bounds the description. Letting it raise and
            # re-wrapping here keeps one copy of those rules.
            return ProviderSpec(
                name=key,
                display_name=self._text(entry, "display_name", where, required=False),
                description=self._text(entry, "description", where),
                system_prompt=self._text(entry, "prompt", where),
                db_role=self._text(entry, "db_role", where),
                type=self._enum(AgentType, entry.get("type"), AgentType.SUB_AGENT, "type", where),
                status=self._enum(AgentStatus, entry.get("status"), AgentStatus.PENDING, "status", where),
                tables=tables,
                **({"history_table": entry["history_table"]} if "history_table" in entry else {}),
                **({"tools": list(entry["tools"])} if "tools" in entry else {}),
            )
        except (ValueError, TypeError) as e:
            raise RegistryError(f"{where}: {e}") from e

    @staticmethod
    def _text(entry: dict, field: str, where: str, *, required: bool = True) -> str:
        value = entry.get(field, "")
        if not isinstance(value, str):
            raise RegistryError(f"{where} field {field!r} should be a string.")
        if required and not value.strip():
            raise RegistryError(f"{where} has no {field!r}.")
        return value

    @staticmethod
    def _enum(enum_class, value, default, field: str, where: str):
        if value is None:
            return default
        try:
            return enum_class(value)
        except ValueError:
            allowed = ", ".join(repr(m.value) for m in enum_class)
            raise RegistryError(
                f"{where} has {field}={value!r}. Expected one of {allowed}."
            ) from None

    @staticmethod
    def _tables(value, where: str) -> list[str] | None:
        """The declared table list, or None for "whatever the role can read".

        Absent and empty are deliberately different. Absent defers to the
        GRANTs. Empty is a list that says the agent may read nothing, which
        is a mistake worth naming rather than treating as "no opinion".
        """
        if value is None:
            return None
        if not isinstance(value, list) or not all(isinstance(t, str) for t in value):
            raise RegistryError(
                f"{where} field 'allowed_tables' should be a list of table names."
            )
        if not value:
            raise RegistryError(
                f"{where} has an empty 'allowed_tables'. Omit the field to defer "
                "to the role's GRANTs; an empty list would mean an agent that may "
                "read nothing."
            )
        return [t.lower() for t in value]
