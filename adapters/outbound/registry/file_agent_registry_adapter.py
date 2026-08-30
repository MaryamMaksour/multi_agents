"""AgentRegistryPort backed by a JSON file. Phase 1.

Deliberately the simplest thing that satisfies the port. It exists so the
core is deployment-agnostic from the first commit without also requiring the
registry table, the provisioner, and the admin API on day one.

The phase 3 adapter reads the same shape from Postgres. Because both sit
behind AgentRegistryPort, swapping them touches only the composition root.

See seeds/agents.example.json for the file format.
"""

from __future__ import annotations


class FileAgentRegistryAdapter:
    """Loads agent definitions from a JSON file on disk.

    Constructor takes:
        path  str | Path  - location of the registry file

    TODO: implement get() and list_active().

    TODO: load once at construction, not per call. The file cannot change
    under a running process in any way worth supporting - if it changes, the
    process restarts. Do not build TTL machinery here; that belongs in the
    Postgres adapter where the data really does change underneath.

    TODO: validate on load and fail loudly - a malformed registry should stop
    the service at startup, not produce a confusing error on the first
    request. Check at minimum:
      - every key matches AGENT_KEY_PATTERN
      - keys are unique
      - status is one of pending / active / disabled
      - db_role is present and a valid identifier

    TODO: allowed_tables in the file is a redundancy check, not the source of
    truth. After introspecting through the agent's role, compare the two sets
    and refuse to start on a mismatch, naming both sides. A drift here means
    either the file or the GRANTs are wrong, and guessing which is worse than
    stopping.
    """

    # TODO: implement
    ...
