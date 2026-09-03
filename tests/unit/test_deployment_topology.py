"""What the compose file publishes, which is a security decision.

A sub-agent answers /run with no authentication - it trusts its caller, and
its caller is the orchestrator on the internal network. Publishing one on all
interfaces puts the whole delegation layer on the network for anyone who can
reach the host, while a loopback binding still leaves
`curl localhost:8001/health` working, which is what the published port was
for.

Parsed as YAML rather than grepped, so a reformat of the file does not fail
the test and a moved port does not pass it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

COMPOSE = Path(__file__).resolve().parents[2] / "deploy" / "docker-compose.yml"


@pytest.fixture(scope="module")
def services() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]


def sub_agents(services: dict) -> dict:
    return {
        name: service for name, service in services.items()
        if str(service.get("environment", {}).get("AGENT_KEY", "")).strip()
    }


def test_the_sub_agents_are_bound_to_loopback(services):
    assert sub_agents(services), "the topology should still have sub-agents"

    for name, service in sub_agents(services).items():
        for published in service.get("ports", []):
            assert str(published).startswith("127.0.0.1:"), (
                f"{name} publishes {published} on every interface; /run has no "
                "authentication because it trusts the orchestrator"
            )


def test_the_orchestrator_is_the_one_published_service(services):
    """It is the end that talks to a person, so it is the only one that
    should be reachable from off the host."""
    orchestrator = services["orchestrator"]

    assert any(str(p).endswith("8000:8000") and not str(p).startswith("127.0.0.1")
               for p in orchestrator.get("ports", []))


def test_every_agent_runs_the_same_image(services):
    """One image, many roles: what a container becomes is AGENT_KEY, not a
    build. If the image were per agent, registering one would mean a build
    to keep in step with the registry, and the two would drift."""
    images = {
        service.get("image") or str(service.get("build"))
        for name, service in services.items()
        if name == "orchestrator" or name in sub_agents(services)
    }
    assert len(images) == 1, f"agents run different images: {images}"
