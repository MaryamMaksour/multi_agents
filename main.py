"""Entry point. One image; what it becomes is decided by AGENT_KEY.

    AGENT_KEY=catalog uvicorn main:app     one sub-agent, serving /run
    uvicorn main:app                       the orchestrator, serving /ask

Nothing here but the wiring to uvicorn. The process's actual shape is decided
in libs/agent_core/composition.py, and everything it needs is read from the
environment and the registry - so the same image, with the same command,
becomes a different agent by changing one variable.
"""

from __future__ import annotations

from adapters.inbound.http.app import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    from libs.agent_core import config

    uvicorn.run(app, host=config.HTTP_HOST, port=config.HTTP_PORT)
