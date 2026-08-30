# One image. What a container becomes is decided by AGENT_KEY at runtime, not
# at build time - so the orchestrator and every sub-agent are the same bytes,
# and there is no per-agent build to keep in step with the registry.

FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so editing source does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Not root. A process whose entire job is running SQL somebody else wrote has
# no business owning its own filesystem.
RUN useradd --create-home --uid 10001 agent && chown -R agent /app
USER agent

EXPOSE 8000

# Reads AGENT_KEY from the environment: set it and this serves that agent,
# leave it empty and this is the orchestrator.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
