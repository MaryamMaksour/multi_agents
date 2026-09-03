"""Request and response shapes for the HTTP edge.

Pydantic here and nowhere else. The domain speaks in entities, and a model
that leaked inward would make the core depend on the transport it happens to
be reached through today.

The sub-agent's request shape is not a choice: it is whatever
HttpDelegateToolAdapter posts, because the orchestrator is the caller. Any
divergence would be a 422 that only appears once two components are running
together.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from libs.agent_core import config

# One question is an embedding call plus several model calls priced per
# token, so the length of the text is a cost limit as much as a sanity
# check. Rejected here, by the schema, rather than after the first model
# call: a 422 costs nothing.
MAX_QUESTION_CHARS = config.MAX_QUESTION_CHARS


class DelegateContext(BaseModel):
    """The correlation values the caller carries, not the model's choices.

    `cursor` is the only field here the model ever sets, and it sets it by
    passing back a value it was given. `turn_id` ties a sub-agent's history
    rows to the orchestrator turn that caused them, which is what makes a
    trace readable after the fact.
    """

    cursor: Optional[str] = Field(default=None, max_length=100_000)
    turn_id: Optional[str] = Field(default=None, max_length=200)


class RunRequest(BaseModel):
    """What the orchestrator posts to a sub-agent.

    Mirrors HttpDelegateToolAdapter.call_tool exactly.
    """

    session_id: str = Field(default="", max_length=200)
    user_input: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    context: DelegateContext = Field(default_factory=DelegateContext)


class AskRequest(BaseModel):
    """What a person (or the console) posts to the orchestrator."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field(min_length=1, max_length=200)


class DelegatedQuestion(BaseModel):
    """One question the orchestrator sent to one agent.

    Returned so that a wrong answer can be attributed. The orchestrator
    rewrites the user's question into a self-contained one before delegating,
    and that rewrite can drop a constraint - asked for novels it may send
    "how many English books", and the agent then answers that question
    correctly. Without this, the two failures look identical from outside and
    the wrong component gets changed.
    """

    agent: str
    question: str


class TurnResponse(BaseModel):
    """One turn's answer, plus what the caller needs to continue it.

    `pagination` is per tool rather than a single cursor because a turn may
    have paged through more than one, and collapsing them would make "get the
    next page" ambiguous about which.
    """

    answer: str
    session_id: str
    turn_id: str
    pagination: dict[str, Any] = Field(default_factory=dict)

    # Empty for a sub-agent, which delegates to nobody.
    delegated: list[DelegatedQuestion] = Field(default_factory=list)


class AgentSummary(BaseModel):
    key: str
    display_name: str
    description: str
    status: str


class HealthResponse(BaseModel):
    """Deliberately more than "ok".

    A sub-agent that is up but reading the wrong tables is the failure this
    whole design is built to prevent, so the health endpoint reports the
    tables it actually resolved. That makes a misconfiguration visible from
    outside the process, without a query and without logs.
    """

    status: str
    kind: str
    agent: Optional[str] = None
    tables: list[str] = Field(default_factory=list)
    routes_to: list[str] = Field(default_factory=list)

    # Which model is answering. A process running a different model than the
    # deployment believes is a quality difference with no other symptom.
    model: str = ""
