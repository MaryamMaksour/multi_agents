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

# Upper bounds on everything a caller controls.
#
# The endpoint had a floor and no ceiling, and the two are not symmetric. An
# empty question is a mistake; a four-megabyte one is a bill. It is embedded,
# sent to the model as the last message of a prompt that already carries the
# tool schemas, stored in the conversation window, and written to history -
# so one request costs an embedding call, a model call priced per token, and
# a row that stays for the retention window. Nothing above these limits is a
# real question, and everything above them is someone finding that out.
#
# Generous rather than tight: 8000 characters is several pages of Arabic, far
# more than any question, and small enough that no accident is expensive.
MAX_QUESTION_CHARS = 8_000

# A session id is a correlation value that becomes a Redis key and a history
# column. It never needs to be long, and an unbounded one is an unbounded key.
MAX_SESSION_ID_CHARS = 200

# A cursor is issued by this system - compressed, base64 - and handed back
# unchanged. The cap is on what a caller may return, not on what is produced;
# a longer one was never issued here.
MAX_CURSOR_CHARS = 8_000


class DelegateContext(BaseModel):
    """The correlation values the caller carries, not the model's choices.

    `cursor` is the only field here the model ever sets, and it sets it by
    passing back a value it was given. `turn_id` ties a sub-agent's history
    rows to the orchestrator turn that caused them, which is what makes a
    trace readable after the fact.
    """

    cursor: Optional[str] = Field(default=None, max_length=MAX_CURSOR_CHARS)
    turn_id: Optional[str] = Field(default=None, max_length=MAX_SESSION_ID_CHARS)


class RunRequest(BaseModel):
    """What the orchestrator posts to a sub-agent.

    Mirrors HttpDelegateToolAdapter.call_tool exactly.
    """

    session_id: str = Field(default="", max_length=MAX_SESSION_ID_CHARS)
    user_input: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    context: DelegateContext = Field(default_factory=DelegateContext)


class AskRequest(BaseModel):
    """What a person (or the console) posts to the orchestrator."""

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str = Field(min_length=1, max_length=MAX_SESSION_ID_CHARS)


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

    `model` is here for the same reason and a smaller one: comparing two
    models means restarting the containers with QWEN_MODEL changed, and a
    stale container answering with the old one is a silent way to attribute a
    result to the wrong model. Asking is one curl.
    """

    status: str
    kind: str
    agent: Optional[str] = None
    model: str = ""
    thinking: bool = False
    tables: list[str] = Field(default_factory=list)
    routes_to: list[str] = Field(default_factory=list)
