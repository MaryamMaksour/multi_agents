# domain/exceptions.py

class DomainError(Exception):
    """Base class for all domain-level errors."""
    pass


class LLMRequestError(DomainError):
    """Raised when a request to the LLM fails."""
    pass


class EmbeddingError(DomainError):
    """Raised when an embedding request fails."""
    pass


class UnknownToolError(DomainError):
    """Raised when an unrecognized tool name is requested."""
    pass


class ToolExecutionError(DomainError):
    """Raised when a tool fails while executing."""
    pass


class DatabaseError(DomainError):
    """Raised when a database operation fails."""
    pass


class CacheError(DomainError):
    """Raised when a cache operation fails."""
    pass


class HistoryError(DomainError):
    """Raised when a history operation fails."""
    pass

class SessionBusyError(DomainError):
    """Raised when a session's lock could not be acquired."""
    pass


class RegistryError(DomainError):
    """Raised when the agent registry cannot be read or is malformed.

    A startup error on purpose. A registry that half-loads leaves a service
    running with an incomplete set of agents, which looks like a routing bug
    rather than a configuration one.
    """
    pass


class UnknownAgentError(DomainError):
    """Raised when no agent is registered under a key.

    Not a None return: every caller of get() needs an agent to continue, so
    None would only be checked and re-raised at each call site.
    """
    pass


class GrantMismatchError(DomainError):
    """Raised when an agent's declared tables disagree with its role's GRANTs.

    Refuses startup rather than picking a winner. The declared list and the
    GRANTs are two statements of the same fact, and when they disagree one of
    them is wrong - continuing means either an agent that cannot read what it
    was promised, or one that can read more than anybody wrote down.
    """
    pass
