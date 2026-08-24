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