from typing import Any, Protocol

class CachePort(Protocol):

    async def get(self, key: str) -> Any:
        """Retrieve a value from the cache by its key.

        Raises:
            CacheError: if the cache call fails.
        """
        ...
        
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Store a value in the cache with a time-to-live (TTL).

        Raises:
            CacheError: if the cache call fails.
        """
        ...

    async def delete(self, key: str) -> None:
        """Remove a value from the cache by its key.

        Raises:
            CacheError: if the cache call fails.
        """
        ...

    async def lock(self, key: str) -> None:
        """Acquire a lock for a specific key in the cache.

        Raises:
            CacheError: if the cache call fails.
        """
        ...

    async def acquire_lock(self, key: str, timeout: int) -> bool:
        """Try to acquire a lock. Returns True if acquired, False if already held.

        Raises:
            CacheError: if the cache call fails.
        """
        ...

    async def release_lock(self, key: str) -> None:
        """Release a previously acquired lock.

        Raises:
            CacheError: if the cache call fails.
        """
        ...
        