from typing import Any 
import json

import redis.asyncio as redis

from dataclasses import asdict

from domain.exceptions import CacheError
from domain.entities.chat_message import ChatMessage, Role, ToolCall

class RedisCacheAdapter:
    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client
        self._locks: dict[str, Any] = {} # key -> redis lock object, between acquire و release


    async def get(self, key: str) -> Any:
        """Retrieve a value from the cache by its key.
        
            Raises:
                CacheError: if the cache call fails.
        """
        try:
            value = await self._redis.get(key)

            if value is None:
                return None

            new_messages = json.loads(value)

            return [
                 ChatMessage(
                      role=Role(msg["role"]),
                      content=msg.get("content"),
                      tool_calls=[ToolCall(**tc) for tc in msg["tool_calls"]] if msg.get("tool_calls") else None,
                      tool_call_id= msg.get("tool_call_id"),
                      name = msg.get("name")
                 ) 
                 for msg in new_messages
            ]

        except Exception as e:
           raise CacheError(f"error {e} while getting cache for key: {key}") from e

    async def set(self, key: str, value: Any, ttl: int) -> None:
            """Store a value in the cache with a time-to-live (TTL).
    
            Raises:
                CacheError: if the cache call fails.
            """

            try:
                  serializable = [
                        {**asdict(msg), "role": msg.role.value} for msg in value
                  ] # convert ChatMessage list to JSON-serializable dicts.

                  _value = json.dumps(serializable)

                  await self._redis.set(key, _value, ex=ttl)

            except Exception as e:
                  raise CacheError(f"error {e} while set on cache  key: {key} ") from e
    
    async def delete(self, key: str) -> None:
            """Remove a value from the cache by its key.
    
            Raises:
                CacheError: if the cache call fails.
            """

            try:
                  await self._redis.delete(key)

            except Exception as e:
                raise CacheError(f"error {e} while deleting from cache  key: {key} ") from e
                
            
    
    async def acquire_lock(self, key: str, timeout: int) -> bool:
            """Try to acquire a lock. Returns True if acquired, False if already held.
    
            Raises:
                CacheError: if the cache call fails.
            """
            lock = self._redis.lock(key, timeout = timeout)

            try:
                acquired =  await lock.acquire(blocking=False)
            except Exception as e:
                raise CacheError(f"error {e} while acquiring lock for key: {key}") from e

            if acquired:
                self._locks[key] = lock
                return True

            return False

    
    async def release_lock(self, key: str) -> None:
            """Release a previously acquired lock.
    
            Raises:
                CacheError: if the cache call fails.
            """
            lock = self._locks.pop(key, None)

            if lock is None:
                 return
            try:
                await lock.release()
            except Exception as e:
                 raise CacheError(f"error {e} while releasing lock for key: {key}") from e



    