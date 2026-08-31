from typing import Any 
import json

import redis.asyncio as redis

from dataclasses import asdict

from domain.exceptions import CacheError
from domain.entities.chat_message import ChatMessage, Role, ToolCall

# Values and locks live in separate key namespaces, and they have to.
#
# RunAgentTurn locks a session and then reads that session's window, passing
# the same key to both - which is the right shape for a port: "the lock for
# this session" and "the window for this session" are two facts about one
# thing. But redis-py's Lock stores its token at exactly the key it is given,
# so without a prefix the lock overwrites the window, and get() then tries to
# json.loads a uuid. It fails on the very first question of every new session
# with "Extra data: line 1 column 9", which names neither locks nor sessions.
VALUE_PREFIX = "session:"
LOCK_PREFIX = "lock:"


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
            value = await self._redis.get(VALUE_PREFIX + key)

            # An empty window, not None. A session that has said nothing yet
            # and a session that does not exist are the same thing to the
            # caller, and returning None makes every caller write the same
            # check before it can concatenate.
            if value is None:
                return []

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

                  await self._redis.set(VALUE_PREFIX + key, _value, ex=ttl)

            except Exception as e:
                  raise CacheError(f"error {e} while set on cache  key: {key} ") from e
    
    async def delete(self, key: str) -> None:
            """Remove a value from the cache by its key.
    
            Raises:
                CacheError: if the cache call fails.
            """

            try:
                  await self._redis.delete(VALUE_PREFIX + key)

            except Exception as e:
                raise CacheError(f"error {e} while deleting from cache  key: {key} ") from e
                
            
    
    async def acquire_lock(self, key: str, timeout: int) -> bool:
            """Try to acquire a lock. Returns True if acquired, False if already held.
    
            Raises:
                CacheError: if the cache call fails.
            """
            lock = self._redis.lock(LOCK_PREFIX + key, timeout = timeout)

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



    