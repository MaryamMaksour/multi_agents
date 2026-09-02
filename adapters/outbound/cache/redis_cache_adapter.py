from typing import Any
import json
import logging

import redis.asyncio as redis

from domain.exceptions import CacheError
from domain.entities.chat_message import ChatMessage, from_plain, to_plain
from libs.agent_core.logging_setup import log_event

logger = logging.getLogger(__name__)

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

# What a stored payload is, written next to it.
#
# CachePort carries two different shapes: a conversation window, which is a
# list of ChatMessage, and an embedding token's vector, which is a list of
# floats. This adapter used to assume the first - `asdict(msg)` over whatever
# it was given - so embed_query_tool could not store a vector at all:
#
#     CacheError: asdict() should be called on dataclass instances
#
# which named neither vectors nor the tool that was storing one. Sniffing the
# shape on the way out would work and would be guesswork; a tag written on the
# way in is one field and says what the value is.
MESSAGES = "messages"
VALUE = "value"


def _is_message_window(value: Any) -> bool:
    """Whether this is a conversation window rather than a plain value.

    Checked on the way in, where the caller's type is still visible, rather
    than sniffed on the way out. An empty list is not a window: there is
    nothing to convert either way, and calling it one would mean a stored
    empty vector came back as an empty message list.
    """
    return bool(value) and all(isinstance(item, ChatMessage) for item in value)


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
                log_event(logger, "cache.miss", level=logging.DEBUG, key=key)
                return []

            stored = json.loads(value)

            if stored.get("kind") != MESSAGES:
                log_event(logger, "cache.hit", level=logging.DEBUG,
                          key=key, kind=VALUE)
                return stored["data"]

            messages = [from_plain(msg) for msg in stored["data"]]
            log_event(logger, "cache.hit", level=logging.DEBUG,
                      key=key, kind=MESSAGES, messages=len(messages))
            return messages

        except Exception as e:
           log_event(logger, "cache.error", level=logging.ERROR,
                     operation="get", key=key, error=type(e).__name__)
           raise CacheError(f"error {e} while getting cache for key: {key}") from e

    async def set(self, key: str, value: Any, ttl: int) -> None:
            """Store a value in the cache with a time-to-live (TTL).
    
            Raises:
                CacheError: if the cache call fails.
            """

            try:
                  if _is_message_window(value):
                        payload = {"kind": MESSAGES,
                                   "data": [to_plain(msg) for msg in value]}
                  else:
                        # A vector, or anything else JSON can carry. Stored
                        # as it is and handed back as it is.
                        payload = {"kind": VALUE, "data": value}

                  await self._redis.set(VALUE_PREFIX + key, json.dumps(payload), ex=ttl)

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

            # Not an error - the caller turns it into a 409 - but the line
            # that distinguishes "two people asked at once", which is normal,
            # from "a lock is never released", which is not.
            log_event(logger, "cache.lock_contended", level=logging.WARNING, key=key)
            return False

    
    async def release_lock(self, key: str) -> None:
            """Release a previously acquired lock. Never raises.

            Deliberately the one method here that does not turn a failure into
            a CacheError - see the comment below.
            """
            lock = self._locks.pop(key, None)

            if lock is None:
                 return
            try:
                await lock.release()
            except Exception as e:
                # Logged, not raised, and that is a bug fix rather than a
                # style choice. RunAgentTurn releases in a `finally`, so an
                # exception here replaces whatever the turn was returning -
                # including a correct answer. And the most likely cause is
                # precisely a turn that took longer than the lock's timeout:
                # the lock expired on its own, someone else may hold it now,
                # and there is nothing left to release. Failing the answer
                # over that is the worst available outcome.
                log_event(logger, "cache.lock_release_failed",
                          level=logging.WARNING, key=key, error=type(e).__name__,
                          detail=str(e))



    