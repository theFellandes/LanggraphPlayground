"""Redis-backed distributed lock helpers used across rag_qa_api_pro replicas.

Mirrors lesson 31's pattern (SETNX + Lua-ownership release). Kept in its
own module so the graph and the API can both import it without circular
deps.
"""

from __future__ import annotations

import asyncio
import uuid


# Ownership-checked release (Lua → atomic in Redis).
RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


async def acquire(redis_client, key: str, owner: str, ttl_seconds: int = 60) -> bool:
    return bool(await redis_client.set(f"lock:{key}", owner, nx=True, ex=ttl_seconds))


async def release(redis_client, key: str, owner: str) -> bool:
    return bool(await redis_client.eval(RELEASE_LUA, 1, f"lock:{key}", owner))


async def wait_for_release(redis_client, key: str, max_wait: float = 60.0) -> bool:
    """Spin until the lock is gone, then return True. Returns False on timeout."""
    deadline = asyncio.get_event_loop().time() + max_wait
    while asyncio.get_event_loop().time() < deadline:
        if not await redis_client.exists(f"lock:{key}"):
            return True
        await asyncio.sleep(0.5)
    return False


def new_owner() -> str:
    return str(uuid.uuid4())
