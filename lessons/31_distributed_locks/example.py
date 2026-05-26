"""Lesson 31 · Distributed locks — Redis-backed.

Prerequisites:
    docker compose -f ../29_vector_databases/docker-compose.yml up -d redis
    uv add redis

Run:
    uv run python -m lessons.31_distributed_locks.example
    uv run python -m lessons.31_distributed_locks.example --setnx
    uv run python -m lessons.31_distributed_locks.example --heartbeat
    uv run python -m lessons.31_distributed_locks.example --fencing
    uv run python -m lessons.31_distributed_locks.example --idempotency
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid

from shared.pretty import console, section

REDIS_URL = "redis://localhost:6379"


# Atomic ownership-checked release.
RELEASE_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


async def _redis():
    try:
        import redis.asyncio as redis
    except ImportError as e:
        console.print("[yellow]redis package missing. Install: uv add redis[/]")
        raise SystemExit(1) from e
    return redis.from_url(REDIS_URL)


# --- Pattern A · SETNX -------------------------------------------------------
async def demo_setnx() -> None:
    section("A · Redis SETNX + Lua-release")

    r = await _redis()

    async def acquire(key: str, owner: str, ttl: int = 10) -> bool:
        return bool(await r.set(f"lock:{key}", owner, nx=True, ex=ttl))

    async def release(key: str, owner: str) -> bool:
        result = await r.eval(RELEASE_LUA, 1, f"lock:{key}", owner)
        return result == 1

    async def worker(name: str):
        owner = str(uuid.uuid4())
        if await acquire("demo", owner, ttl=5):
            console.print(f"  [green]{name}[/] acquired (owner={owner[:8]})")
            await asyncio.sleep(0.5)
            ok = await release("demo", owner)
            console.print(f"  [green]{name}[/] released, ok={ok}")
        else:
            console.print(f"  [red]{name}[/] could not acquire")

    # Three workers racing. Expect 1 wins, 2 lose.
    await asyncio.gather(worker("A"), worker("B"), worker("C"))
    await r.aclose()


# --- Pattern B · heartbeat ---------------------------------------------------
async def demo_heartbeat() -> None:
    section("B · Heartbeat — extend TTL while work is in progress")

    r = await _redis()
    key = "demo_hb"
    owner = str(uuid.uuid4())
    ttl = 3

    if not await r.set(f"lock:{key}", owner, nx=True, ex=ttl):
        console.print("could not acquire lock (already taken — try later)")
        await r.aclose()
        return

    async def heartbeat():
        while True:
            await asyncio.sleep(ttl / 3)
            extended = await r.expire(f"lock:{key}", ttl)
            console.print(f"  [dim]heartbeat[/] extended={bool(extended)}")

    hb = asyncio.create_task(heartbeat())

    async def slow_work():
        for s in range(6):
            console.print(f"  working {s + 1}/6...")
            await asyncio.sleep(1)
        return "done"

    try:
        result = await asyncio.wait_for(slow_work(), timeout=10)
        console.print(f"work result: {result}")
    finally:
        hb.cancel()
        await r.eval(RELEASE_LUA, 1, f"lock:{key}", owner)
        await r.aclose()


# --- Pattern C · fencing tokens ---------------------------------------------
async def demo_fencing() -> None:
    section("C · Fencing tokens — the storage layer enforces ordering")

    r = await _redis()

    # Reset.
    await r.delete("lock:idx:counter", "lock:idx", "idx:last_token")

    async def acquire_with_token(key: str, ttl: int = 5) -> int | None:
        token = await r.incr(f"lock:{key}:counter")
        if await r.set(f"lock:{key}", token, nx=True, ex=ttl):
            return token
        return None

    async def write_idx(token: int, payload: str) -> bool:
        """Storage layer accepts only monotonic tokens."""
        check = await r.eval(
            "local cur = tonumber(redis.call('get', KEYS[1]) or '0')\n"
            "if tonumber(ARGV[1]) < cur then return 0 end\n"
            "redis.call('set', KEYS[1], ARGV[1])\n"
            "return 1\n",
            1, "idx:last_token", token,
        )
        if check:
            console.print(f"  ✅ accepted write token={token} payload={payload!r}")
        else:
            console.print(f"  ⛔ rejected write token={token} (stale)")
        return bool(check)

    # Replica A acquires, then "stalls" (we'll simulate by not releasing yet).
    token_a = await acquire_with_token("idx", ttl=1)
    console.print(f"  A acquired token {token_a}")

    # TTL expires (1s).
    await asyncio.sleep(1.2)

    # Replica B acquires (gets a higher token).
    token_b = await acquire_with_token("idx", ttl=5)
    console.print(f"  B acquired token {token_b}")

    # B writes successfully.
    await write_idx(token_b, "B-payload")

    # A wakes up, tries to write with its stale token.
    await write_idx(token_a, "A-payload")    # rejected

    await r.aclose()


# --- Pattern D · idempotency key (no lock) ----------------------------------
async def demo_idempotency() -> None:
    section("D · Idempotency keys — the lock-free alternative")

    # We pretend there's a DB.
    db: dict[str, str] = {}

    async def charge_card(amount: float, idempotency_key: str) -> str:
        if existing := db.get(idempotency_key):
            console.print(f"  ↺ idempotent replay: returning cached {existing!r}")
            return existing
        await asyncio.sleep(0.1)         # simulate API call
        receipt = f"receipt_{idempotency_key}_{amount}"
        db[idempotency_key] = receipt
        console.print(f"  💳 new charge ${amount} → {receipt!r}")
        return receipt

    key = "session-42-attempt-1"

    # Simulate two concurrent retries from two replicas.
    a, b = await asyncio.gather(
        charge_card(99.0, key),
        charge_card(99.0, key),
    )
    assert a == b, f"idempotency broken: {a} != {b}"
    console.print("[green]Both calls returned the same receipt → idempotent ✅[/]")


# --- entry point ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setnx", action="store_true")
    parser.add_argument("--heartbeat", action="store_true")
    parser.add_argument("--fencing", action="store_true")
    parser.add_argument("--idempotency", action="store_true")
    args = parser.parse_args()

    selected: list = []
    if args.setnx:         selected.append(demo_setnx)
    if args.heartbeat:     selected.append(demo_heartbeat)
    if args.fencing:       selected.append(demo_fencing)
    if args.idempotency:   selected.append(demo_idempotency)
    if not selected:
        # Idempotency works without Redis; put it first.
        selected = [demo_idempotency, demo_setnx, demo_heartbeat, demo_fencing]

    async def run_all():
        for fn in selected:
            try:
                await fn()
            except SystemExit:
                return

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
