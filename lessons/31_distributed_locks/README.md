# Lesson 31 · Distributed locks (Tier 6 deep dive)

Lesson 27 covered locks **inside one Python process**. The moment you
deploy two replicas of `rag_qa_api`, or you have a worker queue +
on-demand API both able to rebuild an index, or two LangGraph
applications share a Postgres checkpointer, you need a lock that lives
*outside* any single process.

This lesson goes deep on the three production options, the famous
**Redlock debate**, fencing tokens, and the lease-expiry problem.

## What you'll learn

1. **Why `asyncio.Lock` is not enough** once you scale past one process.
2. **Redis `SETNX` + TTL** — the workhorse pattern, with the bug it has.
3. **Fencing tokens** — the Kleppmann fix for the bug.
4. **Postgres advisory locks** — the "no new infra" alternative.
5. **Redlock** — the multi-Redis algorithm, and why most teams skip it.
6. **Idempotency keys** — the non-lock alternative for side-effect tools.
7. **The lease-expiry problem** — what happens when the lock TTL fires mid-work.

## Why `asyncio.Lock` is not enough

```
process A (replica 1):  async with _lock:  rebuild_index()
process B (replica 2):  async with _lock:  rebuild_index()
```

`_lock` is an `asyncio.Lock` in each process. **They are different
locks.** Two replicas both rebuild the index, double the load, possibly
corrupt the index if writes race. Same applies to any cross-process
coordination — cron job + on-demand handler, worker queue + API, etc.

You need a lock whose state lives in **shared storage**: Redis,
Postgres, ZooKeeper, etcd, Consul. For LangGraph apps, Redis is the
sweet spot.

## Pattern A · Redis SETNX + TTL (the workhorse)

```python
import asyncio
import redis.asyncio as redis
import uuid

r = redis.from_url("redis://localhost:6379")

async def acquire(key: str, owner: str, ttl_seconds: int = 30) -> bool:
    """True if we got the lock, False otherwise."""
    return await r.set(f"lock:{key}", owner, nx=True, ex=ttl_seconds)

async def release(key: str, owner: str) -> None:
    """Release ONLY if we still own it (lua to make it atomic)."""
    lua = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    await r.eval(lua, 1, f"lock:{key}", owner)


# Usage:
owner = str(uuid.uuid4())
if await acquire("index_rebuild", owner, ttl_seconds=60):
    try:
        await rebuild_index()
    finally:
        await release("index_rebuild", owner)
else:
    print("Someone else is rebuilding; skip")
```

**The critical detail:** release uses a Lua script that checks ownership
**before** deleting. Without it, this race occurs:

```
t=0    A acquires lock, ttl=60s
t=70   A is still working (slow), but lock has expired
t=71   B acquires lock (NX succeeds — A's lock is gone)
t=72   A finishes work, calls DEL lock:foo  ← DELETES B's LOCK
t=73   C acquires the lock B should still hold
```

The Lua script makes step `t=72` a no-op because `GET` returns B's
owner id, not A's.

## The lease-expiry problem (the bug Redlock tries to fix)

Even with the Lua-fix release, there's a deeper bug: **A's work continues
past the TTL**. While A is mid-write, B has the lock and is also
writing. They corrupt each other.

Three mitigations, listed from "good enough" to "bulletproof":

### Mitigation 1 — Make the work fast and the TTL generous

If `rebuild_index` reliably takes < 30s, a 5-minute TTL means even a 10x
slowdown is safe. Pair with timeouts on individual steps. This solves
the problem for 90% of real-world cases.

### Mitigation 2 — Lock renewal (heartbeat)

Spawn a background coroutine that re-extends the TTL every N seconds:

```python
async def with_heartbeat_lock(key, owner, ttl, work):
    if not await acquire(key, owner, ttl):
        raise LockBusy()

    async def heartbeat():
        while True:
            await asyncio.sleep(ttl / 3)
            await r.expire(f"lock:{key}", ttl)        # extend

    hb = asyncio.create_task(heartbeat())
    try:
        return await work()
    finally:
        hb.cancel()
        await release(key, owner)
```

If `work()` hangs forever, the heartbeat keeps the lock alive forever.
Combine with a **wall-clock timeout** on `work()` to cap the worst case.

### Mitigation 3 — Fencing tokens

The bulletproof fix (per Kleppmann). The lock acquisition returns a
**monotonically-increasing token**. Every downstream write includes the
token. The storage layer rejects writes with stale tokens.

```python
async def acquire_with_token(key):
    token = await r.incr(f"lock:{key}:counter")    # monotonic
    if not await r.set(f"lock:{key}", token, nx=True, ex=60):
        return None
    return token

async def write_index(token, data):
    # The storage layer checks: only accept writes if token >= current_token.
    if not await r.eval(
        "if tonumber(ARGV[1]) < tonumber(redis.call('get', KEYS[1])) then return 0 else return 1 end",
        1, f"lock:{key}:counter", token,
    ):
        raise StaleTokenError()
    await actual_write(data)
```

Now if A's lock expires and B acquires (token = 7 > A's 6), A's write
attempt with token 6 is rejected. The storage layer is the source of
truth, not the lock.

Fencing tokens are mandatory for *consistency*-critical workloads
(billing, inventory, idempotent tool calls). They are *overkill* for
"don't rebuild the index twice." Match the technique to the stakes.

## Pattern B · Postgres advisory locks

If you already run Postgres (e.g., for LangGraph's `PostgresSaver`),
you don't need Redis. Postgres has **advisory locks** — application-defined
locks that live for the duration of a connection or transaction.

```python
import psycopg

async def with_advisory_lock(conn, lock_key: int):
    """Session-scoped advisory lock. Released on connection close."""
    await conn.execute("SELECT pg_advisory_lock(%s)", [lock_key])
    try:
        yield
    finally:
        await conn.execute("SELECT pg_advisory_unlock(%s)", [lock_key])
```

Use `pg_try_advisory_lock` for non-blocking acquire. Use the
transaction-scoped variant (`pg_advisory_xact_lock`) inside a
transaction — it auto-releases on commit/rollback.

**Advantages:**

- Zero new infra if you already have Postgres
- Lock dies with the connection — no leaked locks from crashed processes
- Pairs naturally with your business-data transactions

**Disadvantages:**

- Slower than Redis (~5x for high-contention scenarios)
- Lock key is a single bigint — hash your string key first
- Hash collisions are theoretically possible (cheap to mitigate with 2-key locks)

## Pattern C · Redlock — and the famous debate

[Redlock](https://redis.io/docs/latest/develop/use/patterns/distributed-locks/)
is Redis's proposed *multi-instance* algorithm: acquire the lock on
N/2 + 1 of N Redis instances. Designed for HA at the cost of
complexity.

**Martin Kleppmann's critique** ([blog post](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html))
is required reading. Summary: Redlock is *not* safe against GC pauses,
network partitions, or clock drift. For *liveness* (we don't want
deadlock), it's fine. For *safety* (no two clients ever hold the lock
at once), it's not — you need fencing tokens regardless.

**Practical guidance:** unless you have a specific reason to use
Redlock (multi-region, multi-Redis, can't tolerate a Redis outage),
use single-Redis SETNX + fencing tokens. It's simpler, faster, and the
safety guarantees are the same.

## Pattern D · Idempotency keys (the non-lock answer)

For tool calls with side effects (`charge_card`, `send_email`,
`create_ticket`), **don't use a lock at all** — use an idempotency
key:

```python
@tool
async def charge_card(amount: float, idempotency_key: str) -> str:
    """Idempotent: same key returns same result, never charges twice."""
    if existing := await db.charge_lookup(idempotency_key):
        return existing.result

    # Stripe accepts an idempotency key natively — they handle the dedup
    result = await stripe.Charge.create(
        amount=amount,
        idempotency_key=idempotency_key,
    )
    await db.charge_save(idempotency_key, result)
    return result.id
```

If two replicas both try to charge, Stripe sees the same idempotency
key and returns the same result both times. No lock needed. The
service you call provides the guarantee.

**For tools you build yourself**, the pattern is: store
`(idempotency_key, result)` in your DB, check first, write last,
make the check-then-insert atomic with a UNIQUE constraint or a row
lock. Locks become a fallback for when the underlying service doesn't
support idempotency natively.

## Decision matrix

| Scenario | Use |
|---|---|
| One process, one event loop | `asyncio.Lock` (lesson 27) |
| Multiple processes, you have Redis | Redis SETNX + Lua release |
| Multiple processes, you have Postgres but no Redis | `pg_advisory_lock` |
| Side-effect tool (`charge_card`, `send_email`) | Idempotency key (no lock) |
| Multi-region, can't tolerate Redis outage | Redlock OR upgrade to ZooKeeper/etcd |
| Consistency-critical writes (billing, inventory) | Fencing tokens on top of any of the above |

## Run it

```bash
# Bring up Redis (from lesson 29's compose stack):
docker compose -f ../29_vector_databases/docker-compose.yml up -d redis

uv run python -m lessons.31_distributed_locks.example
uv run python -m lessons.31_distributed_locks.example --setnx
uv run python -m lessons.31_distributed_locks.example --heartbeat
uv run python -m lessons.31_distributed_locks.example --fencing
uv run python -m lessons.31_distributed_locks.example --idempotency
```

Without Redis, only `--idempotency` works. With Redis up, all four
scenarios run.

## Anti-patterns

| Smell | Fix |
|---|---|
| `del lock` without ownership check | Lua + ownership; otherwise A's late release kills B's lock |
| No TTL on a Redis lock | If the owner crashes, the lock is permanent. Always `ex=` |
| TTL shorter than worst-case work duration | Lock expires mid-work → two owners. Use heartbeats or longer TTL |
| Same lock key for unrelated resources | Per-resource keys: `lock:user:{id}:msg` not `lock:everything` |
| Locks where idempotency would suffice | Side-effect tools: idempotency key first, lock as backup |
| Redlock for a single-Redis deployment | Single SETNX is simpler and equally safe |
| No fencing token in consistency-critical paths | Read Kleppmann; if your work corrupts state on dual-owner, you need fencing |

## Pairs with

- **[Lesson 27 · Locks](../27_locks_and_concurrency/README.md)** — the single-process baseline
- **[Lesson 25 · Tool design](../25_tool_design/README.md)** — idempotency keys belong in the tool signature
- **[Lesson 30 · Advanced graphs](../30_advanced_graphs/README.md)** — circuit breaker state could live in Redis for cross-replica sharing
- **[`projects/rag_qa_api_pro`](../../projects/rag_qa_api_pro/README.md)** — uses Redis to coordinate index-rebuild across replicas

## References

- [Redis distributed locks](https://redis.io/docs/latest/develop/use/patterns/distributed-locks/) — official; includes Redlock spec
- [Kleppmann · How to do distributed locking](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html) — the famous critique. Required reading
- [Postgres advisory locks](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS) — official docs
- [`redis-py` async](https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html) — Python async client
- [Stripe idempotency](https://docs.stripe.com/api/idempotent_requests) — canonical idempotency-key implementation
- [ZooKeeper recipes · Locks](https://zookeeper.apache.org/doc/r3.9.0/recipes.html#sc_recipes_Locks) — if you outgrow Redis

## Next →

[Lesson 32 · Prompt engineering lab](../32_prompt_engineering_lab/README.md) — A/B testing prompts, registry pattern, automatic eval-driven iteration.
