# Lesson 27 · Locks & concurrency in LangGraph

> "If two agents try to charge the same card at the same time, you have
> a problem the LLM cannot solve."

LangGraph applications look single-threaded from the inside of one
graph run, but the *process* hosting them isn't. The moment you deploy
behind FastAPI, run a background worker, or open a second chat thread
on the same user, you are concurrent. This lesson covers the **four
locking patterns** you need to know, why each exists, and how to wire
them into a `StateGraph` without making everything serial.

## What you'll learn

| # | Pattern | Where it lives | What it stops |
|---|---|---|---|
| 1 | **`asyncio.Lock`** — in-process mutex | Single Python process | Two coroutines mutating the same dict |
| 2 | **`asyncio.Semaphore`** — bounded concurrency | Single Python process | 50 parallel branches blasting one rate-limited API |
| 3 | **Per-key lock map** — `dict[key, asyncio.Lock]` | Single process | Two requests for the *same* user racing; different users still parallel |
| 4 | **Distributed lock** — Redis `SETNX` / Postgres advisory | Across processes / machines | Two API replicas both rebuilding the vector index at startup |

Lesson 31 (Tier 6) takes pattern 4 deeper — sharded locks, fencing
tokens, the lease-expiry problem. This lesson is the **practical
ground floor**: when do you reach for which lock, and how do you
plumb it through a graph.

## Why concurrency hurts agents specifically

Three failure modes you will hit if you ignore locking:

1. **Tool double-fire.** The agent retries a tool call (network blip,
   timeout, or the model decides to call it twice "to be sure"). If the
   tool is `charge_card`, `send_email`, or `create_jira_ticket`, you
   just did the thing twice. The fix isn't "tell the model not to" —
   it's an **idempotency key** + a lock.

2. **State stomp.** Two messages from the same user arrive seconds
   apart. Both load the checkpoint, both append a message, both write
   back. The second write *clobbers* the first message because the
   checkpoint version it loaded was stale. LangGraph's checkpointers
   detect this with version numbers, but **you have to handle the
   error** — silently retrying without a lock just produces the same
   race.

3. **Rate-limit storms.** A fan-out node spawns 30 parallel research
   branches. Each calls Anthropic. The provider returns `429 Too Many
   Requests` for 28 of them. Without a `Semaphore`, your "go faster"
   change made everything slower because retries pile up.

## The four patterns, fastest to most powerful

### Pattern 1 — `asyncio.Lock` (mutex)

```python
import asyncio

_index_lock = asyncio.Lock()

async def rebuild_index_node(state):
    async with _index_lock:                      # waits if another task is inside
        await chroma.reset()
        await chroma.add_documents(state["docs"])
    return {"index_status": "rebuilt"}
```

Use when: **at most one** coroutine may be inside the critical section
at a time, in a single process. Cheap, no external dependency. Does
NOT help across replicas — for that, jump to pattern 4.

### Pattern 2 — `asyncio.Semaphore` (bounded concurrency)

```python
_anthropic_sem = asyncio.Semaphore(8)            # 8 concurrent calls max

async def research_branch(query: str):
    async with _anthropic_sem:
        return await llm.ainvoke(query)
```

Use when: **N** concurrent operations are fine but **N + 1 is not**.
The classic case is rate-limited APIs (Anthropic: 50 RPS on Tier 4,
Tavily: 100/min on the free tier). Pair with `asyncio.gather` to
fan-out 30 tasks but only have 8 in flight at once.

### Pattern 3 — Per-key lock map

```python
from collections import defaultdict
_user_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

async def handle_message(user_id: str, msg: str):
    async with _user_locks[user_id]:             # serialises THIS user
        ...                                       # other users still parallel
```

Use when: you want **serial-per-user, parallel-across-users**. This
is the right shape for chat servers: two messages from user `alice` must
not interleave, but `alice` and `bob` can run concurrently. The map
itself is a `defaultdict`, so locks are created on demand.

**Memory pitfall:** the lock map grows forever. In production, wrap it
in a `TTLCache` (`cachetools`) or sweep it when the lock has zero
waiters and zero owners. The lesson's `example.py` shows the sweep.

### Pattern 4 — Distributed lock (Redis)

```python
# pip install redis>=5
import redis.asyncio as redis

r = redis.from_url("redis://localhost:6379")

async def rebuild_index_distributed():
    # SET key value NX (only if not exists) EX 60 (expire in 60s)
    acquired = await r.set("lock:index_rebuild", "owner-1", nx=True, ex=60)
    if not acquired:
        return  # another replica is doing it
    try:
        await chroma.reset()
        await chroma.add_documents(...)
    finally:
        await r.delete("lock:index_rebuild")
```

Use when: **multiple processes** must coordinate (API replicas, worker
queue, scheduled job + on-demand handler). The Redis line is one of
two production-grade options:

| Option | Strength | Weakness |
|---|---|---|
| Redis `SETNX` + TTL | Fast, simple, well-understood | "Liveness lock" — if the owner crashes mid-work, the TTL expires and someone else grabs it. Fencing tokens (lesson 31) cover the edge cases. |
| Postgres `pg_advisory_lock` | No new infra if you already have Postgres | Slightly slower; lock dies if the connection dies |

Both are taught in lesson 31. For 95% of cases, **Redis is the answer
you'll reach for**.

## Locks inside a LangGraph node

LangGraph runs nodes sequentially by default — if your graph is `A →
B → C`, you don't need locks. But you DO need them when:

1. **Fan-out edges** (lesson 30) — the graph adds multiple edges from
   one node into a `Send(...)` list, and those branches share state.
2. **The same graph runs in multiple threads** (different `thread_id`s,
   same process) — the checkpointer keeps each thread's state separate,
   but any **module-global** state (a singleton retriever, an in-memory
   cache) is shared and needs protection.
3. **External side-effects** (`charge_card`, `send_email`) — these
   need locks PLUS idempotency keys, not one or the other.

The pattern: keep locks at **module level** (not in graph state — locks
are not serialisable!) and `async with` them inside the node body. The
demo in `example.py` shows three nodes — one without a lock (races),
one with `asyncio.Lock`, one with a per-key map — and prints the
output so you can see the bug.

## Idempotency keys vs locks

**Locks prevent concurrent execution. Idempotency keys make repeated
execution safe.** You almost always want both:

```python
@tool
def charge_card(amount: float, idempotency_key: str) -> str:
    """Charge the customer card. Same idempotency_key returns the same result."""
    async with _payment_locks[idempotency_key]:
        if existing := await db.charge_lookup(idempotency_key):
            return existing.result                    # same key, same answer
        result = await stripe.charge(amount, idempotency_key=idempotency_key)
        await db.charge_save(idempotency_key, result)
        return result.id
```

The agent generates the idempotency key once (e.g., from the user's
session id + a counter) and reuses it on retries. The lock makes the
"check then insert" atomic. Stripe and most modern payment APIs have a
native `Idempotency-Key` header — use it.

## Run it

```bash
uv run python -m lessons.27_locks_and_concurrency.example
```

The script runs four scenarios:

1. `--race` — two coroutines mutating a shared counter without a lock (you'll see lost updates)
2. `--mutex` — same code, fixed with `asyncio.Lock`
3. `--semaphore` — 30 calls fanned out with a 5-wide semaphore
4. `--per-user` — two users sending messages in parallel, each user serialised

No API key needed — the demos use a fake LLM with a `random.uniform`
sleep so you can see the races.

## Debug it

Set a breakpoint inside the racy node:

```python
async def racy_node(state):
    breakpoint()                  # ipdb drops in
    val = COUNTER["n"]
    await asyncio.sleep(0.01)
    COUNTER["n"] = val + 1
```

Then run with `PYTHONBREAKPOINT=ipdb.set_trace`. Step through with two
coroutines and watch them both read the same stale `val`.

## Try it yourself

1. Add a **timeout** to the per-user lock with `asyncio.wait_for(lock.acquire(), timeout=5)` — what should the graph do when it can't grab the lock in time?
2. Replace the in-process `asyncio.Lock` with a Redis lock (lesson 31 has the recipe) and run two copies of the script — they should now mutually exclude.
3. Add a **dead-letter queue**: if the lock can't be acquired, push the message onto a list and process later. This is the shape Celery / Arq use.

## Anti-patterns

| Smell | Fix |
|---|---|
| `threading.Lock()` in async code | Use `asyncio.Lock` — `threading.Lock` blocks the event loop |
| Lock around the LLM call | You just serialised your whole agent. Lock only the *shared mutation*, not the slow call |
| Storing a lock in graph state | Locks aren't picklable. Module-level only; lookup by key inside the node |
| One global lock for everything | Per-resource locks. Otherwise concurrency = 1 |
| Distributed lock with no TTL | If the owner crashes, the lock is permanent. Always set an expiry |
| Catching `asyncio.CancelledError` and ignoring it | Propagate it — cancellation is how `asyncio.wait_for` enforces the timeout |

## Pairs with

- **[Lesson 28 · Dynamic prompting](../28_dynamic_prompting/README.md)** — when prompt templates have shared rendering state, the same locking principles apply to the renderer cache.
- **[Lesson 30 · Advanced graphs](../30_advanced_graphs/README.md)** — fan-out / fan-in is where pattern 2 (semaphore) earns its keep.
- **[Lesson 31 · Distributed locks](../31_distributed_locks/README.md)** — fencing tokens, Redlock debate, Postgres advisory locks.
- **[Lesson 25 · Tool design](../25_tool_design/README.md)** — idempotency keys belong in the tool's signature.

## References

- [Python asyncio Locks](https://docs.python.org/3/library/asyncio-sync.html) — the stdlib primitives.
- [Stripe idempotency keys](https://docs.stripe.com/api/idempotent_requests) — the canonical example.
- [Redis distributed locks (Redlock)](https://redis.io/docs/latest/develop/use/patterns/distributed-locks/) — official doc, including caveats.
- [Martin Kleppmann · How to do distributed locking](https://martin.kleppmann.com/2016/02/08/how-to-do-distributed-locking.html) — the famous critique of Redlock; required reading before you build anything mission-critical.
- [Postgres advisory locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS) — the "free if you already have Postgres" option.

## Next →

[Lesson 28 · Dynamic prompting](../28_dynamic_prompting/README.md) — making prompts a first-class, versioned, templated artefact instead of a Python f-string.
