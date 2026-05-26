# Capstone · `rag_qa_api_pro`

The Tier 6 version of `rag_qa_api`. Multi-replica-safe FastAPI service
with vector-store choice, distributed locking, Jinja prompts, and a
complex graph (rewrite → retrieve → grade → maybe-rerank → generate →
cite).

| Layer | What's added |
|---|---|
| Vector store | `pgvector` OR `qdrant` — picked by env var |
| Graph | Query-rewrite → retrieve → grade → rerank → generate → cite |
| Locks | Redis `SETNX` for index-rebuild coordination across replicas |
| Prompts | Jinja templates, versioned, tenant-aware |
| Reliability | Tenacity retries + circuit breaker around the vector DB |
| Streaming | Token + node-update SSE channels |

```
            HTTP request (multi-tenant)
                  │
                  ▼
              FastAPI ── auth check (header)
                  │
                  ▼
              graph.ainvoke
                  │
                  ▼
              rewrite         ← multi-query expansion (lesson 07)
                  │
                  ▼
              retrieve        ← pgvector OR qdrant, tenant-filtered
                  │
                  ▼
              grade           ← LLM judge: relevant chunks only
                  │
                  ▼
            (low recall? → fallback → enrich)
                  │
                  ▼
              generate        ← Jinja-rendered grounded prompt
                  │
                  ▼
              cite            ← appends Sources: footer
                  │
                  ▼
                 END
```

## What it teaches (concept → lesson)

| Element | Lesson |
|---|---|
| FastAPI + LangGraph | 14, 22 |
| Multi-query / parent doc / contextual compression | 07 |
| Postgres / pgvector | 12, 29 |
| Qdrant pre-filter | 29 |
| Distributed lock for index rebuild | 31 |
| Jinja prompts | 28 |
| Bounded retry + circuit breaker | 30 |
| Streaming | 14 |

## Prerequisites

```bash
uv sync --extra api

# Bring up Postgres + Qdrant + Redis from lesson 29:
docker compose -f ../../lessons/29_vector_databases/docker-compose.yml up -d
```

## Configuration

The service picks its backend from env vars (with safe defaults):

| Var | Default | Effect |
|---|---|---|
| `VECTOR_BACKEND` | `pgvector` | `pgvector` \| `qdrant` |
| `POSTGRES_URL` | `postgresql://postgres:postgres@localhost:5432/langgraph` | pgvector + checkpointer |
| `QDRANT_URL` | `http://localhost:6333` | qdrant only |
| `REDIS_URL` | `redis://localhost:6379` | distributed lock |
| `API_KEY` | `dev-key` | header `X-API-Key` for /chat and /stream |
| `PROMPT_VERSION` | `v1` | which `prompts/qa.v{N}.j2` to use |

## Run it

```bash
# Local with pgvector backend
uv run uvicorn projects.rag_qa_api_pro.app:app --reload

# In Docker (recommended — uses the lesson 29 compose stack)
cd projects/rag_qa_api_pro
docker compose up --build
```

Then:

```bash
curl -s -X POST localhost:8000/chat \
  -H "x-api-key: dev-key" \
  -H "content-type: application/json" \
  -d '{"thread_id": "t1", "message": "How many PTO days do I get?", "tenant_id": "acme"}'
```

## The distributed-lock story

Two API replicas both start. Both notice the vector index is empty.
Both want to index `data/sample_docs/`. Without a lock: double-write,
duplicate chunks, wasted embedding budget.

With the lock (`projects.rag_qa_api_pro.locks.acquire`):

```python
async def ensure_index(redis_client):
    if await is_indexed():
        return
    owner = str(uuid.uuid4())
    if not await acquire(redis_client, "lock:idx_rebuild", owner, ttl=120):
        # Another replica is indexing — wait until done.
        await wait_until_indexed()
        return
    try:
        await rebuild_index()
    finally:
        await release(redis_client, "lock:idx_rebuild", owner)
```

The release uses the Lua-script ownership check from lesson 31.

## The prompt story

`prompts/qa.v1.j2` is the current prompt. Drop in `qa.v2.j2`, change
`PROMPT_VERSION=v2`, restart — that's the deployment.

The grader, rewriter, and citer have their own templates so you can
A/B them independently.

## Try it yourself

1. **Tenant filtering.** Add a `tenant_id` field to every chunk's metadata at index time; have the retriever read `tenant_id` from the request and pass it as a filter (lesson 29 has the pgvector + Qdrant filter syntax).
2. **Hybrid search.** Switch the Qdrant retriever to `RetrievalMode.HYBRID` (lesson 29) — measure recall@5 before/after.
3. **Rerank.** Add a cross-encoder reranker between `grade` and `generate` (lesson 26 Topic 7).
4. **Fencing tokens.** Promote the index-rebuild lock to a fenced lock (lesson 31). Now even if the rebuilder hangs past its TTL, the storage layer rejects its late writes.

## Pairs with

- [Lesson 22](../../lessons/22_architecture/README.md), [Lesson 29](../../lessons/29_vector_databases/README.md), [Lesson 30](../../lessons/30_advanced_graphs/README.md), [Lesson 31](../../lessons/31_distributed_locks/README.md).
