# Lesson 29 · Vector databases — deep dive

The entire RAG curriculum so far has used **Chroma** because it works
with zero infrastructure: `pip install langchain-chroma`, point at a
folder, you have a vector index. That's the right default for learning
and for prototypes. The moment you ship to production, you trade
"works locally" for *concurrency, durability, metadata filtering,
horizontal scaling, hybrid search, and an ops story*. This lesson is
the **honest comparison and ops walkthrough** for the six options that
matter in 2026.

## What you'll learn

1. **The five things a vector DB has to do well** — index, filter, hybrid-search, persist, scale.
2. **Six contenders side-by-side** — Chroma, pgvector, Qdrant, Weaviate, Milvus, Pinecone, LanceDB. What each is great at, what it isn't.
3. **The pgvector path** — extending the Postgres you already have (lesson 12 / `rag_qa_api`) into a vector store.
4. **The Qdrant path** — purpose-built, filterable, hybrid search.
5. **Both, side-by-side** — the same RAG question answered by both stacks so the difference is concrete.
6. **Docker-compose** — one `compose.yml` that brings up Postgres + pgvector AND Qdrant, ready to plug into your LangGraph.
7. **Hybrid search** — vector + BM25 keyword, why it beats either alone.
8. **Production checklist** — what to monitor, how to migrate, when to switch.

## The five jobs a vector DB has to do

| Job | What it means | Where Chroma struggles |
|---|---|---|
| **Index** | ANN search (HNSW, IVF, etc.) over millions of vectors in <100ms | Chroma uses HNSW too, but doesn't tune well past ~1M vectors |
| **Filter** | "vector match WHERE tenant_id = 'acme' AND created_at > X" | Chroma's metadata filter is post-filter — slow at scale |
| **Hybrid** | Combine dense vectors with sparse BM25 / keyword | Chroma has no native sparse; you bolt it on |
| **Persist** | Survive restarts, support backup/restore, ACID-ish | Chroma persists to SQLite; no replication |
| **Scale** | Shard across nodes, replicate, observe | Single-process; no built-in clustering |

If your workload doesn't push any of these, **stay on Chroma**. The
moment one becomes painful, the comparison below tells you where to go.

## The 2026 vector DB landscape

| Database | License | Self-host? | Strengths | Weak spots | When to pick |
|---|---|---|---|---|---|
| **Chroma** | Apache 2 | Yes (or hosted) | Easiest dev exp; embedded mode; native LangChain | No clustering; metadata filter not pre-applied; weak hybrid | Prototypes, <1M vectors, single process |
| **pgvector** | PG (BSD-style) | Yes (just install the extension) | Boring tech — already-have-Postgres path; full SQL filtering; transactional with your business data | Slower than dedicated DBs at scale; HNSW added in 0.5+ | If you already run Postgres. Up to ~10M vectors easily. |
| **Qdrant** | Apache 2 | Yes (Rust binary, low resource); also Qdrant Cloud | Filter-while-search (true pre-filter); fast Rust core; sparse + dense hybrid; sharding | Smaller ecosystem than Weaviate; SQL-style filters not as flexible as pgvector | 10M-100M vectors with heavy filtering. Strong default. |
| **Weaviate** | BSD-3 | Yes; Weaviate Cloud | Modular embeddings (built-in vectorizer); GraphQL API; class-based schema | Heavier to operate; opinionated schema; Java | When you want the DB to embed for you, or use GraphQL |
| **Milvus** | Apache 2 | Yes (heavy: etcd + pulsar/kafka + storage); Zilliz Cloud | Massive scale (billion-vector); GPU indexing; many index types | Operational complexity; overkill below 100M | Billion-scale or GPU-accelerated indexing |
| **Pinecone** | Proprietary | No (managed only) | Zero ops; great hybrid; serverless tier; very fast | Vendor lock-in; cost at scale; data sovereignty | Want "credit card, done." Heavy reliance on SaaS. |
| **LanceDB** | Apache 2 | Yes (embedded) | Embedded in your process; columnar (Arrow); great for ML notebooks | Newer; smaller ecosystem; serverless story still maturing | Notebooks, ML pipelines, "just one binary" |

**Two-sentence verdict for picking a default:**

- If you already have Postgres → **pgvector**. Boring + correct.
- If you don't and you want self-host with great filtering → **Qdrant**.

Pinecone wins if "no ops, no infra" outranks cost. Weaviate wins if
GraphQL is your jam. Milvus wins at billion scale. Chroma wins in
notebooks. LanceDB wins when you want zero processes.

## Path A — pgvector (the boring choice)

Postgres + an extension. Same connection, transactional with the rest
of your data, indexed with HNSW or IVF-Flat.

### Docker

The lesson's `docker-compose.yml` uses the official `pgvector/pgvector:pg16`
image, which is `postgres:16-alpine` with the extension preinstalled.
First-time setup runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB,
    embedding   VECTOR(384)                            -- match your embedder's dim
);
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

384 because `BAAI/bge-small-en-v1.5` produces 384-dim vectors. Pick the
matching dim for your embedding model — `1536` for OpenAI's
`text-embedding-3-small`, `3072` for `text-embedding-3-large`, etc.

### LangChain wiring

```python
from langchain_postgres.vectorstores import PGVector

store = PGVector(
    connection="postgresql+psycopg://postgres:postgres@localhost:5432/langgraph",
    embeddings=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    collection_name="docs",
    use_jsonb=True,
)
store.add_documents(documents)
retriever = store.as_retriever(search_kwargs={"k": 4, "filter": {"tenant_id": "acme"}})
```

Filter operates as a *pre-filter* (Postgres WHERE before the HNSW
scan), which is the right shape for tenant-isolated SaaS.

## Path B — Qdrant (the "purpose-built" choice)

A Rust-native vector DB with a Python client. Filterable, fast,
clustering-capable, and runs from a single Docker image.

### Docker

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"        # REST + dashboard
    - "6334:6334"        # gRPC (faster)
  volumes:
    - qdrant_data:/qdrant/storage
```

The dashboard at <http://localhost:6333/dashboard> visualises
collections, points, and queries.

### LangChain wiring

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

store = QdrantVectorStore.from_documents(
    documents,
    embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    url="http://localhost:6333",
    collection_name="docs",
)

retriever = store.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"must": [{"key": "metadata.tenant_id", "match": {"value": "acme"}}]},
    },
)
```

Note the filter syntax differs from pgvector — Qdrant uses its own
nested JSON filter DSL, but it's expressive (must/must_not/should,
range, geo, full-text).

### Hybrid search (Qdrant 1.10+)

Qdrant supports **named vectors** so you can store dense + sparse
side-by-side. With LangChain:

```python
from langchain_qdrant import FastEmbedSparse, RetrievalMode

store = QdrantVectorStore.from_documents(
    documents,
    embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
    sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
    url="http://localhost:6333",
    collection_name="docs_hybrid",
    retrieval_mode=RetrievalMode.HYBRID,
)
```

The retriever now fuses dense ANN + BM25 — measurable accuracy lift
on most RAG benchmarks (typically +5-15% recall@10 over dense-only).

## The same RAG, both stacks, side-by-side

The lesson's `example.py` indexes the **same** corpus
(`data/sample_docs/`) into **both** pgvector and Qdrant, runs the same
five questions, and prints the answers + latency side-by-side. The
takeaway:

- For small corpora (~50 chunks), both feel instantaneous.
- pgvector's `filter=` returns first-class because it's just SQL.
- Qdrant's hybrid mode beats either dense-only retrieval on the keyword-heavy queries ("PTO days", "refund").

## Docker Compose

The `docker-compose.yml` in this lesson brings up **both** databases
alongside the existing rag_qa_api pattern, so you can run the whole
stack with one command:

```bash
cd lessons/29_vector_databases
docker compose up -d
```

Three services:

| Service | Image | Port | URL |
|---|---|---|---|
| `pgvector` | `pgvector/pgvector:pg16` | 5432 | `postgresql://postgres:postgres@localhost:5432/langgraph` |
| `qdrant` | `qdrant/qdrant:latest` | 6333 / 6334 | <http://localhost:6333/dashboard> |
| `redis` | `redis:7-alpine` | 6379 | `redis://localhost:6379` (used by lesson 31's distributed lock + rag_qa_api_pro) |

`redis` shows up here because lesson 31 and the pro capstones need it
— bundling the three together saves you from running three composes.

## Connecting back to LangGraph

The `retriever` is just a Runnable. Drop it into any existing graph:

```python
from langgraph.graph import StateGraph, START, END, MessagesState

def retrieve_node(state):
    docs = retriever.invoke(state["messages"][-1].content)
    return {"context": "\n\n".join(d.page_content for d in docs)}

def generate_node(state):
    prompt = f"Answer using:\n{state['context']}\n\nQ: {state['messages'][-1].content}"
    return {"messages": [get_llm().invoke(prompt)]}

g = StateGraph(MessagesState)
g.add_node("retrieve", retrieve_node)
g.add_node("generate", generate_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
```

Swapping `retriever` from Chroma → pgvector → Qdrant is **a one-line
change**. That's the value of treating the retriever as an interface,
not as an implementation.

## Metadata filtering — the most underrated feature

Three places people get burned:

1. **Tenant isolation.** Multi-tenant SaaS *must* filter by tenant_id, and the filter has to be pre-applied (rows excluded before ANN). pgvector does this natively. Qdrant does this natively. Chroma can do it but it's post-filter (slow on large indexes).

2. **Document freshness.** "Only search docs updated in the last 30 days" — needs a date range filter. Both pgvector and Qdrant support this; check the filter syntax for your client.

3. **ACL / permissions.** "Only return docs the user has read access to." The cleanest path is a `permissions: ["user:42", "group:acme"]` array on each chunk; filter with `permissions HAS ANY ($user_perms)`.

If your DB doesn't pre-filter, every retrieval at scale becomes "scan
the whole index, then throw 99% of results away." Performance dies
silently — you don't notice until P99 latency climbs into the seconds.

## Hybrid search — when dense alone falls short

Embedding similarity finds *semantic* matches but misses *exact-string*
matches. Queries that hurt dense-only:

- Acronyms / product names: `K8s`, `Stripe`, `ABC-1234`
- Numbers: `200ms`, `$250 refund`, `lesson 29`
- Code identifiers: `RecursiveCharacterTextSplitter`

BM25 (sparse) is dumb but finds these. Hybrid fuses both with **RRF
(Reciprocal Rank Fusion)** or a learned reweighter, getting the recall
of BM25 with the semantic generalisation of vectors.

In production, **always evaluate hybrid before claiming "we tried RAG
and it didn't work."** The fix is usually "switch to hybrid" before
"switch model."

## Migration playbook (Chroma → pgvector / Qdrant)

Six-step process that has zero downtime:

1. **Dual-write.** Start writing every new chunk to both stores. Reads still hit Chroma.
2. **Backfill.** Re-embed your existing corpus into the new store. Idempotent — re-runs are safe.
3. **Dual-read.** Read from both, compare top-5 overlap. Log discrepancies.
4. **Promote read.** Switch reads to the new store. Keep dual-write.
5. **Stop writing to Chroma.** Now single-source.
6. **Delete Chroma.** Reclaim disk.

This is the same shape as a relational-DB migration. The killer is step
3 — without it, you'll discover *after* cutover that the new store
returns slightly different top-k and your eval scores dropped 8%.

## Production checklist

| Concern | What to monitor / set |
|---|---|
| **Recall** | Pin an eval suite (lesson 26 Topic 4); track recall@5 over time |
| **Latency** | P50 / P95 / P99 for retriever.invoke; alert on regressions |
| **Index size** | Vectors × dim × 4 bytes ≈ raw size. Plus ~50% for HNSW graph |
| **Reindex strategy** | When you change embedding models, you MUST reindex. Use the migration playbook above |
| **Backup** | pgvector → `pg_dump`. Qdrant → snapshot API (`POST /collections/{name}/snapshots`) |
| **Vector dim mismatch** | Lock the embedding model in code; assert dim at startup |
| **Tenant filter** | Always-applied at the retriever layer, NOT in application code |
| **Concurrent writes** | pgvector: transactional. Qdrant: eventually-consistent; tune with `wait=true` |

## Run it

```bash
# bring up both databases
cd lessons/29_vector_databases
docker compose up -d

# wait ~10s for services to be ready, then run the comparison
uv run python -m lessons.29_vector_databases.example
```

The script:
1. Indexes `data/sample_docs/*.md` into both pgvector and Qdrant.
2. Asks five questions of each.
3. Prints answers + latency side-by-side.
4. Optionally runs the hybrid-search demo (`--hybrid`).

## Try it yourself

1. Add a `tenant_id` to every chunk's metadata. Run two retrievals with the same query but different tenant filters — confirm they return disjoint sets.
2. Swap `BAAI/bge-small-en-v1.5` (384 dim) for `BAAI/bge-base-en-v1.5` (768 dim). You'll need to reindex. Measure recall@5 — does it improve?
3. Add a third stack: an in-process `Chroma` for comparison. Plot latency + recall for all three.
4. Wire one of the retrievers into [Lesson 32 · prompt engineering lab](../32_prompt_engineering_lab/README.md)'s eval suite and run a real A/B.

## Anti-patterns

| Smell | Fix |
|---|---|
| Picking Pinecone "because everyone uses it" | If you self-host Postgres, pgvector first. Pinecone is great but it's a vendor decision |
| Filtering in Python after retrieval | Push the filter into the DB. Otherwise you're scanning the whole index every call |
| Using dense-only on keyword-heavy queries | Add BM25; measure hybrid recall before defaulting to "the LLM should handle it" |
| Hardcoding the embedding dim | Read it from the embedder at startup: `model.embed_query("x").__len__()` |
| No reindex plan when changing embedder | Migration playbook above. Embedder change = full re-embed |
| One giant collection per workspace | Shard by tenant where possible; smaller indexes are faster |

## Pairs with

- **[Lesson 06 · RAG basics](../06_rag_basics/README.md)** — Chroma intro; this lesson is the "now what" answer
- **[Lesson 07 · RAG advanced](../07_rag_advanced/README.md)** — multi-query, parent-document, contextual compression all work the same regardless of vector DB
- **[Lesson 26 · Miscellaneous](../26_misc/README.md)** — Topic 7 reranking complements hybrid search
- **[Lesson 31 · Distributed locks](../31_distributed_locks/README.md)** — Redis from the same compose stack
- **[`projects/rag_qa_api_pro`](../../projects/rag_qa_api_pro/README.md)** — capstone where you wire one of these in

## References

- [pgvector](https://github.com/pgvector/pgvector) — extension + index types
- [`langchain-postgres` PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector/) — LangChain wrapper
- [Qdrant docs](https://qdrant.tech/documentation/) — concepts + API reference
- [`langchain-qdrant`](https://python.langchain.com/docs/integrations/vectorstores/qdrant/) — including hybrid setup
- [Weaviate](https://weaviate.io/developers/weaviate) — class schema, modules
- [Milvus](https://milvus.io/docs) — at-scale ops
- [Pinecone serverless](https://docs.pinecone.io/) — managed, hosted only
- [LanceDB](https://lancedb.github.io/lancedb/) — embedded columnar
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — pick your embedder; the DB choice is downstream of this
- [Reciprocal Rank Fusion paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the math behind hybrid

## Next →

[Lesson 30 · Advanced graphs](../30_advanced_graphs/README.md) — parallel fan-out / fan-in, map-reduce in LangGraph, dynamic subgraph spawning.
