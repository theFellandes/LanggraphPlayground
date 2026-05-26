# Lesson 33 · Vector database internals — how they actually work

Lesson 29 compares vector databases as *products* — what they do, when to
pick which, how to wire them into LangChain. This lesson goes one layer
down: **what algorithms are inside each one**, why those algorithms exist,
and how the algorithmic choice shapes everything you observe at the API
layer (latency, recall, memory, filterability, build time).

By the end, you'll be able to read any vector-DB benchmark and predict
which knob the vendor is hiding behind the marketing — and pick
intelligently from FAISS, Annoy, Qdrant, Milvus, Weaviate, Pinecone,
pgvector, LanceDB, ScaNN, Vespa, Chroma.

## What you'll learn

1. **Why ANN exists at all** — the math of exact search and where it stops scaling.
2. **The six algorithms that matter** — Flat, LSH, IVF, IVF-PQ, HNSW, ScaNN/DiskANN — each in 200-400 words, with a diagram.
3. **The per-vendor map** — which DB uses which algorithm, what they tune, what they sacrifice.
4. **The benchmarks that lie** — ann-benchmarks vs your actual workload; the four metrics that matter.
5. **Wiring patterns in LangChain / LangGraph** — one minimal connector example per major vendor.
6. **Decision matrix** — given your data size, filter complexity, ops budget, and consistency needs, which to pick.

This lesson is **pedagogically heavy on the algorithms**. The runnable
demo (`example.py`) builds a 30k-vector index in FAISS with every major
index type and prints the latency × recall × memory tradeoff so you can
*see* the curves.

---

## Part 1 · The math problem

Vector search is "given query vector q ∈ ℝᵈ and a corpus of N vectors,
return the k closest by cosine / inner product / L2." Exact search is
trivially correct but does N×d work per query.

| Corpus size N | Dim d | Ops/query (cosine) | Real-world latency |
|---|---|---|---|
| 10k | 768 | 7.7M | ~1ms |
| 1M | 768 | 770M | ~80ms |
| 100M | 768 | 77B | ~8s |
| 1B | 1536 | 1.5T | ~3min |

That last row is why ANN (Approximate Nearest Neighbour) is a thing. ANN
trades 1-5% recall for **100-1000× speedup** by:

1. **Pre-filtering** which vectors to score (clustering, graphs).
2. **Compressing** vectors so each comparison is cheaper (quantization).
3. **Stopping early** when "good enough" has been found.

Every algorithm below picks a different combination of those three.

### The metrics that actually matter

| Metric | What it measures | Trap |
|---|---|---|
| **Recall@k** | What % of true top-k did we return? | If recall@10 ≥ 0.95, your downstream RAG will not notice the difference vs exact |
| **QPS** | Queries per second per host | Often reported single-threaded; check the cores used |
| **P99 latency** | The slow tail | Important; "average" hides the bad cases |
| **Build time** | How long to index N vectors | Brushed under the rug in benchmarks. Matters for re-embedding |
| **Memory** | RAM (or disk) per vector | 1B × 768 × 4 bytes = 3 TB raw. Quantization is the only way down |
| **Insert/update latency** | Time to add one vector | Some indexes only batch-build (Annoy); some support live insert (HNSW) |
| **Filter recall** | Recall when also filtering by metadata | Post-filter looks fine on small filters, dies on `tenant_id` |

---

## Part 2 · The six algorithms that matter

### 2.1 · Flat (brute force) — the gold standard

```
for v in corpus:        cosine(q, v)
keep top-k.
```

- **Recall**: 1.0 (exact)
- **Speed**: O(N·d) per query — slow at scale
- **Build**: O(0) — no index, just the matrix
- **Memory**: N·d·4 bytes (or N·d for fp16)

**Use for**: benchmarking ("am I within 1% of exact?"), small corpora
(<100k vectors), and as the inner search inside IVF (see below).

FAISS calls this `IndexFlatL2` or `IndexFlatIP`. Every DB has it.

### 2.2 · LSH (Locality-Sensitive Hashing) — the historical baseline

```
for each of L hash families:
    hash(q) → bucket
collect vectors in q's buckets across all families
score those exactly, take top-k
```

LSH hashes similar vectors into the same bucket *with high probability*.
Original LSH (Indyk-Motwani 1998) uses random hyperplane projections for
cosine; bit-sampling for Hamming; min-hash for Jaccard.

- **Recall**: tunable via #families L and bucket width (typically 0.7-0.9 at the production end)
- **Speed**: O(L·avg_bucket_size·d)
- **Build**: O(N·L·d)
- **Memory**: vectors + L hash tables

**Status in 2026**: largely superseded by HNSW for in-memory workloads.
Still relevant in specialised settings (one-bit Hamming, MinHash for
near-duplicate detection).

### 2.3 · IVF (Inverted File) — clustering-based ANN

The first algorithm to scale beyond a few million vectors.

```
BUILD:
  1. Run k-means on the corpus → n_clusters centroids
  2. Assign each vector to its nearest centroid
  3. Store an "inverted file": cluster_id → list of (vec, id)

QUERY:
  1. Find n_probe nearest centroids to q (one Flat search over centroids)
  2. Exact-search the union of those n_probe inverted lists
  3. Return top-k
```

ASCII picture:

```
                          ╔══════════════╗
       ┌──────┐    o      ║   centroid   ║          ┌──────┐
       │ vec  │  o   o    ║      C2      ║          │ vec  │
       └──────┘     o     ╚══════════════╝          └──────┘
                            ↑                    o
        ┌──────┐            │                   o   o
        │ vec  │            │   q ── searches    o
        └──────┘            │      C2 & C5 only
       ╔══════════════╗     │                    ╔══════════════╗
       ║   centroid   ║─────┘                    ║   centroid   ║
       ║      C1      ║                          ║      C5      ║
       ╚══════════════╝                          ╚══════════════╝
```

- **Recall**: depends on `n_probe / n_clusters`. n_probe=8/4096 ≈ 0.85 recall@10 typically.
- **Speed**: O(sqrt(N)·d) with `n_clusters ≈ sqrt(N)` — the canonical rule
- **Build**: O(iters·N·n_clusters·d) — k-means dominates
- **Memory**: same as Flat plus a centroid table

**The knob you tune**: `n_probe` (called `nprobe` in FAISS, `efSearch`-adjacent
in spirit). Higher = better recall + slower.

FAISS: `IndexIVFFlat`. Used internally by Milvus, the older pgvector
versions (pre-0.5).

### 2.4 · IVF-PQ (Product Quantization) — for billion-scale memory savings

PQ is **the** trick that made billion-vector indexes fit on a single
machine.

Idea: split each vector into `m` sub-vectors of dimension `d/m`. Run
k-means on each sub-space independently with 256 centroids. Replace each
sub-vector with the index of its nearest centroid (1 byte). A 1024-dim
fp32 vector goes from 4 KB → 32 bytes (128×). Recall drops a few points;
you usually claw it back with a Flat re-rank on the top-100.

```
ORIGINAL          d=128 floats × 4 bytes  = 512 bytes/vector
PQ (m=8, b=8)     8 codes  × 1  byte      = 8 bytes/vector
```

Combined with IVF: cluster + quantize. This is the standard for >100M
vectors on a single host.

- **Recall**: 0.85-0.95 with re-ranking (Flat on top-100 of PQ-prefiltered candidates)
- **Memory**: 32-64x less than fp32
- **Speed**: similar to IVF; sub-vector lookups are table-driven
- **Build**: O(iters·m·N·256·d/m) — more expensive than IVF

FAISS: `IndexIVFPQ`. Used by Milvus, Vespa, LanceDB, Vald.

**OPQ** (Optimized PQ) is PQ + a learned rotation matrix that aligns the
sub-spaces with the data's actual variance. Same memory, ~3 points
better recall. Always use it when available.

### 2.5 · HNSW (Hierarchical Navigable Small World) — the 2026 default

Graph-based ANN. The dominant in-memory algorithm.

```
BUILD:
  For each new vector v:
    1. Sample a level L ∈ {0, 1, 2, ...} from a geometric distribution
    2. From the entry point, greedy-search down through levels
       inserting bidirectional edges to v's M nearest neighbours at each level
  Result: a multilayer graph; upper layers are sparse "highways," bottom layer has everything

QUERY q:
  Start at entry point on the top layer
  At each layer: greedy walk, keep a candidate heap of size ef
  Descend to the next layer with the best-so-far
  At layer 0, return the top-k from the candidate heap
```

ASCII picture:

```
  Layer 2:    ●─────────────●─────────────●           sparse "highways"
              ↓             ↓             ↓
  Layer 1:    ●──●────●─────●────●────●──●──●         intermediate
              ↓  ↓    ↓     ↓    ↓    ↓  ↓  ↓
  Layer 0:    ● ●●●●●●●●●●●●●●●●●●●●●●●●●●●● ●●        every vector
```

- **Recall**: 0.95+ at `ef_search = 50` typically. The closest thing to exact at scale.
- **Speed**: O(log N · M · d) — *logarithmic* in corpus size. The big win.
- **Build**: O(N · log N · M · d) — slow but parallelisable
- **Memory**: vectors + the graph (`M ≈ 16-48` edges per node, ~96-300 bytes overhead)

**The knobs you tune**:
- `M` — max edges per node. Higher = better recall + bigger index. 16-48 is normal.
- `ef_construction` — search width during build. Higher = better graph + slower build. 100-500 normal.
- `ef_search` — search width per query. **Tune this at query time, not build time.** 10-200 typical.

Used by: Qdrant (custom Rust), Weaviate (custom Go), pgvector ≥ 0.5,
Chroma (via hnswlib), LanceDB (newer versions), Vespa, FAISS
(`IndexHNSWFlat`).

**Why HNSW won**: per-query logarithmic complexity + incremental
insert/delete + no batch rebuild + tunable per-query without
re-indexing. The first algorithm that's good at *all* of those at once.

### 2.6 · ScaNN, DiskANN, SPANN — the research frontier

**ScaNN** (Google, 2020) — `Anisotropic Vector Quantization`. PQ but
the quantization minimises *inner-product* error on the *near* neighbours
specifically, not L2 error globally. ~3-5 points better recall than OPQ
at the same memory. Used in Vertex AI Vector Search.

**DiskANN** (Microsoft, 2019) — HNSW-like graph stored on SSD. Holds
billion-vector indexes with a few GB of RAM (only the graph entry points
+ a Bloom filter live in memory). Each step does one SSD read. ~10ms
latency on commodity NVMe.

**SPANN** (Microsoft, 2021) — Hybrid: in-memory inverted file points to
posting lists on disk. Each list is sequential reads. Great fit for
streaming corpus where most queries hit a small "hot" subset.

**NSG**, **NHQ**, **HNSW+PQ** — variants that combine graphs with
quantization for the best of both. Milvus exposes most of them.

You don't usually pick these directly — you pick the DB that uses them
under the hood. But knowing they exist explains why "Pinecone is fast"
or "Milvus handles billion-scale" — it's the algorithm.

---

## Part 3 · The per-vendor algorithmic map

| DB | Default algorithm | What's special |
|---|---|---|
| **FAISS** | Library — supports Flat, IVF, IVF-PQ, IVF-OPQ, HNSW, NSG, all of them | The reference impl. Facebook AI's library. SIMD + GPU-accelerated. C++ core, Python bindings. Used inside many other systems |
| **Annoy** | Random projection forests | Spotify's library. Multiple trees of random hyperplane splits; query traverses all trees. Memory-mappable (mmap)-friendly. Build-once, no updates |
| **ScaNN** | Anisotropic VQ + brute-force re-rank | Google's library + Vertex's hosted variant. Best-in-class recall@same-memory |
| **hnswlib** | HNSW | The C++ HNSW implementation — fast, used by Chroma, LangChain's `FAISS.from_documents` when you swap in, and many others |
| **Chroma** | hnswlib (HNSW) under the hood | Sqlite-backed metadata. Embedded mode (file folder) or client/server. Filter is post-filter |
| **pgvector** | HNSW (>= 0.5) or IVFFlat (older) | Postgres extension. Pre-filter via WHERE on indexed columns. Transactional. Slower than dedicated stores, faster than you'd guess |
| **Qdrant** | Custom Rust HNSW + filter integration | The filter is fused into graph search — true pre-filter. Sparse + dense via named vectors. Mmap-friendly |
| **Weaviate** | Custom Go HNSW + Sled-style vector cache | Modular: built-in embedding modules, GraphQL API, classes (schemas). Heavier ops than Qdrant |
| **Milvus** | Pluggable — HNSW, IVF, IVF-PQ, DiskANN, ScaNN-like, GPU IVF-PQ | Multi-shard, multi-replica, etcd + pulsar/kafka under the hood. Massive scale. Heavy ops |
| **Vespa** | HNSW + lots of other ML (BM25, ONNX scoring) | Yahoo / Verizon Media open-sourced. The closest thing to "Elastic for vectors" |
| **Pinecone** | Proprietary; ~ HNSW + DiskANN-style serverless | Managed only. The "credit card and ship" option. Hybrid sparse+dense |
| **LanceDB** | IVF-PQ on disk via the Lance columnar format | Embedded ("S3-native"). Vectors stored alongside data in Arrow. No process needed |
| **Vald** | Custom NGT (Yahoo Japan) — graph-based | Less common in the West; included for completeness. K8s-native |
| **MarQo** | HNSW + multi-modal text+image | Bundled vectorizers; good for "I want a CLIP+text index without ops" |

### What FAISS gives you that nothing else does

FAISS is the **algorithm playground**. If you want to compare IVF vs
HNSW vs IVF-PQ vs OPQ on *your* data without spinning up four
databases, FAISS is the answer:

```python
import faiss
import numpy as np

xb = np.random.random((100_000, 128)).astype("float32")
xq = np.random.random((100, 128)).astype("float32")

# Three indexes — same data
flat = faiss.IndexFlatL2(128)
flat.add(xb)

ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(128), 128, n_clusters=256, m=8, n_bits=8)
ivfpq.train(xb); ivfpq.add(xb)
ivfpq.nprobe = 10

hnsw = faiss.IndexHNSWFlat(128, 32); hnsw.hnsw.efConstruction = 100; hnsw.add(xb)
hnsw.hnsw.efSearch = 50

for name, idx in [("flat", flat), ("ivfpq", ivfpq), ("hnsw", hnsw)]:
    D, I = idx.search(xq, 10)
    # compare recall vs flat, time the call
```

The lesson's `example.py` runs exactly this comparison and prints the
latency × recall × memory table — the *clearest* way to internalise
what each algorithm trades.

---

## Part 4 · Why benchmarks lie (and how to read them honestly)

[ann-benchmarks.com](https://ann-benchmarks.com) is the canonical
leaderboard. It's also a trap if you don't read it carefully:

1. **Dataset matters more than algorithm.** SIFT-1M ≠ GloVe ≠
   text-embedding-3. Recall curves on one dataset don't transfer.
2. **They measure single-thread QPS.** Your production traffic is
   concurrent — pick a DB whose architecture parallelises well (Qdrant,
   Milvus, FAISS-GPU) rather than one with a fast single-thread number.
3. **They don't measure filterability.** A DB with great recall@10 on
   un-filtered queries can be useless when you add `WHERE tenant_id =
   '...'` if the filter is post-applied.
4. **They don't measure build time.** Algorithms with slow build (HNSW
   at large `M`, OPQ training) hurt you when you re-embed a corpus.
5. **They don't measure ops cost.** Milvus has the best billion-scale
   numbers and the worst ops story.

**The benchmark you should run**: your own embeddings, your own filter
distribution, your own concurrency. 50-100 cases is enough for a signal
on recall; 1000 sequential calls is enough for a signal on QPS.

---

## Part 5 · LangChain / LangGraph wiring per vendor

Every vector store implements LangChain's `VectorStore` interface, so
swapping between them is **one line** at the import site. What changes
is *configuration* and *filter syntax*. Below: the minimal "make a
retriever" recipe for each.

### FAISS — in-process, no infra

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
store = FAISS.from_documents(docs, emb)
store.save_local("./faiss_index")        # serialises to .faiss + .pkl

# later:
store = FAISS.load_local("./faiss_index", emb, allow_dangerous_deserialization=True)
retriever = store.as_retriever(search_kwargs={"k": 4})
```

Best for: notebooks, single-process services, demos. Updates require a
rebuild — FAISS is fundamentally a batch index.

### Chroma — embedded or client/server

```python
from langchain_chroma import Chroma
store = Chroma.from_documents(docs, emb, persist_directory="./chroma_db")
retriever = store.as_retriever(search_kwargs={"k": 4, "filter": {"tenant_id": "acme"}})
```

Best for: prototypes that need persistence + filtering. Single-process.

### pgvector — Postgres extension

```python
from langchain_postgres.vectorstores import PGVector
store = PGVector.from_documents(
    docs, emb,
    connection="postgresql+psycopg://postgres:postgres@localhost:5432/langgraph",
    collection_name="docs",
    use_jsonb=True,
)
retriever = store.as_retriever(search_kwargs={"k": 4, "filter": {"tenant_id": "acme"}})
```

Best for: when you already run Postgres. Tenant filtering is *pre-filtered*
because Postgres applies the WHERE before HNSW.

### Qdrant — Rust-native, filter-fused HNSW

```python
from langchain_qdrant import QdrantVectorStore
store = QdrantVectorStore.from_documents(
    docs, emb, url="http://localhost:6333", collection_name="docs",
)
retriever = store.as_retriever(search_kwargs={
    "k": 4,
    "filter": {"must": [{"key": "metadata.tenant_id", "match": {"value": "acme"}}]},
})
```

Best for: heavy filtering at moderate-to-large scale. Qdrant's
filter-during-search means filtered queries don't lose recall.

### Weaviate — class-based schemas + modules

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate
client = weaviate.connect_to_local()
store = WeaviateVectorStore.from_documents(docs, emb, client=client, index_name="Docs")
```

Best for: when you want the DB to handle embedding too (built-in
vectorizer modules). GraphQL fans.

### Milvus — billion-scale, multi-replica

```python
from langchain_milvus import Milvus
store = Milvus.from_documents(
    docs, emb,
    connection_args={"uri": "http://localhost:19530"},
    collection_name="docs",
    index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
)
```

Best for: 100M+ vectors, multi-shard, GPU indexing. Heaviest ops.

### Pinecone — managed only

```python
from langchain_pinecone import PineconeVectorStore
store = PineconeVectorStore.from_documents(
    docs, emb, index_name="docs",  # PINECONE_API_KEY from env
)
```

Best for: "no infrastructure, please." Serverless tier is cheap up to a
point. Vendor-locked.

### LanceDB — embedded columnar

```python
from langchain_community.vectorstores import LanceDB
import lancedb
db = lancedb.connect("./lancedb")
store = LanceDB.from_documents(docs, emb, connection=db, table_name="docs")
```

Best for: notebook / batch pipelines. Files on disk, no process
required. Plays nicely with Arrow / Polars.

### ScaNN (via langchain_community)

```python
from langchain_community.vectorstores import ScaNN
store = ScaNN.from_documents(docs, emb)
```

Best for: when you want Google's anisotropic VQ in-process. Less
ecosystem support than FAISS.

### Vespa — search engine with vectors

```python
from langchain_community.vectorstores import VespaStore
# (configuration: an existing Vespa application package)
```

Best for: when you also want BM25 + ONNX model scoring + ranking pipelines.
Not a "vector DB" — a "search platform."

---

## Part 6 · Plugging the retriever into LangGraph

The whole point of swapping vector stores is that **none of your graph
code changes**. The retriever is a Runnable. Drop it in:

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from shared import get_llm

def make_retrieve_node(retriever):
    def node(state):
        docs = retriever.invoke(state["messages"][-1].content)
        return {"context": "\n\n".join(d.page_content for d in docs)}
    return node

def generate_node(state):
    msg = state["messages"][-1].content
    prompt = f"Use this context:\n{state['context']}\n\nQ: {msg}"
    return {"messages": [get_llm().invoke(prompt)]}

g = StateGraph(MessagesState)
g.add_node("retrieve", make_retrieve_node(your_retriever))
g.add_node("generate", generate_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "generate")
g.add_edge("generate", END)
```

To switch from FAISS to Qdrant, change the one line that builds
`your_retriever`. That's it.

For a more interesting use case — a multi-source RAG that fans out
across **three** different vector stores in parallel — see lesson 30
(pattern 1, fan-out) combined with this lesson's vendor choices. A
realistic shape:

```
                ┌── retriever_chroma  ── docs (chunks from /handbook/)
START → query ──┤
                ├── retriever_pgvector ── tickets (filtered by tenant)
                └── retriever_qdrant   ── support transcripts (hybrid)
                        ↓ merge
                    grader → generate → cite → END
```

The grader picks the best chunks across all three sources. This is the
shape production support-bots end up at: different stores for different
data shapes, queried in parallel.

---

## Part 7 · Decision matrix

Read down. Pick the first row whose constraints match yours.

| If you need... | ...then pick |
|---|---|
| Notebook prototype, <1M vectors, no infra | **FAISS** (in-process) |
| Embedded persistence + filters, <5M vectors | **Chroma** or **LanceDB** |
| You already run Postgres, want one DB | **pgvector** |
| Heavy metadata filtering, 1M-500M vectors | **Qdrant** |
| Multi-modal, built-in vectorizer modules | **Weaviate** |
| Billion-scale, willing to operate | **Milvus** |
| BM25 + vectors + custom ranking | **Vespa** |
| "Just give me a managed API" | **Pinecone** |
| In-process, columnar, batch ML | **LanceDB** |
| Maximum recall at fixed memory | **ScaNN** (in-process or Vertex) |
| Algorithm research / benchmarking | **FAISS** (every algorithm in one library) |

---

## Run it

```bash
uv sync --extra ml          # for faiss-cpu, numpy
uv run python -m lessons.33_vector_database_internals.example
```

The demo:

1. Generates 30k synthetic 128-dim vectors.
2. Builds **four FAISS indexes** on the same data: Flat (exact), IVF, IVF-PQ, HNSW.
3. Runs 100 queries against each.
4. Prints a side-by-side latency × recall@10 × memory table.

You'll see Flat is most accurate but slowest. IVF is fast but trades
recall. IVF-PQ is the most memory-efficient. HNSW dominates on
recall-per-millisecond.

Sample output sketch:

```
index       build_ms  query_us  recall@10  memory_MB
flat               1     820       1.00        15.3
ivf              480     115       0.94        15.4
ivfpq            520      90       0.87         0.5
hnsw            8200      20       0.97        21.8
```

Numbers will vary; the *shape* of the curves won't.

## Try it yourself

1. Re-run with 300k vectors instead of 30k. HNSW's logarithmic curve becomes obvious.
2. Add `IndexHNSWPQ` (HNSW graph + PQ quantization) and see the memory-recall sweet spot.
3. Compare cosine vs L2 (`IndexFlatIP` after L2-normalising) — for normalised vectors they're equivalent up to a constant, and IP is faster in SIMD.
4. Build a Qdrant index with the same vectors and the same `M`/`efConstruction`. Compare to FAISS HNSW. Difference is small; Qdrant's edge is filter integration.

## Anti-patterns

| Smell | Fix |
|---|---|
| Picking a DB by GitHub stars | Pick by algorithm + filter requirement |
| Tuning `ef_construction` at query time | That's a build-time knob. Query-time is `ef_search` |
| `nprobe=1` "for speed" | Recall craters. Tune `nprobe` to your recall target |
| Using IVF-PQ without OPQ | OPQ is free recall. Always enable |
| Comparing benchmarks across datasets | Run your own. SIFT-1M ≠ your embeddings |
| Filter applied after retrieval | Pre-filter (Qdrant, pgvector) at scale. Post-filter only works at <100k |
| Re-embedding without re-indexing | Vector dim changed? Full rebuild. Vectors changed? Live update only with HNSW |
| Storing 1B fp32 vectors uncompressed | 3 TB. Use PQ/IVF-PQ. The recall hit is small |

## Pairs with

- **[Lesson 29 · Vector databases](../29_vector_databases/README.md)** — the vendor-comparison + ops view. Read 29 first if you're picking; this lesson if you're tuning.
- **[Lesson 06 · RAG basics](../06_rag_basics/README.md)** — the original Chroma intro
- **[Lesson 07 · RAG advanced](../07_rag_advanced/README.md)** — multi-query, parent doc, compression
- **[Lesson 30 · Advanced graphs](../30_advanced_graphs/README.md)** — fan-out across multiple retrievers
- **[Lesson 26 · Misc](../26_misc/README.md)** — Topic 7 reranking, the natural companion to a "high-recall ANN + rerank" pipeline
- **[`ml_foundations/04_finetuning_encoders`](../../ml_foundations/04_finetuning_encoders/README.md)** — *what* gets indexed downstream of *how* it gets indexed

## References

### Algorithms

- [Malkov & Yashunin · Efficient and robust ANN search using HNSW](https://arxiv.org/abs/1603.09320) — the HNSW paper
- [Jegou et al. · Product Quantization for Nearest Neighbour Search](https://hal.inria.fr/inria-00514462v2/document) — original PQ paper
- [Ge et al. · Optimized Product Quantization (OPQ)](https://www.kaiminghe.com/publications/pami13opq.pdf)
- [Guo et al. · Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)](https://arxiv.org/abs/1908.10396)
- [Jayaram Subramanya et al. · DiskANN](https://suhasjs.github.io/files/diskann_neurips19.pdf)
- [Chen et al. · SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search](https://proceedings.neurips.cc/paper_files/paper/2021/file/299dc35e747eb77177d9cea10a802da2-Paper.pdf)
- [FAISS wiki: indexing primer](https://github.com/facebookresearch/faiss/wiki) — Facebook's own pedagogical reference

### Benchmarks

- [ann-benchmarks.com](https://ann-benchmarks.com) — canonical leaderboard (read with caution; see Part 4)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — for embeddings, not indexes
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench) — Zilliz's vendor benchmark suite (biased but useful)

### Vendor docs

- [FAISS](https://github.com/facebookresearch/faiss/wiki) · [Annoy](https://github.com/spotify/annoy)
- [Qdrant](https://qdrant.tech/documentation/) · [Weaviate](https://weaviate.io/developers/weaviate) · [Milvus](https://milvus.io/docs)
- [Pinecone](https://docs.pinecone.io/) · [pgvector](https://github.com/pgvector/pgvector) · [LanceDB](https://lancedb.github.io/lancedb/)
- [Vespa](https://docs.vespa.ai/) · [ScaNN](https://github.com/google-research/google-research/tree/master/scann)

## Next →

[Lesson 34 · LLM observability + tracing](../34_observability_tracing/README.md) — start of Tier 7 (production AI engineer).

Or jump to [`ml_foundations/03_transformer_architecture`](../../ml_foundations/03_transformer_architecture/README.md) for what's inside the model that *produces* the vectors you've just learned to index.
