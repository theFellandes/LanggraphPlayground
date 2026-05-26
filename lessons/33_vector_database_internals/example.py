"""Lesson 33 · Vector database internals — FAISS algorithm comparison.

Builds four FAISS indexes on the same synthetic data and reports
latency × recall@10 × memory per index type:

    flat   — exact brute force (gold standard)
    ivf    — IVF + Flat inverted lists
    ivfpq  — IVF + Product Quantization (memory savings)
    hnsw   — Hierarchical Navigable Small World (current default)

Then shows the LangChain wiring story: wrap FAISS as a LangChain
VectorStore, use it as a retriever, plug into a tiny LangGraph node.

Run:
    uv sync --extra ml                            # for faiss-cpu, numpy
    uv run python -m lessons.33_vector_database_internals.example
    uv run python -m lessons.33_vector_database_internals.example --langchain
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from shared.pretty import console, section


def _synth_corpus(n: int = 30_000, d: int = 128, seed: int = 42):
    """Generate random unit-normalised vectors so cosine ≈ inner product."""
    rng = np.random.default_rng(seed)
    xb = rng.standard_normal((n, d), dtype=np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    xq = rng.standard_normal((100, d), dtype=np.float32)
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    return xb, xq


def _ground_truth(xb, xq, k=10):
    """Brute-force top-k via cosine for ranking comparison."""
    sims = xq @ xb.T
    return np.argsort(-sims, axis=1)[:, :k]


def _index_memory_mb(idx) -> float:
    """Approximate via FAISS's deserialise-size."""
    try:
        import faiss
        buf = faiss.serialize_index(idx)
        return len(buf) / (1024 * 1024)
    except Exception:
        return float("nan")


def benchmark_algorithms() -> None:
    section("FAISS algorithm comparison · 30k × 128-d")

    try:
        import faiss
    except ImportError:
        console.print(
            "[yellow]Missing `faiss`. Install with:[/]\n"
            "  uv add faiss-cpu     (CPU-only, ~50MB)\n"
            "  # or for the ml extras:  uv sync --extra ml"
        )
        return

    n, d, k = 30_000, 128, 10
    xb, xq = _synth_corpus(n, d)
    truth = _ground_truth(xb, xq, k)

    def recall(I_pred, I_true) -> float:
        # For each query: |pred ∩ true| / k
        hits = 0
        for p, t in zip(I_pred, I_true):
            hits += len(set(p.tolist()) & set(t.tolist()))
        return hits / (len(I_pred) * k)

    results = []

    # --- Flat (exact L2; equivalent to cosine after normalisation) ---
    t0 = time.perf_counter()
    flat = faiss.IndexFlatIP(d)        # inner product on normalised = cosine
    flat.add(xb)
    build_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _, I = flat.search(xq, k)
    query_us = (time.perf_counter() - t0) * 1e6 / len(xq)

    results.append(("flat", build_ms, query_us, recall(I, truth), _index_memory_mb(flat)))

    # --- IVF (clustering) ---
    nlist = 256
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)

    t0 = time.perf_counter()
    ivf.train(xb)
    ivf.add(xb)
    build_ms = (time.perf_counter() - t0) * 1000

    ivf.nprobe = 16                    # tune for recall
    t0 = time.perf_counter()
    _, I = ivf.search(xq, k)
    query_us = (time.perf_counter() - t0) * 1e6 / len(xq)
    results.append(("ivf (nprobe=16)", build_ms, query_us, recall(I, truth), _index_memory_mb(ivf)))

    # --- IVF-PQ (clustering + compression) ---
    m_pq, bits = 8, 8
    quant = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quant, d, nlist, m_pq, bits, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivfpq.train(xb)
    ivfpq.add(xb)
    build_ms = (time.perf_counter() - t0) * 1000

    ivfpq.nprobe = 16
    t0 = time.perf_counter()
    _, I = ivfpq.search(xq, k)
    query_us = (time.perf_counter() - t0) * 1e6 / len(xq)
    results.append(("ivfpq (m=8, b=8)", build_ms, query_us, recall(I, truth), _index_memory_mb(ivfpq)))

    # --- HNSW ---
    M = 32
    hnsw = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 100
    t0 = time.perf_counter()
    hnsw.add(xb)
    build_ms = (time.perf_counter() - t0) * 1000

    hnsw.hnsw.efSearch = 50
    t0 = time.perf_counter()
    _, I = hnsw.search(xq, k)
    query_us = (time.perf_counter() - t0) * 1e6 / len(xq)
    results.append((f"hnsw (M={M}, efS=50)", build_ms, query_us, recall(I, truth), _index_memory_mb(hnsw)))

    # --- print -------------------------------------------------------------
    console.print("")
    console.print(
        f"{'index':28}  {'build_ms':>9}  {'query_us':>9}  {'recall@10':>10}  {'mem_MB':>7}"
    )
    console.print("-" * 70)
    for name, build_ms, query_us, r, mem in results:
        console.print(
            f"{name:28}  {build_ms:9.1f}  {query_us:9.1f}  {r:10.3f}  {mem:7.2f}"
        )
    console.print("")
    console.print(
        "[bold]Reading the table:[/]\n"
        "  flat is the gold standard (recall = 1.0) but pays full O(N·d) per query.\n"
        "  ivf trades a few recall points for ~10× speedup.\n"
        "  ivfpq is the same speed at ~30× less memory (note the mem_MB column).\n"
        "  hnsw wins on recall-per-microsecond — at the cost of a slower build.\n"
    )
    gc.collect()


# --- LangChain wiring demo --------------------------------------------------
def demo_langchain_wiring() -> None:
    section("FAISS as a LangChain VectorStore")

    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
    except ImportError:
        console.print(
            "[yellow]Missing langchain_community / fastembed. "
            "Try: uv sync (core deps already include these).[/]"
        )
        return

    docs = [
        Document(page_content="LangGraph models stateful workflows as graphs of nodes with shared state.",
                 metadata={"source": "langgraph"}),
        Document(page_content="LCEL composes Runnables with the pipe operator into stream/batch-aware chains.",
                 metadata={"source": "lcel"}),
        Document(page_content="Full-time employees get 20 PTO days per year.",
                 metadata={"source": "handbook"}),
        Document(page_content="Refunds are processed within 7 business days for orders under $500.",
                 metadata={"source": "handbook"}),
        Document(page_content="HNSW builds a navigable small-world graph layered hierarchically.",
                 metadata={"source": "vectors"}),
    ]
    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    store = FAISS.from_documents(docs, emb)
    retriever = store.as_retriever(search_kwargs={"k": 2})

    section("Use as a LangGraph retriever node")
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        question: str
        context: str
        answer: str

    def retrieve(state):
        out = retriever.invoke(state["question"])
        return {"context": "\n\n".join(d.page_content for d in out)}

    def answer(state):
        return {"answer": f"(Answer would use this context)\n---\n{state['context']}"}

    g = StateGraph(S)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

    graph = g.compile()
    for q in ("How many PTO days?", "What is HNSW?", "Difference between LCEL and LangGraph?"):
        console.rule(f"[bold]Q:[/] {q}")
        out = graph.invoke({"question": q, "context": "", "answer": ""})
        console.print(out["answer"])


# --- entry point ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langchain", action="store_true", help="run the LangChain wiring demo only")
    parser.add_argument("--bench", action="store_true", help="run the FAISS benchmark only")
    args = parser.parse_args()

    if args.langchain:
        demo_langchain_wiring()
    elif args.bench:
        benchmark_algorithms()
    else:
        benchmark_algorithms()
        demo_langchain_wiring()


if __name__ == "__main__":
    main()
