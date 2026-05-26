"""Lesson 29 · Vector databases — pgvector vs Qdrant, side-by-side.

Prerequisites:
    cd lessons/29_vector_databases
    docker compose up -d     # brings up pgvector, qdrant, redis

    uv sync --extra api      # for langchain-postgres, qdrant-client

Run:
    uv run python -m lessons.29_vector_databases.example
    uv run python -m lessons.29_vector_databases.example --hybrid     # Qdrant hybrid
    uv run python -m lessons.29_vector_databases.example --filter     # tenant filter demo

The script:
  1. Indexes the same corpus into pgvector and Qdrant.
  2. Asks the same five questions of both.
  3. Prints answers + retrieval latency side-by-side.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared import settings
from shared.pretty import console, section

EMBED_MODEL = "BAAI/bge-small-en-v1.5"     # 384-dim
SOURCE_DIR = settings.data_dir / "sample_docs"
QDRANT_URL = "http://localhost:6333"
PG_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/langgraph"

QUESTIONS = [
    "How many PTO days do full-time employees get?",
    "What's the difference between LCEL and LangGraph?",
    "What's the policy on remote work equipment?",
    "What is a StateGraph?",
    "Can I expense a $250 laptop stand?",
]


def _load_chunks():
    loader = DirectoryLoader(
        str(SOURCE_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    # Tag tenant so the filter demo has something to filter on.
    for d in docs:
        d.metadata["tenant_id"] = "acme"
        d.metadata["source_file"] = Path(d.metadata.get("source", "")).name
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    return splitter.split_documents(docs)


def _embedder():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


# --- pgvector ---------------------------------------------------------------
def _build_pgvector_retriever(chunks):
    from langchain_postgres.vectorstores import PGVector

    store = PGVector.from_documents(
        documents=chunks,
        embedding=_embedder(),
        collection_name="lesson29",
        connection=PG_URL,
        use_jsonb=True,
        pre_delete_collection=True,    # idempotent re-runs
    )
    return store.as_retriever(search_kwargs={"k": 4})


# --- Qdrant -----------------------------------------------------------------
def _build_qdrant_retriever(chunks, hybrid: bool = False):
    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    kwargs = dict(
        documents=chunks,
        embedding=_embedder(),
        url=QDRANT_URL,
        collection_name="lesson29" + ("_hybrid" if hybrid else ""),
        force_recreate=True,
    )

    if hybrid:
        from langchain_qdrant import FastEmbedSparse

        kwargs["sparse_embedding"] = FastEmbedSparse(model_name="Qdrant/bm25")
        kwargs["retrieval_mode"] = RetrievalMode.HYBRID

    store = QdrantVectorStore.from_documents(**kwargs)
    return store.as_retriever(search_kwargs={"k": 4})


# --- comparison harness -----------------------------------------------------
def _time_invoke(retriever, q: str) -> tuple[float, list]:
    t0 = time.perf_counter()
    docs = retriever.invoke(q)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, docs


def run_comparison(hybrid: bool) -> None:
    chunks = _load_chunks()
    console.print(f"Indexed [bold]{len(chunks)}[/] chunks into both stores.")

    section("Building retrievers")
    pg = _build_pgvector_retriever(chunks)
    qd = _build_qdrant_retriever(chunks, hybrid=hybrid)

    section(f"Querying both stores · qdrant_hybrid={hybrid}")
    for q in QUESTIONS:
        pg_ms, pg_docs = _time_invoke(pg, q)
        qd_ms, qd_docs = _time_invoke(qd, q)
        console.rule(f"[bold]Q:[/] {q}")
        console.print(
            f"  [bold cyan]pgvector[/]  {pg_ms:6.1f} ms  → top: {pg_docs[0].page_content[:80]!r}"
        )
        console.print(
            f"  [bold magenta]qdrant  [/]  {qd_ms:6.1f} ms  → top: {qd_docs[0].page_content[:80]!r}"
        )


def run_filter_demo() -> None:
    """Demonstrates pre-filtering by tenant on both stores."""
    section("Tenant-filter demo")

    chunks = _load_chunks()
    # Mark half the chunks as a different tenant.
    for i, c in enumerate(chunks):
        c.metadata["tenant_id"] = "acme" if i % 2 == 0 else "globex"

    from langchain_postgres.vectorstores import PGVector
    from langchain_qdrant import QdrantVectorStore

    pg = PGVector.from_documents(
        chunks, _embedder(),
        collection_name="lesson29_filter",
        connection=PG_URL, use_jsonb=True, pre_delete_collection=True,
    )
    qd = QdrantVectorStore.from_documents(
        chunks, _embedder(),
        url=QDRANT_URL, collection_name="lesson29_filter", force_recreate=True,
    )

    q = "What's the refund policy?"

    pg_acme = pg.as_retriever(search_kwargs={"k": 4, "filter": {"tenant_id": "acme"}})
    pg_globex = pg.as_retriever(search_kwargs={"k": 4, "filter": {"tenant_id": "globex"}})

    qd_acme = qd.as_retriever(search_kwargs={
        "k": 4,
        "filter": {"must": [{"key": "metadata.tenant_id", "match": {"value": "acme"}}]},
    })
    qd_globex = qd.as_retriever(search_kwargs={
        "k": 4,
        "filter": {"must": [{"key": "metadata.tenant_id", "match": {"value": "globex"}}]},
    })

    for label, r in [
        ("pgvector · acme  ", pg_acme),
        ("pgvector · globex", pg_globex),
        ("qdrant   · acme  ", qd_acme),
        ("qdrant   · globex", qd_globex),
    ]:
        docs = r.invoke(q)
        tenants = {d.metadata.get("tenant_id") for d in docs}
        console.print(
            f"[bold]{label}[/]  → {len(docs)} docs, tenants seen: {tenants}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid", action="store_true", help="enable Qdrant hybrid (dense+sparse)")
    parser.add_argument("--filter", action="store_true", help="run the tenant-filter demo")
    args = parser.parse_args()

    if args.filter:
        run_filter_demo()
    else:
        run_comparison(hybrid=args.hybrid)


if __name__ == "__main__":
    main()
