"""LangGraph definition for rag_qa_api_pro.

Pipeline:
    rewrite → retrieve → grade → (fallback if low recall) → generate → cite → END

Coordination:
    - Vector store choice via VECTOR_BACKEND={pgvector|qdrant}
    - Redis SETNX lock for index-rebuild across replicas
    - Jinja-templated prompts (version pinned via PROMPT_VERSION)
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

import redis.asyncio as aioredis
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from projects.rag_qa_api_pro.locks import acquire, new_owner, release, wait_for_release
from shared import get_llm, settings

log = logging.getLogger(__name__)

VECTOR_BACKEND = os.environ.get("VECTOR_BACKEND", "pgvector").lower()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v1")
COLLECTION = "rag_qa_pro"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

PROMPTS_DIR = Path(__file__).parent / "prompts"
SOURCE_DIR = settings.data_dir / "sample_docs"

env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


# --- state ------------------------------------------------------------------
class QAState(MessagesState):
    tenant_id: str
    tier: str
    rewrites: list[str]
    candidates: list
    relevant: list
    cited_answer: str


# --- vector store factories -------------------------------------------------
def _embedder():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def _load_chunks(tenant_id: str):
    loader = DirectoryLoader(
        str(SOURCE_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    for d in docs:
        d.metadata["tenant_id"] = tenant_id
        d.metadata["source_file"] = Path(d.metadata.get("source", "")).name
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)


def _build_pg_retriever(initial_docs):
    from langchain_postgres.vectorstores import PGVector

    return PGVector.from_documents(
        documents=initial_docs,
        embedding=_embedder(),
        connection=settings.postgres_url.replace("postgresql://", "postgresql+psycopg://"),
        collection_name=COLLECTION,
        use_jsonb=True,
        pre_delete_collection=False,
    ).as_retriever(search_kwargs={"k": 6})


def _build_qdrant_retriever(initial_docs):
    from langchain_qdrant import QdrantVectorStore

    return QdrantVectorStore.from_documents(
        documents=initial_docs,
        embedding=_embedder(),
        url=QDRANT_URL,
        collection_name=COLLECTION,
        force_recreate=False,
    ).as_retriever(search_kwargs={"k": 6})


# --- index lifecycle (with distributed lock) --------------------------------
async def ensure_index(redis_client, tenant_id: str):
    """Idempotently make sure the vector store is populated.

    Cross-replica safe: only one process re-indexes at a time; others wait.
    """
    key = f"idx_rebuild:{COLLECTION}:{tenant_id}"
    owner = new_owner()
    got = await acquire(redis_client, key, owner, ttl_seconds=120)
    if not got:
        log.info("Another replica is rebuilding; waiting...")
        await wait_for_release(redis_client, key, max_wait=120)
        return _build_retriever_only()        # other replica finished; just connect

    try:
        chunks = _load_chunks(tenant_id)
        log.info("Rebuilding index with %d chunks (backend=%s)", len(chunks), VECTOR_BACKEND)
        if VECTOR_BACKEND == "qdrant":
            return _build_qdrant_retriever(chunks)
        return _build_pg_retriever(chunks)
    finally:
        await release(redis_client, key, owner)


def _build_retriever_only():
    """Connect to an already-indexed store without re-uploading documents."""
    if VECTOR_BACKEND == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        return QdrantVectorStore.from_existing_collection(
            embedding=_embedder(),
            url=QDRANT_URL,
            collection_name=COLLECTION,
        ).as_retriever(search_kwargs={"k": 6})
    from langchain_postgres.vectorstores import PGVector
    return PGVector(
        embeddings=_embedder(),
        connection=settings.postgres_url.replace("postgresql://", "postgresql+psycopg://"),
        collection_name=COLLECTION,
        use_jsonb=True,
    ).as_retriever(search_kwargs={"k": 6})


# --- nodes ------------------------------------------------------------------
async def rewrite_node(state: QAState) -> dict:
    question = state["messages"][-1].content
    prompt = env.get_template(f"rewriter.{PROMPT_VERSION}.j2").render(question=question)
    out = await get_llm().ainvoke(prompt)
    txt = out.content if hasattr(out, "content") else str(out)
    match = re.search(r"\[.*\]", txt, flags=re.S)
    try:
        rewrites = json.loads(match.group(0)) if match else [question]
    except json.JSONDecodeError:
        rewrites = [question]
    rewrites = [str(r).strip() for r in rewrites if str(r).strip()][:3] or [question]
    return {"rewrites": rewrites}


async def retrieve_node(state: QAState) -> dict:
    retriever = state.get("_retriever") or _build_retriever_only()
    seen: dict[str, object] = {}
    for q in state["rewrites"]:
        for d in await retriever.ainvoke(q):
            seen[d.page_content[:100]] = d
    return {"candidates": list(seen.values())}


async def grade_node(state: QAState) -> dict:
    question = state["messages"][-1].content
    keep = []
    grader_tpl = env.get_template(f"grader.{PROMPT_VERSION}.j2")
    for d in state["candidates"]:
        prompt = grader_tpl.render(question=question, chunk=d.page_content)
        verdict = await get_llm().ainvoke(prompt)
        txt = (verdict.content if hasattr(verdict, "content") else str(verdict)).strip().upper()
        if txt.startswith("Y"):
            keep.append(d)
    return {"relevant": keep[:4]}


async def generate_node(state: QAState) -> dict:
    question = state["messages"][-1].content
    docs = state.get("relevant") or state.get("candidates", [])[:4]

    prompt = env.get_template(f"qa.{PROMPT_VERSION}.j2").render(
        tenant_id=state.get("tenant_id", "default"),
        tier=state.get("tier", "free"),
        docs=docs,
        question=question,
    )
    reply = await get_llm().ainvoke(prompt)
    return {"messages": [reply]}


def cite_node(state: QAState) -> dict:
    docs = state.get("relevant") or state.get("candidates", [])[:4]
    sources = {d.metadata.get("source_file", "unknown") for d in docs}
    last = state["messages"][-1]
    body = last.content + "\n\nSources: " + ", ".join(sorted(sources))
    last_copy = last.model_copy(update={"content": body})
    return {"messages": [last_copy], "cited_answer": body}


# --- graph ------------------------------------------------------------------
def build_graph_definition() -> StateGraph:
    g = StateGraph(QAState)
    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("generate", generate_node)
    g.add_node("cite", cite_node)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_edge("grade", "generate")
    g.add_edge("generate", "cite")
    g.add_edge("cite", END)
    return g


@asynccontextmanager
async def compiled_graph():
    """App lifespan helper. Sets up checkpointer + ensures index + yields graph."""
    redis_client = aioredis.from_url(REDIS_URL)
    try:
        # Ensure index exists (cross-replica safe).
        await ensure_index(redis_client, tenant_id="default")

        async with AsyncPostgresSaver.from_conn_string(settings.postgres_url) as cp:
            await cp.setup()
            yield build_graph_definition().compile(checkpointer=cp)
    finally:
        await redis_client.aclose()
