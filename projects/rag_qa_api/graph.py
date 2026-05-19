"""LangGraph definition for the RAG Q&A API capstone.

Pipeline:
    user question
        │
        ▼
   retrieve_node  →  uses Chroma + FastEmbed to find relevant chunks
        │
        ▼
   generate_node  →  asks the LLM for a grounded answer
        │
        ▼
       END

Persisted with PostgresSaver — every thread (= chat session) survives
restarts of the API process.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from shared import get_llm, settings

PERSIST_DIR = str(settings.data_dir / "chroma_qa_api")
SOURCE_DIR = settings.data_dir / "sample_docs"


def _build_retriever():
    """Index sample docs into Chroma — idempotent."""
    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    if not (settings.data_dir / "chroma_qa_api" / "chroma.sqlite3").exists():
        loader = DirectoryLoader(
            str(SOURCE_DIR),
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_documents(docs)
        Chroma.from_documents(chunks, embedding=embedding, persist_directory=PERSIST_DIR, collection_name="qa")
    store = Chroma(
        embedding_function=embedding,
        persist_directory=PERSIST_DIR,
        collection_name="qa",
    )
    return store.as_retriever(search_kwargs={"k": 4})


retriever = _build_retriever()


class QAState(MessagesState):
    context: str


def retrieve_node(state: QAState) -> dict:
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return {"context": context}


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer using ONLY the provided context. "
                   "If the context is insufficient, say so explicitly.\n\n"
                   "Context:\n{context}"),
        ("placeholder", "{messages}"),
    ]
)


def generate_node(state: QAState) -> dict:
    chain = PROMPT | get_llm()
    reply = chain.invoke({"context": state["context"], "messages": state["messages"]})
    return {"messages": [reply]}


def build_graph_definition() -> StateGraph:
    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g


@asynccontextmanager
async def compiled_graph():
    """Async context manager that yields a graph compiled with PostgresSaver."""
    async with AsyncPostgresSaver.from_conn_string(settings.postgres_url) as cp:
        await cp.setup()
        yield build_graph_definition().compile(checkpointer=cp)
