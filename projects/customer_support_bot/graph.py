"""Customer-support bot with HITL escalation + Sqlite persistence.

Architecture:
  - `create_agent` (lesson 10) drives the conversation
  - `summarization_middleware` (lesson 11) keeps long chats under context
  - The `request_refund` tool calls `interrupt()` so a human approves it
  - `SqliteSaver` (lesson 12) persists state per thread_id

Run an interactive CLI session:
    uv run python -m projects.customer_support_bot.graph

Inside the chat:
    > How many PTO days do I get?           (RAG-answered, no escalation)
    > I want a refund of $250 on order #99  (triggers HITL)
    > /approve                              (resumes with approval)
    > /reject                               (resumes with rejection)
    > /quit                                 (exit)
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

from shared import get_llm, settings
from shared.pretty import console, section


PERSIST_DIR = settings.data_dir / "chroma_support_bot"


def _build_retriever():
    """Index the company handbook for FAQ retrieval."""
    if not (PERSIST_DIR / "chroma.sqlite3").exists():
        docs = TextLoader(
            str(settings.data_dir / "sample_docs" / "company_handbook.md"),
            encoding="utf-8",
        ).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60).split_documents(docs)
        Chroma.from_documents(
            chunks,
            embedding=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
            persist_directory=str(PERSIST_DIR),
            collection_name="support",
        )
    store = Chroma(
        embedding_function=FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5"),
        persist_directory=str(PERSIST_DIR),
        collection_name="support",
    )
    return store.as_retriever(search_kwargs={"k": 3})


retriever = _build_retriever()


@tool
def lookup_handbook(question: str) -> str:
    """Look up the answer to a policy question in the company handbook."""
    docs = retriever.invoke(question)
    return "\n\n---\n\n".join(d.page_content for d in docs) or "(no relevant policy found)"


@tool
def request_refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund. Refunds over $100 require human approval."""
    if amount > 100:
        decision = interrupt(
            {
                "type": "refund_approval",
                "order_id": order_id,
                "amount": amount,
                "reason": reason,
            }
        )
        if str(decision).lower() not in {"approve", "approved", "yes"}:
            return f"Refund of ${amount:.2f} REJECTED by human reviewer ({decision})."
    return f"Refund of ${amount:.2f} for order {order_id} processed. Reason: {reason}."


def build_agent():
    db_path = settings.data_dir / "support_bot.sqlite"
    cp = SqliteSaver.from_conn_string(str(db_path)).__enter__()  # caller closes
    return create_agent(
        model=get_llm(),
        tools=[lookup_handbook, request_refund],
        system_prompt=(
            "You are Acme's customer-support bot. Use lookup_handbook for "
            "policy questions; use request_refund for refund requests. "
            "Be concise and warm."
        ),
        middleware=[SummarizationMiddleware(model=get_llm(), trigger=("tokens", 1500))],
        checkpointer=cp,
    )


def cli() -> None:
    section("Acme support bot — type /quit to exit")
    agent = build_agent()
    cfg = {"configurable": {"thread_id": "cli-session"}}

    pending_interrupt = None

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in {"/quit", "/exit"}:
            break

        if pending_interrupt and user in {"/approve", "/reject"}:
            decision = "approve" if user == "/approve" else "reject"
            result = agent.invoke(Command(resume=decision), cfg)
            pending_interrupt = None
        else:
            result = agent.invoke({"messages": [{"role": "user", "content": user}]}, cfg)

        interrupts = result.get("__interrupt__", [])
        if interrupts:
            pending_interrupt = interrupts[0]
            console.print(
                f"[yellow]⚠  HITL needed:[/] {pending_interrupt.value}"
                f"\n  Type [bold green]/approve[/] or [bold red]/reject[/]."
            )
            continue

        console.print(f"[bold green]bot:[/] {result['messages'][-1].content}")


if __name__ == "__main__":
    cli()
