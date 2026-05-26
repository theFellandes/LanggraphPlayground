"""Customer-support bot — pro version.

Layers on top of the simpler `customer_support_bot`:
  - Per-customer asyncio lock map (lesson 27)
  - Jinja-templated, dynamically-rendered system prompt (lesson 28)
  - Idempotency-keyed refund tool (lesson 31)
  - Bounded LLM concurrency
  - Sqlite/Postgres backend toggle via env var

Run:
    uv run python -m projects.customer_support_bot_pro.graph
    SUPPORT_BOT_BACKEND=postgres uv run python -m projects.customer_support_bot_pro.graph

CLI commands:
    /login <email>     switch customer
    /approve / /reject resume a pending interrupt
    /quit              exit
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
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

PROMPTS_DIR = Path(__file__).parent / "prompts"
PERSIST_DIR = settings.data_dir / "chroma_support_bot_pro"

env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


# --- concurrency primitives -------------------------------------------------
_customer_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
LLM_SEM = asyncio.Semaphore(8)


# --- in-memory idempotency DB (replace with real DB in production) ----------
_refund_db: dict[str, dict] = {}


# --- RAG retriever ----------------------------------------------------------
def _build_retriever():
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


# --- tools ------------------------------------------------------------------
@tool
def lookup_handbook(question: str) -> str:
    """Look up the answer to a policy question in the company handbook."""
    docs = retriever.invoke(question)
    return "\n\n---\n\n".join(d.page_content for d in docs) or "(no relevant policy found)"


@tool
def request_refund(order_id: str, amount: float, reason: str, idempotency_key: str) -> dict:
    """Issue a refund. Same idempotency_key returns the same receipt — safe to retry.

    Refunds over $100 require human approval (HITL pause).
    """
    if cached := _refund_db.get(idempotency_key):
        return {"status": "replay", "receipt": cached}

    if amount > 100:
        decision = interrupt({
            "type": "refund_approval",
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
            "idempotency_key": idempotency_key,
        })
        if str(decision).lower() not in {"approve", "approved", "yes"}:
            result = {"status": "rejected", "reason": str(decision)}
            _refund_db[idempotency_key] = result
            return result

    receipt = f"R-{idempotency_key}"
    result = {"status": "approved", "receipt": receipt, "amount": amount, "order_id": order_id}
    _refund_db[idempotency_key] = result
    return result


@tool
def escalate_ticket(summary: str, severity: str = "p3") -> dict:
    """Escalate to a human support agent. Returns a ticket id."""
    # Recoverable error contract: validate, return structured failures
    if severity not in {"p1", "p2", "p3", "p4"}:
        return {"error": "recoverable", "message": f"severity must be one of p1..p4, got {severity!r}"}
    return {"ticket_id": f"T-{abs(hash(summary)) % 10000:04d}", "severity": severity, "status": "open"}


# --- dynamic system prompt --------------------------------------------------
def system_for_turn(state, runtime) -> str:
    customer = runtime.context.get("customer", {"id": "unknown", "name": "guest", "tier": "free", "locale": "en-US"})
    return env.get_template("agents/support.j2").render(
        agent_name="Acme Support",
        company="Acme",
        customer=customer,
        tier=customer.get("tier", "free"),
        locale=customer.get("locale", "en-US"),
    )


# --- agent factory ----------------------------------------------------------
def build_agent():
    backend = os.environ.get("SUPPORT_BOT_BACKEND", "sqlite").lower()

    if backend == "postgres":
        # Caller must close the context manager. Postgres URL from settings.
        from langgraph.checkpoint.postgres import PostgresSaver
        cp_ctx = PostgresSaver.from_conn_string(settings.postgres_url)
        cp = cp_ctx.__enter__()
        cp.setup()
    else:
        db_path = settings.data_dir / "support_bot_pro.sqlite"
        cp = SqliteSaver.from_conn_string(str(db_path)).__enter__()

    return create_agent(
        model=get_llm(),
        tools=[lookup_handbook, request_refund, escalate_ticket],
        system_prompt=system_for_turn,
        middleware=[SummarizationMiddleware(model=get_llm(), trigger=("tokens", 1500))],
        checkpointer=cp,
    )


# --- CLI --------------------------------------------------------------------
DEFAULT_CUSTOMERS = {
    "alice@acme.com":  {"id": "alice@acme.com", "name": "Alice",   "tier": "enterprise", "locale": "en-US"},
    "bob@globex.com":  {"id": "bob@globex.com", "name": "Bob",     "tier": "pro",         "locale": "tr-TR"},
    "guest":           {"id": "guest",          "name": "Guest",   "tier": "free",        "locale": "en-US"},
}


async def handle(agent, customer: dict, message: str, pending) -> tuple[dict, object | None]:
    """Process one message under the per-customer lock + LLM semaphore."""
    cfg = {
        "configurable": {"thread_id": f"thread-{customer['id']}"},
        "context": {"customer": customer},
    }

    async with _customer_locks[customer["id"]]:
        async with LLM_SEM:
            if pending and message in {"/approve", "/reject"}:
                decision = "approve" if message == "/approve" else "reject"
                result = await agent.ainvoke(Command(resume=decision), cfg)
                return result, None
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": message}]}, cfg,
            )

    interrupts = result.get("__interrupt__", [])
    return result, interrupts[0] if interrupts else None


async def cli() -> None:
    section("Acme support bot (pro) — /login <email>  /approve  /reject  /quit")
    agent = build_agent()
    customer = DEFAULT_CUSTOMERS["guest"]
    pending = None
    console.print(f"[dim]logged in as[/] [bold]{customer['id']}[/]  ({customer['tier']})")

    loop = asyncio.get_event_loop()
    while True:
        try:
            user = await loop.run_in_executor(None, sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            print()
            break
        user = (user or "").strip()
        if not user:
            continue
        if user in {"/quit", "/exit"}:
            break
        if user.startswith("/login "):
            email = user.split(maxsplit=1)[1].strip()
            customer = DEFAULT_CUSTOMERS.get(email, {"id": email, "name": email, "tier": "free", "locale": "en-US"})
            console.print(f"[dim]now logged in as[/] [bold]{customer['id']}[/]  ({customer['tier']})")
            pending = None
            continue

        result, pending = await handle(agent, customer, user, pending)

        if pending:
            console.print(
                f"[yellow]⚠  HITL needed:[/] {pending.value}\n"
                f"  Type [bold green]/approve[/] or [bold red]/reject[/]."
            )
        else:
            console.print(f"[bold green]bot:[/] {result['messages'][-1].content}")


def main() -> None:
    asyncio.run(cli())


if __name__ == "__main__":
    main()
