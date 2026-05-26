"""Lesson 34 · Observability + tracing — runnable demo.

Builds a tiny graph (retrieve → generate) with the canonical metadata
set attached to every LLM invocation. Emits to whichever backend you
have configured (LangSmith / Langfuse / OTel collector / stdout).

Run:
    uv run python -m lessons.34_observability_tracing.example
    uv run python -m lessons.34_observability_tracing.example --langsmith
    uv run python -m lessons.34_observability_tracing.example --langfuse
    uv run python -m lessons.34_observability_tracing.example --otel

Env vars consulted:
    LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT  (LangSmith)
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST       (Langfuse)
    OTEL_EXPORTER_OTLP_ENDPOINT                                   (any OTel backend)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from shared import get_llm
from shared.pretty import console, section


# --- canonical metadata builder --------------------------------------------
def call_metadata(
    *,
    prompt_name: str,
    prompt_version: str,
    prompt_text: str,
    feature: str,
    user_id: str,
    tenant_id: str = "default",
    user_segment: str = "free",
    ab_variant: str = "control",
) -> dict:
    """The 15-field canonical metadata set (lesson 34, Topic 5)."""
    return {
        "request_id":      str(uuid.uuid4()),
        "user_id_hash":    hashlib.sha256(user_id.encode()).hexdigest()[:12],
        "tenant_id":       tenant_id,
        "user_segment":    user_segment,
        "prompt_name":     prompt_name,
        "prompt_version":  prompt_version,
        "prompt_sha":      hashlib.sha256(prompt_text.encode()).hexdigest()[:16],
        "ab_variant":      ab_variant,
        "feature":         feature,
        # latency / tokens / cost filled in after the call
    }


# --- LangSmith setup --------------------------------------------------------
def configure_langsmith() -> bool:
    if not os.environ.get("LANGSMITH_API_KEY"):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "lesson-34")
    console.print("[green]LangSmith[/] tracing enabled (project=lesson-34)")
    return True


# --- Langfuse setup ---------------------------------------------------------
def configure_langfuse():
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        from langfuse.callback import CallbackHandler
    except ImportError:
        console.print("[yellow]Langfuse not installed: uv add langfuse[/]")
        return None
    handler = CallbackHandler(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    console.print("[green]Langfuse[/] callback handler attached")
    return handler


# --- OTel setup -------------------------------------------------------------
def configure_otel() -> bool:
    try:
        from traceloop.sdk import Traceloop
    except ImportError:
        console.print("[yellow]traceloop-sdk not installed: uv add traceloop-sdk[/]")
        return False
    Traceloop.init(
        app_name="lesson-34",
        api_endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
        disable_batch=True,        # flush immediately so the demo's output is complete
    )
    console.print("[green]OTel[/] auto-instrumentation initialised")
    return True


# --- graph ------------------------------------------------------------------
class State(TypedDict):
    question: str
    context: str
    answer: str
    metadata: dict


def retrieve_node(state: State) -> dict:
    """Fake retrieval — in real life this hits Chroma/pgvector/Qdrant."""
    catalog = {
        "pto": "Full-time employees receive 20 paid time off (PTO) days per year.",
        "refund": "Refunds over $100 require human approval.",
        "remote": "Remote work allowed up to 3 days/week with manager approval.",
    }
    q = state["question"].lower()
    hits = [v for k, v in catalog.items() if k in q]
    return {"context": "\n".join(hits) if hits else "(no relevant policy)"}


def generate_node(state: State) -> dict:
    prompt = (
        f"Use this context to answer concisely:\n{state['context']}\n\n"
        f"Question: {state['question']}"
    )
    meta = call_metadata(
        prompt_name="qa.generate",
        prompt_version="v1",
        prompt_text=prompt,
        feature="support-bot",
        user_id=state["metadata"].get("user_id", "anon"),
        tenant_id=state["metadata"].get("tenant_id", "default"),
        user_segment=state["metadata"].get("user_segment", "free"),
        ab_variant=state["metadata"].get("ab_variant", "control"),
    )

    t0 = time.perf_counter()
    config = {
        "metadata": meta,
        "tags": ["lesson-34", "qa-pipeline", meta["feature"]],
        "run_name": "qa.generate",
    }
    if state["metadata"].get("_langfuse_handler"):
        config["callbacks"] = [state["metadata"]["_langfuse_handler"]]
    reply = get_llm().invoke(prompt, config=config)
    latency_ms = (time.perf_counter() - t0) * 1000

    text = reply.content if hasattr(reply, "content") else str(reply)
    usage = getattr(reply, "usage_metadata", None) or {}
    meta.update({
        "latency_ms":   round(latency_ms, 1),
        "input_tokens":  usage.get("input_tokens", -1),
        "output_tokens": usage.get("output_tokens", -1),
    })
    return {"answer": text, "metadata": {**state["metadata"], "_last_call": meta}}


def build_graph():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


# --- entry ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langsmith", action="store_true")
    parser.add_argument("--langfuse", action="store_true")
    parser.add_argument("--otel", action="store_true")
    args = parser.parse_args()

    section("Configure backends")
    ls_on = configure_langsmith() if (args.langsmith or not any(vars(args).values())) else False
    lf_handler = configure_langfuse() if (args.langfuse or not any(vars(args).values())) else None
    otel_on = configure_otel() if args.otel else False

    if not (ls_on or lf_handler or otel_on):
        console.print("[dim]No backend configured — metadata will only print to stdout.[/]")
        console.print("[dim]Set LANGSMITH_API_KEY or LANGFUSE_PUBLIC_KEY in .env to enable.[/]")

    section("Run the graph with rich metadata")
    graph = build_graph()
    questions = [
        ("alice@acme.com",  "enterprise", "How many PTO days do I get?"),
        ("bob@globex.com",  "pro",        "Can I get a refund for $250?"),
        ("guest",           "free",       "Can I work remote?"),
    ]
    for user_id, segment, q in questions:
        meta = {
            "user_id": user_id,
            "tenant_id": "acme" if "@acme" in user_id else "globex" if "@globex" in user_id else "guest",
            "user_segment": segment,
            "ab_variant": "v1",
            "_langfuse_handler": lf_handler,
        }
        out = graph.invoke({"question": q, "context": "", "answer": "", "metadata": meta})
        last = out["metadata"]["_last_call"]
        console.rule(f"[bold]Q:[/] {q}")
        console.print(f"  [bold]answer:[/] {out['answer'][:200]}")
        console.print("  [bold]metadata:[/]")
        for k, v in last.items():
            console.print(f"    {k:18}  {v}")


if __name__ == "__main__":
    main()
