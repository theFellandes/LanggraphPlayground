"""Lesson 38 · Reasoning models + routing — runnable demos.

Three demos:
    --thinking   Claude extended thinking at two budget levels
    --route      heuristic router dispatches to cheap vs reasoning tier
    --cascade    cheap → reasoning escalation pattern in LangGraph

Run:
    uv run python -m lessons.38_reasoning_and_routing.example
    uv run python -m lessons.38_reasoning_and_routing.example --thinking
    uv run python -m lessons.38_reasoning_and_routing.example --route
    uv run python -m lessons.38_reasoning_and_routing.example --cascade
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from shared import get_llm
from shared.pretty import console, section


REASONING_KEYWORDS = re.compile(
    r"\b(solve|prove|refactor|debug|step[- ]by[- ]step|plan|"
    r"calculate|derive|optimi[sz]e|algorithm|complexity)\b",
    re.IGNORECASE,
)


def needs_reasoning(query: str) -> bool:
    """Heuristic router — fast and bad. Real production uses semantic-router or RouteLLM."""
    return bool(REASONING_KEYWORDS.search(query)) or len(query) > 200


# --- Demo 1 · extended thinking --------------------------------------------
def demo_thinking() -> None:
    section("Claude extended thinking · two budget levels")
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        console.print("[yellow]Missing langchain-anthropic. Install: uv add langchain-anthropic[/]")
        return

    problem = (
        "How many distinct prime factors does 30030 have? "
        "Show your reasoning step by step in the final answer."
    )

    for budget in (2000, 8000):
        try:
            llm = ChatAnthropic(
                model="claude-sonnet-4-6",
                thinking={"type": "enabled", "budget_tokens": budget},
                max_tokens=budget + 4000,
            )
            t0 = time.perf_counter()
            reply = llm.invoke(problem)
            elapsed = time.perf_counter() - t0
            content = reply.content if isinstance(reply.content, str) else _extract_text(reply.content)
            console.rule(f"[bold]budget = {budget} tokens[/]")
            console.print(f"  elapsed: {elapsed:.1f}s")
            console.print(f"  answer:  {content[:400]}")
        except Exception as e:
            console.print(f"[red]thinking call failed:[/] {type(e).__name__}: {e}")
            console.print("[dim](This model requires API access to the extended-thinking feature.)[/]")


def _extract_text(content) -> str:
    """Pull plain text out of a list-of-blocks content payload."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


# --- Demo 2 · routing -------------------------------------------------------
def demo_route() -> None:
    section("Router · cheap vs reasoning tier per query")
    queries = [
        "What's the capital of Belgium?",
        "Refactor this Python function to remove the nested loop and reduce complexity",
        "Translate 'good morning' to German",
        "Prove that the sum of the first n odd integers is n squared",
        "Summarise the company's PTO policy in one sentence",
    ]
    for q in queries:
        tier = "reasoning" if needs_reasoning(q) else "cheap"
        symbol = "🧠" if tier == "reasoning" else "⚡"
        console.print(f"  {symbol} [bold]{tier:9}[/]  {q[:90]}")


# --- Demo 3 · cascading cheap → reasoning ----------------------------------
class CascadeState(TypedDict):
    query: str
    tier: Literal["cheap", "reasoning"]
    answer: str
    confidence: str


UNCERTAINTY_RE = re.compile(
    r"\b(I'm not sure|I don't (know|have enough)|unclear|cannot determine|uncertain)\b",
    re.IGNORECASE,
)


def cheap_node(state):
    out = get_llm().invoke(
        f"Answer concisely. If you're not certain, say 'I'm not sure'.\n\nQ: {state['query']}"
    )
    text = out.content if hasattr(out, "content") else str(out)
    return {"answer": text, "tier": "cheap"}


def grade_node(state):
    if UNCERTAINTY_RE.search(state["answer"]):
        return {"confidence": "low"}
    return {"confidence": "high"}


def reasoning_node(state):
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            thinking={"type": "enabled", "budget_tokens": 4000},
            max_tokens=8000,
        )
        out = llm.invoke(state["query"])
        text = out.content if isinstance(out.content, str) else _extract_text(out.content)
        return {"answer": text, "tier": "reasoning"}
    except Exception:
        # Fall back to a vanilla Sonnet call if extended-thinking unavailable.
        out = get_llm().invoke(state["query"])
        text = out.content if hasattr(out, "content") else str(out)
        return {"answer": text, "tier": "reasoning"}


def pick(state) -> str:
    return "reasoning" if state["confidence"] == "low" else END


def demo_cascade() -> None:
    section("Cascade · cheap first, escalate when uncertain")

    g = StateGraph(CascadeState)
    g.add_node("cheap", cheap_node)
    g.add_node("grade", grade_node)
    g.add_node("reasoning", reasoning_node)
    g.add_edge(START, "cheap")
    g.add_edge("cheap", "grade")
    g.add_conditional_edges("grade", pick, {"reasoning": "reasoning", END: END})
    g.add_edge("reasoning", END)
    graph = g.compile()

    queries = [
        "What's the capital of Belgium?",                     # cheap should handle
        "Prove that the sum of the first n odd integers is n squared",  # reasoning likely
    ]
    for q in queries:
        out = graph.invoke({"query": q, "tier": "cheap", "answer": "", "confidence": ""})
        console.rule(f"[bold]Q:[/] {q}")
        console.print(f"  tier used: [bold]{out['tier']}[/]")
        console.print(f"  answer:    {out['answer'][:300]}")


# --- entry ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--route", action="store_true")
    parser.add_argument("--cascade", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.thinking: selected.append(demo_thinking)
    if args.route:    selected.append(demo_route)
    if args.cascade:  selected.append(demo_cascade)
    if not selected:
        # Route is the only zero-API-cost demo; do it first.
        selected = [demo_route, demo_cascade, demo_thinking]

    for fn in selected:
        try:
            fn()
        except Exception as e:
            console.print(f"[red]demo failed:[/] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
