"""Multi-agent research assistant — pro version.

Topology:

       user
        │
        ▼
     planner                   ← decomposes the question into N subtopics
        │
        │   Send(...) × N      ← fan-out (lesson 30 pattern 1)
        ▼
   researcher #1 ─┐
   researcher #2 ─┤            ← run under LLM_SEM + per-topic lock
   researcher #N ─┘
        │
        ▼
      reduce                   ← merges per-subtopic claim lists
        │
        ▼
      writer                   ← drafts the report
        │
        ▼
      critic ─→ END           ← APPROVED?
        │
        └────→ writer          ← else loop (capped at 3 revisions, lesson 30 pattern 4)

Run:
    uv run python -m projects.research_assistant_pro.graph "your question"
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from operator import add
from pathlib import Path
from typing import Annotated, TypedDict

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from shared import get_llm, settings
from shared.pretty import console, section

from projects.research_assistant_pro.concurrency import (
    LLM_SEM,
    SEARCH_SEM,
    lock_for,
    tavily_breaker,
)

log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


# --- state ------------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    question: str
    locale: str
    vip: bool
    subtopics: list[str]
    subtopic: str
    claims: Annotated[list[str], add]
    draft: str
    critique: str
    revisions: int


# --- tools ------------------------------------------------------------------
async def _stub_search(query: str) -> list[dict]:
    """Offline stub when TAVILY_API_KEY is unset."""
    await asyncio.sleep(0.1)
    return [
        {"url": f"https://example.com/{query.replace(' ', '-')}-{i}", "title": query, "content": f"Stub result {i} for {query}."}
        for i in range(3)
    ]


async def search_web(query: str) -> list[dict]:
    """Tavily search wrapped with semaphore + circuit breaker + stub fallback."""
    if not settings.tavily_api_key:
        return await _stub_search(query)

    if tavily_breaker.is_open():
        return [{"url": "", "title": "search unavailable", "content": "(circuit open)"}]

    async with SEARCH_SEM:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            t = TavilySearchResults(max_results=4, tavily_api_key=settings.tavily_api_key)
            results = await t.ainvoke({"query": query})
            tavily_breaker.record(True)
            return results
        except Exception as e:
            tavily_breaker.record(False)
            log.warning("Tavily failed: %s", e)
            return await _stub_search(query)


# --- nodes ------------------------------------------------------------------
async def planner_node(state: ResearchState) -> dict:
    """Decompose the question into 3-5 subtopics. Robust to non-JSON LLM output."""
    prompt = env.get_template("agents/planner.j2").render(
        agent_name="Planner",
        company="Acme",
        question=state["question"],
    )
    async with LLM_SEM:
        out = await get_llm().ainvoke(prompt)
    text = out.content if hasattr(out, "content") else str(out)

    # Extract JSON array even if the model wrapped it in prose.
    match = re.search(r"\[.*\]", text, flags=re.S)
    try:
        subs = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        subs = []

    if not subs:
        subs = [state["question"]]      # fall back to one giant subtopic

    subs = [str(s).strip() for s in subs if str(s).strip()][:5]
    console.print(f"[bold]planner →[/] {len(subs)} subtopics: {subs}")
    return {"subtopics": subs, "revisions": 0}


def dispatch(state: ResearchState) -> list[Send]:
    """Fan-out: one Send per subtopic."""
    return [
        Send("researcher", {**state, "subtopic": s})
        for s in state["subtopics"]
    ]


async def researcher_node(state: ResearchState) -> dict:
    """One researcher per subtopic.

    Coordination:
      - per-topic lock prevents duplicate work
      - semaphore caps LLM + Tavily concurrency
    """
    subtopic = state["subtopic"]

    async with lock_for(subtopic):
        # Search.
        results = await search_web(subtopic)

        # Render prompt.
        prompt = env.get_template("agents/researcher.j2").render(
            agent_name="Researcher",
            company="Acme",
            subtopic=subtopic,
            locale=state.get("locale", "en-US"),
            vip=state.get("vip", False),
            tier="enterprise",
        )

        # Synthesise claims with the LLM (under sem).
        evidence = "\n".join(f"- ({r.get('url','')}) {r.get('content','')[:200]}" for r in results)
        full = f"{prompt}\n\nEvidence:\n{evidence}\n\nReturn 2-3 cited claims."
        async with LLM_SEM:
            out = await get_llm().ainvoke(full)
        text = out.content if hasattr(out, "content") else str(out)

        # Tag each claim with its subtopic for the writer's section structure.
        tagged = [f"[{subtopic}] {line}" for line in text.splitlines() if line.strip()]
        console.print(f"[dim]  researcher({subtopic[:30]}) → {len(tagged)} claims[/]")
        return {"claims": tagged}


def reduce_node(state: ResearchState) -> dict:
    """No-op: the `add` reducer on `claims` already merged everything."""
    console.print(f"[bold]reduce → {len(state['claims'])} total claims[/]")
    return {}


async def writer_node(state: ResearchState) -> dict:
    body = "\n".join(state["claims"])
    prompt = env.get_template("agents/writer.j2").render(
        agent_name="Writer", company="Acme", word_budget=500,
    )
    async with LLM_SEM:
        out = await get_llm().ainvoke(
            f"{prompt}\n\nQuestion: {state['question']}\n\n"
            f"Working set of cited claims:\n{body}\n\n"
            f"Write the report."
        )
    return {"draft": out.content if hasattr(out, "content") else str(out)}


async def critic_node(state: ResearchState) -> dict:
    prompt = env.get_template("agents/critic.j2").render(
        agent_name="Critic", company="Acme", strict_mode=True,
    )
    async with LLM_SEM:
        out = await get_llm().ainvoke(f"{prompt}\n\nDraft:\n{state['draft']}")
    crit = out.content if hasattr(out, "content") else str(out)
    return {"critique": crit, "revisions": state.get("revisions", 0) + 1}


def route_after_critic(state: ResearchState) -> str:
    """Bounded revision cycle (lesson 30 pattern 4)."""
    if "APPROVED" in (state.get("critique", "") or "").upper():
        return END
    if state.get("revisions", 0) >= 3:
        return END
    return "writer"


# --- graph ------------------------------------------------------------------
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("reduce", reduce_node)
    g.add_node("writer", writer_node)
    g.add_node("critic", critic_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", dispatch, ["researcher"])
    g.add_edge("researcher", "reduce")
    g.add_edge("reduce", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"writer": "writer", END: END})

    return g.compile()


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What are the latest advances in fusion energy in 2026?"
    section(f"Research request: {question}")

    graph = build_graph()
    console.print(graph.get_graph().draw_ascii())

    out = asyncio.run(graph.ainvoke({
        "question": question,
        "locale": "en-US",
        "vip": False,
        "claims": [],
    }))

    section("Final report")
    console.print(out.get("draft", "(no draft produced)"))
    section("Critic verdict")
    console.print(out.get("critique", "(no critique)"))
    console.print(f"\n[dim]revisions: {out.get('revisions', 0)}[/]")


if __name__ == "__main__":
    main()
