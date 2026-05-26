"""Lesson 30 · Advanced graph patterns — runnable, offline demos.

Run:
    uv run python -m lessons.30_advanced_graphs.example
    uv run python -m lessons.30_advanced_graphs.example --fan-out
    uv run python -m lessons.30_advanced_graphs.example --map-reduce
    uv run python -m lessons.30_advanced_graphs.example --bounded-cycle
    uv run python -m lessons.30_advanced_graphs.example --retry
    uv run python -m lessons.30_advanced_graphs.example --breaker
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from shared.pretty import console, section


# --- Fake LLM ---------------------------------------------------------------
async def fake_llm(prompt: str, fail_rate: float = 0.0) -> str:
    """Returns a stub. Optionally throws to test retry/breaker logic."""
    await asyncio.sleep(0.05 + random.uniform(0, 0.05))
    if random.random() < fail_rate:
        raise RuntimeError("simulated upstream failure")
    return f"⟨answer for: {prompt[:40]}⟩"


# --- Pattern 1 · fan-out ----------------------------------------------------
class FanOutState(TypedDict):
    topic: str
    subtopics: list[str]
    summaries: Annotated[list[str], add]      # ← reducer
    final: str


def demo_fan_out() -> None:
    section("1 · Parallel fan-out with Send")

    def plan(state):
        return {"subtopics": ["history", "tech", "economics", "policy"]}

    def dispatch(state):
        return [Send("research", {"subtopic": s}) for s in state["subtopics"]]

    async def research(state):
        s = state["subtopic"]
        out = await fake_llm(f"research: {s}")
        return {"summaries": [f"{s}: {out}"]}

    def reduce(state):
        return {"final": "\n".join(state["summaries"])}

    g = StateGraph(FanOutState)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("reduce", reduce)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", dispatch, ["research"])
    g.add_edge("research", "reduce")
    g.add_edge("reduce", END)

    graph = g.compile()
    t0 = time.perf_counter()
    out = asyncio.run(graph.ainvoke({"topic": "fusion energy"}))
    elapsed = time.perf_counter() - t0
    console.print(f"[bold]Time:[/] {elapsed:.2f}s  (4 branches in parallel)")
    console.print(out["final"])


# --- Pattern 2 · map-reduce with bounded concurrency ------------------------
def demo_map_reduce() -> None:
    section("2 · Map-reduce with a semaphore (concurrency capped at 3)")

    sem = asyncio.Semaphore(3)

    class State(TypedDict):
        items: list[str]
        results: Annotated[list[str], add]
        merged: str

    def plan(state):
        return {"items": [f"doc-{i}" for i in range(10)]}

    def dispatch(state):
        return [Send("process", {"item": x}) for x in state["items"]]

    async def process(state):
        async with sem:
            out = await fake_llm(f"summarise {state['item']}", fail_rate=0)
            return {"results": [out]}

    def merge(state):
        return {"merged": f"merged {len(state['results'])} summaries"}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("process", process)
    g.add_node("merge", merge)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", dispatch, ["process"])
    g.add_edge("process", "merge")
    g.add_edge("merge", END)

    t0 = time.perf_counter()
    out = asyncio.run(g.compile().ainvoke({}))
    elapsed = time.perf_counter() - t0
    console.print(f"[bold]Time:[/] {elapsed:.2f}s for 10 items, semaphore=3")
    console.print(out["merged"])


# --- Pattern 4 · bounded cycle ----------------------------------------------
def demo_bounded_cycle() -> None:
    section("4 · Bounded self-correction (max 3 loops)")

    class State(TypedDict):
        draft: str
        critique: str
        revisions: int

    async def writer(state):
        rev = state.get("revisions", 0)
        return {"draft": f"draft attempt {rev + 1}", "revisions": rev + 1}

    async def critic(state):
        # Approve on attempt 3 to demonstrate the cycle.
        if state["revisions"] >= 3:
            return {"critique": "APPROVED"}
        return {"critique": "needs more detail"}

    def route(state) -> str:
        if state["revisions"] >= 3:
            return END
        if "APPROVED" in state.get("critique", ""):
            return END
        return "writer"

    g = StateGraph(State)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_edge(START, "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route, {"writer": "writer", END: END})

    out = asyncio.run(g.compile().ainvoke({"draft": "", "critique": "", "revisions": 0}))
    console.print(f"[bold]Final revisions: {out['revisions']}[/]   critique={out['critique']!r}")


# --- Pattern 5 · retry with backoff -----------------------------------------
def demo_retry() -> None:
    section("5 · Retry with exponential backoff (tenacity)")

    try:
        from tenacity import retry, stop_after_attempt, wait_exponential
    except ImportError:
        console.print("[yellow]Install tenacity: uv add tenacity[/]")
        return

    attempts = {"n": 0}

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=0.1, max=1))
    async def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError(f"fail #{attempts['n']}")
        return "ok"

    async def node(state):
        out = await flaky()
        return {"result": out}

    g = StateGraph(TypedDict("S", {"result": str}))
    g.add_node("flaky", node)
    g.add_edge(START, "flaky")
    g.add_edge("flaky", END)

    t0 = time.perf_counter()
    out = asyncio.run(g.compile().ainvoke({}))
    elapsed = time.perf_counter() - t0
    console.print(f"Took {elapsed:.2f}s with {attempts['n']} attempts → {out['result']!r}")


# --- Pattern 6 · circuit breaker --------------------------------------------
class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 2.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.fails = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.time() - self.opened_at > self.cooldown:
            self.opened_at = None
            self.fails = 0
            return False
        return True

    def record(self, ok: bool) -> None:
        if ok:
            self.fails = 0
        else:
            self.fails += 1
            if self.fails >= self.threshold:
                self.opened_at = time.time()


def demo_breaker() -> None:
    section("6 · Circuit breaker — fail fast when upstream is down")

    breaker = CircuitBreaker(threshold=3, cooldown=2.0)

    async def call_external():
        # Always fails — represents a hard-down dependency.
        await asyncio.sleep(0.1)
        raise RuntimeError("upstream is down")

    async def node(state):
        i = state.get("i", 0)
        if breaker.is_open():
            return {"i": i + 1, "log": state["log"] + [f"{i}: SKIP (circuit open)"]}
        try:
            await call_external()
            breaker.record(True)
            return {"i": i + 1, "log": state["log"] + [f"{i}: OK"]}
        except Exception as e:
            breaker.record(False)
            return {"i": i + 1, "log": state["log"] + [f"{i}: FAIL ({e})"]}

    State = TypedDict("S", {"i": int, "log": list})
    g = StateGraph(State)
    g.add_node("call", node)
    g.add_edge(START, "call")
    g.add_conditional_edges("call", lambda s: "call" if s["i"] < 8 else END, {"call": "call", END: END})

    out = asyncio.run(g.compile().ainvoke({"i": 0, "log": []}))
    for line in out["log"]:
        console.print(f"  {line}")


# --- entry point ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fan-out", action="store_true")
    parser.add_argument("--map-reduce", action="store_true")
    parser.add_argument("--bounded-cycle", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--breaker", action="store_true")
    args = parser.parse_args()

    selected = []
    if args.fan_out:        selected.append(demo_fan_out)
    if args.map_reduce:     selected.append(demo_map_reduce)
    if args.bounded_cycle:  selected.append(demo_bounded_cycle)
    if args.retry:          selected.append(demo_retry)
    if args.breaker:        selected.append(demo_breaker)
    if not selected:
        selected = [demo_fan_out, demo_map_reduce, demo_bounded_cycle, demo_retry, demo_breaker]

    for fn in selected:
        fn()


if __name__ == "__main__":
    main()
