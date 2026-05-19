"""Lesson 09 · Conditional edges — branching + bounded cycles.

A self-improving writer: it drafts, scores, and either accepts or
revises. A hard cap prevents infinite loops.

    START → draft → score → (good → END | bad → revise → score → …)

Run:
    uv run python -m lessons.09_conditional_edges.example
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from shared import get_llm
from shared.pretty import console, section


MAX_REVISIONS = 3


class State(TypedDict):
    topic: str
    draft: str
    score: int
    revisions: int


def draft_node(state: State) -> dict:
    llm = get_llm()
    if state.get("draft"):
        prompt = (
            f"The previous draft scored {state['score']}/10. Rewrite it to score higher.\n\n"
            f"Topic: {state['topic']}\nPrevious draft:\n{state['draft']}"
        )
    else:
        prompt = f"Write a 3-sentence summary of: {state['topic']}"
    return {"draft": llm.invoke(prompt).content, "revisions": state.get("revisions", 0) + 1}


def score_node(state: State) -> dict:
    llm = get_llm()
    prompt = (
        "On a scale of 1-10, rate the following paragraph for clarity AND vividness. "
        "Reply with just the integer, no commentary.\n\n"
        f"{state['draft']}"
    )
    text = llm.invoke(prompt).content.strip()
    try:
        score = int("".join(c for c in text if c.isdigit())[:2] or 0)
    except ValueError:
        score = 0
    return {"score": score}


def route(state: State) -> Literal["revise", "done"]:
    """Conditional edge: 'done' if good enough OR cap reached, else 'revise'."""
    if state["score"] >= 8 or state["revisions"] >= MAX_REVISIONS:
        return "done"
    return "revise"


def build_graph():
    g = StateGraph(State)
    g.add_node("draft", draft_node)
    g.add_node("score", score_node)

    g.add_edge(START, "draft")
    g.add_edge("draft", "score")
    g.add_conditional_edges("score", route, {"revise": "draft", "done": END})

    return g.compile()


def main() -> None:
    section("Lesson 09 · conditional edges + bounded cycles")
    graph = build_graph()
    console.print(graph.get_graph().draw_ascii())

    final = graph.invoke({"topic": "why bicycles still beat cars in cities"})
    console.print(f"\n[bold]Final draft (after {final['revisions']} revision(s), "
                  f"score = {final['score']}/10):[/]\n")
    console.print(final["draft"])


if __name__ == "__main__":
    main()
