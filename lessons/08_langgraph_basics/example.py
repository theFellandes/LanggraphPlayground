"""Lesson 08 · LangGraph basics — your first StateGraph.

We model a two-step "write then critique" pipeline as a graph:

    START → draft → critique → END

Run:
    uv run python -m lessons.08_langgraph_basics.example
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from shared import get_llm
from shared.pretty import console, print_state, section


class State(TypedDict):
    topic: str
    draft: str
    critique: str


def draft_node(state: State) -> dict:
    """Write a one-paragraph introduction on `state['topic']`."""
    llm = get_llm()
    prompt = f"Write a one-paragraph intro about: {state['topic']}"
    return {"draft": llm.invoke(prompt).content}


def critique_node(state: State) -> dict:
    """Review the draft and return improvement suggestions."""
    llm = get_llm()
    prompt = (
        f"Critique this paragraph in two bullet points. Be specific.\n\n"
        f"Paragraph:\n{state['draft']}"
    )
    return {"critique": llm.invoke(prompt).content}


def build_graph():
    g = StateGraph(State)
    g.add_node("draft", draft_node)
    g.add_node("critique", critique_node)

    g.add_edge(START, "draft")
    g.add_edge("draft", "critique")
    g.add_edge("critique", END)

    return g.compile()


def main() -> None:
    section("Lesson 08 · first StateGraph")

    graph = build_graph()

    # `.get_graph().draw_ascii()` prints a quick visualisation.
    console.print(graph.get_graph().draw_ascii())

    final = graph.invoke({"topic": "the history of the printing press"})
    print_state("draft",    final["draft"])
    print_state("critique", final["critique"])


if __name__ == "__main__":
    main()
