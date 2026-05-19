"""Lesson 15 · Subgraphs — composition of stateful graphs.

A parent graph "writes a blog post" by calling a subgraph "draft +
edit a section" twice. The subgraph has its own internal state; we
pass each call's `topic` in and read each result back into a list on
the parent state.

Run:
    uv run python -m lessons.15_subgraphs.example
"""

from typing import Annotated, TypedDict
from operator import add

from langgraph.graph import END, START, StateGraph

from shared import get_llm
from shared.pretty import console, print_state, section


# ---------------------------------------------------------------------------
# Subgraph: produce one polished section, given a topic.
# ---------------------------------------------------------------------------
class SectionState(TypedDict):
    topic: str
    draft: str
    polished: str


def sub_draft(state: SectionState) -> dict:
    return {"draft": get_llm().invoke(f"Write 2 sentences about {state['topic']}.").content}


def sub_polish(state: SectionState) -> dict:
    return {"polished": get_llm().invoke(f"Rewrite this more vividly:\n{state['draft']}").content}


def build_section_graph():
    g = StateGraph(SectionState)
    g.add_node("draft",   sub_draft)
    g.add_node("polish",  sub_polish)
    g.add_edge(START, "draft")
    g.add_edge("draft", "polish")
    g.add_edge("polish", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Parent graph: writes a post by calling the subgraph for each section.
# ---------------------------------------------------------------------------
class PostState(TypedDict):
    title: str
    sections: Annotated[list[str], add]   # reducer: append on update


def write_section(topic: str):
    """Wrap the subgraph as a callable parent node."""

    section_graph = build_section_graph()

    def _node(state: PostState) -> dict:
        out = section_graph.invoke({"topic": topic})
        return {"sections": [out["polished"]]}

    _node.__name__ = f"write_{topic.replace(' ', '_')}"
    return _node


def build_post_graph():
    g = StateGraph(PostState)
    g.add_node("intro_section",  write_section("the history of sourdough"))
    g.add_node("middle_section", write_section("the chemistry of sourdough fermentation"))
    g.add_edge(START, "intro_section")
    g.add_edge("intro_section", "middle_section")
    g.add_edge("middle_section", END)
    return g.compile()


def main() -> None:
    section("Lesson 15 · subgraph composition")

    graph = build_post_graph()
    console.print(graph.get_graph().draw_ascii())

    final = graph.invoke({"title": "Sourdough: a small history", "sections": []})
    print_state("Final post", final)


if __name__ == "__main__":
    main()
