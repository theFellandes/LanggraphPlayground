"""Lesson 18 · Long-term memory — the `Store` API.

Checkpointers (lesson 12) persist *within* a thread.
The `Store` persists *across* threads — your user-level memory.

Run:
    uv run python -m lessons.18_long_term_memory.example
"""

from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from shared import get_llm
from shared.pretty import console, print_state, section


class State(TypedDict):
    user_id: str
    user_message: str
    reply: str


def remember_node(state: State, *, store) -> dict:
    """Extract any user 'fact' and write it to the store under their namespace."""
    namespace = ("memories", state["user_id"])
    llm = get_llm()
    fact = llm.invoke(
        "If the message contains a personal fact about the user "
        "(e.g. 'I love hiking', 'my dog's name is Rex'), reply with that fact "
        "in one short sentence. Otherwise reply 'NONE'.\n\nMessage: "
        f"{state['user_message']}"
    ).content.strip()

    if fact and fact.upper() != "NONE":
        store.put(namespace, str(uuid4()), {"fact": fact})
        console.print(f"[dim]✚ stored: {fact}[/]")
    return {}


def reply_node(state: State, *, store) -> dict:
    """Use everything we've ever stored about this user to personalise the reply."""
    namespace = ("memories", state["user_id"])
    memories = [item.value["fact"] for item in store.search(namespace)]
    context = "\n".join(f"- {m}" for m in memories) or "(no prior memories)"

    llm = get_llm()
    reply = llm.invoke(
        f"What we know about this user:\n{context}\n\n"
        f"Their message: {state['user_message']}\n\n"
        "Reply warmly, referring to their preferences where relevant."
    ).content
    return {"reply": reply}


def build_graph():
    g = StateGraph(State)
    g.add_node("remember", remember_node)
    g.add_node("reply",    reply_node)
    g.add_edge(START, "remember")
    g.add_edge("remember", "reply")
    g.add_edge("reply", END)
    return g.compile(checkpointer=MemorySaver(), store=InMemoryStore())


def main() -> None:
    section("Lesson 18 · long-term memory via Store")

    graph = build_graph()
    user_id = "alice"

    # Three independent threads — same user. The Store carries facts between them.
    for thread, msg in [
        ("t1", "Hi! I love hiking and I'm vegetarian."),
        ("t2", "Where should I plan my next holiday?"),
        ("t3", "What food should I cook this weekend?"),
    ]:
        cfg = {"configurable": {"thread_id": thread}}
        final = graph.invoke({"user_id": user_id, "user_message": msg}, cfg)
        print_state(f"thread {thread} reply", final["reply"])


if __name__ == "__main__":
    main()
