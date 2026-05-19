"""Lesson 12 · Persistence — checkpointers, threads, time-travel.

Same graph compiled with two different checkpointers:
  - MemorySaver — lives in RAM, perfect for tests
  - SqliteSaver — lives on disk, survives restarts

We then demonstrate time-travel: rewind to a past checkpoint and
branch a new conversation from there.

Run:
    uv run python -m lessons.12_persistence.example
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, START, StateGraph

from shared import get_llm, settings
from shared.pretty import console, print_messages, section


def chat_node(state: MessagesState) -> dict:
    """One-turn chat: reply to whatever the user just said."""
    return {"messages": [get_llm().invoke(state["messages"])]}


def build_graph(checkpointer):
    g = StateGraph(MessagesState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.set_finish_point("chat")
    return g.compile(checkpointer=checkpointer)


def memory_demo() -> None:
    section("Part 1 · MemorySaver + threads")

    graph = build_graph(MemorySaver())

    cfg_alice = {"configurable": {"thread_id": "alice"}}
    cfg_bob   = {"configurable": {"thread_id": "bob"}}

    graph.invoke({"messages": [("user", "Hi! My name is Alice.")]},  cfg_alice)
    graph.invoke({"messages": [("user", "Hi! My name is Bob.")]},    cfg_bob)
    graph.invoke({"messages": [("user", "What's my name?")]},         cfg_alice)
    graph.invoke({"messages": [("user", "What's my name?")]},         cfg_bob)

    console.print("[bold cyan]Alice's thread[/]:")
    print_messages(graph.get_state(cfg_alice).values["messages"])
    console.print("\n[bold cyan]Bob's thread[/]:")
    print_messages(graph.get_state(cfg_bob).values["messages"])


def sqlite_demo() -> None:
    """Same graph, same code — only the checkpointer differs."""
    section("Part 2 · SqliteSaver (survives restarts)")

    db = settings.data_dir / "lesson12.sqlite"
    with SqliteSaver.from_conn_string(str(db)) as cp:
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": "persistent"}}
        graph.invoke({"messages": [("user", "Remember the number 42.")]}, cfg)
        graph.invoke({"messages": [("user", "What number did I tell you?")]}, cfg)

        console.print(f"State persisted at [dim]{db}[/]. "
                      "Re-run this lesson — Part 2's thread keeps growing.")
        print_messages(graph.get_state(cfg).values["messages"])


def time_travel_demo() -> None:
    section("Part 3 · time-travel (rewind to a past checkpoint)")

    graph = build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": "timetravel"}}

    graph.invoke({"messages": [("user", "I love hiking.")]},          cfg)
    graph.invoke({"messages": [("user", "Suggest a hiking holiday.")]}, cfg)
    graph.invoke({"messages": [("user", "Now make it a beach holiday instead.")]}, cfg)

    history = list(graph.get_state_history(cfg))
    console.print(f"There are {len(history)} checkpoints in this thread.")

    # Pick the checkpoint right after "Suggest a hiking holiday." — index 2 from
    # the end. (`get_state_history` returns newest first.)
    rewind_to = history[2].config
    console.print(f"Rewinding to checkpoint: {rewind_to['configurable']['checkpoint_id']}")

    # Branch from that point with a different follow-up.
    graph.invoke({"messages": [("user", "Actually, suggest a city break instead.")]}, rewind_to)

    print_messages(graph.get_state(cfg).values["messages"])


def main() -> None:
    memory_demo()
    sqlite_demo()
    time_travel_demo()


if __name__ == "__main__":
    main()
