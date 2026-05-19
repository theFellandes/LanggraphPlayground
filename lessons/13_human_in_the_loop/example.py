"""Lesson 13 · Human-in-the-loop — `interrupt()` + `Command(resume=...)`.

A two-node graph that pauses for approval before "sending an email."

Run:
    uv run python -m lessons.13_human_in_the_loop.example
"""

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from shared import get_llm
from shared.pretty import console, print_state, section


class State(TypedDict):
    request: str
    draft_email: str
    decision: str
    sent: bool


def draft_node(state: State) -> dict:
    """Compose the email."""
    llm = get_llm()
    prompt = f"Draft a short, polite email for this request: {state['request']}"
    return {"draft_email": llm.invoke(prompt).content}


def approval_node(state: State) -> dict:
    """Pause and wait for a human to approve, reject, or edit."""
    decision = interrupt(
        {
            "question": "Approve, reject, or rewrite this draft?",
            "draft":    state["draft_email"],
        }
    )
    # `interrupt` returns whatever the caller passes in `Command(resume=...)`.
    if isinstance(decision, dict) and "edited" in decision:
        return {"decision": "approved", "draft_email": decision["edited"]}
    return {"decision": str(decision)}


def send_node(state: State) -> dict:
    """The 'side effect' — gated behind the approval."""
    if state["decision"] != "approved":
        console.print(f"[red]Skipped — decision was {state['decision']!r}[/]")
        return {"sent": False}
    console.print(f"[green]✉  pretend-sending email:[/]\n{state['draft_email']}")
    return {"sent": True}


def build_graph():
    g = StateGraph(State)
    g.add_node("draft",   draft_node)
    g.add_node("approve", approval_node)
    g.add_node("send",    send_node)
    g.add_edge(START, "draft")
    g.add_edge("draft", "approve")
    g.add_edge("approve", "send")
    g.add_edge("send", END)
    # Checkpointer is required for interrupts to work — state has to survive
    # the pause.
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    section("Lesson 13 · interrupt() + Command(resume=...)")

    graph = build_graph()
    cfg = {"configurable": {"thread_id": "hitl-1"}}

    # --- 1st run pauses at the interrupt and returns the interrupt payload ---
    result = graph.invoke(
        {"request": "Ask our vendor for a 2-week extension on the deliverable."},
        cfg,
    )
    interrupts = result.get("__interrupt__", [])
    console.print("[yellow]Graph paused. Interrupt payload:[/]")
    print_state("interrupt", interrupts[0].value if interrupts else result)

    # --- Pretend a human approved with an edit -------------------------------
    edited = "Hi team — would you be able to grant us a 2-week extension on the deliverable? Thanks!"
    final = graph.invoke(Command(resume={"edited": edited}), cfg)

    section("final state")
    print_state("final", final)


if __name__ == "__main__":
    main()
