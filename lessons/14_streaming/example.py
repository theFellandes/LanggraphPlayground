"""Lesson 14 · Streaming — values / updates / messages / custom + astream_events.

Run:
    uv run python -m lessons.14_streaming.example
"""

import asyncio
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from shared import get_llm
from shared.pretty import console, section


class State(TypedDict):
    topic: str
    intro: str
    body: str


def intro_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"progress": "writing intro…"})
    return {"intro": get_llm().invoke(f"Write a one-line intro about {state['topic']}.").content}


def body_node(state: State) -> dict:
    writer = get_stream_writer()
    writer({"progress": "writing body…"})
    return {"body": get_llm().invoke(f"Write 2 short sentences about {state['topic']}.").content}


def build_graph():
    g = StateGraph(State)
    g.add_node("intro", intro_node)
    g.add_node("body",  body_node)
    g.add_edge(START, "intro")
    g.add_edge("intro", "body")
    g.add_edge("body", END)
    return g.compile()


def show_modes(graph) -> None:
    payload = {"topic": "sourdough bread"}

    for mode in ("values", "updates", "custom"):
        section(f"stream_mode='{mode}'")
        for chunk in graph.stream(payload, stream_mode=mode):
            console.print(chunk)

    section("stream_mode='messages' (token-level LLM tokens)")
    for chunk, meta in graph.stream(payload, stream_mode="messages"):
        if chunk.content:
            console.print(chunk.content, end="")
    console.print()


async def show_events(graph) -> None:
    """astream_events is the fine-grained one — fires on every Runnable lifecycle."""
    section("astream_events (filter for chat-model streaming chunks)")
    async for event in graph.astream_events({"topic": "sourdough bread"}, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                console.print(chunk.content, end="")
    console.print()


def main() -> None:
    graph = build_graph()
    show_modes(graph)
    asyncio.run(show_events(graph))


if __name__ == "__main__":
    main()
